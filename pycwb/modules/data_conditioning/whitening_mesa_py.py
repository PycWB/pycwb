"""

Pure-Python MESA whitening without ROOT dependencies.

This version produces output compatible with whitening_python() from whitening_py.py,
using the same anchor-point batching logic as cWB's white() mode=0.
"""

import logging

import numpy as np
from memspectrum import MESA
from scipy import signal
from scipy.special import expit
from sklearn.ensemble import IsolationForest
from wdm_wavelet.types.time_frequency_map import TimeFrequencyMap
from wdm_wavelet.wdm import WDM

logger = logging.getLogger(__name__)

_NRMS_DIV_FLOOR = 1.0e-30


def whitening_mesa_python(config, h):
    """
    Pure-Python MESA whitening.

    Returns
    -------
    tuple[pycwb.types.time_series.TimeSeries, TimeFrequencyMap]
        `(conditioned_strain, nRMS_tf_map)` compatible with `whitening_python`.
    """
    from pycwb.types.time_series import TimeSeries

    if not isinstance(h, TimeSeries):
        h_ts = TimeSeries.from_input(h)
    else:
        h_ts = h

    layers = 2 ** config.l_white if getattr(config, "l_white", 0) > 0 else 2 ** config.l_high
    beta_order = getattr(config, "WDM_beta_order", 6)
    precision = getattr(config, "WDM_precision", 10)

    data = np.array(h_ts.data, dtype=np.float64)
    data -= np.mean(data)

    # Use sample rate from config (matching whitening_mesa.py)
    # This is crucial for the sqrt(1/sample_rate) normalization!

    sample_rate = float(h_ts.sample_rate)
    
    nyquist = 0.5 * sample_rate

    logger.info("Whitening data with pure-Python MESA")
    logger.info(
        "autoregressive order=%s, solver=%s beta",
        getattr(config, "mesaOrder", None),
        getattr(config, "mesaSolver", None),
    )
    logger.info(
        "Python whitening:, beta=%s, prec=%s, Window=%ss, Edge=%ss",
        beta_order,
        precision,
        getattr(config, "mesaWindow", 15.0),
    ) 

    # High-pass filter
    low_norm = min(max(float(config.fLow) / nyquist, 1.0e-6), 0.999)
    b_hp, a_hp = signal.butter(8, low_norm, btype="high", analog=False)
    data = signal.filtfilt(b_hp, a_hp, data)

    # MESA window/stride setup
    mesa_window = float(getattr(config, "mesaWindow", 15.0))
    mesa_stride = float(getattr(config, "mesaStride", 5.0))

    if not mesa_stride ==  mesa_window / 3.0:
        logger.warning("mesaStride must be one third of mesaWindow; using mesaWindow/3")
        mesa_stride = mesa_window / 3.0

    window = int(mesa_window * sample_rate)
    stride = int(mesa_stride * sample_rate)

    # Compute number of windows matching whitening_mesa.py logic
    n_windows = (len(data) - window) // stride

    psds = []
    mesa = MESA()
    freqs = None

    # Loop over data segment chunks to compute PSDs
    for i in range(n_windows + 1):
        start = i * stride
        segment = data[start:start + window]
        mesa.solve(segment, method=getattr(config, "mesaSolver", "Fast"), m=getattr(config, "mesaOrder", 500))
        freqs, psd = mesa.spectrum(1.0 / sample_rate)
        psds.append(np.asarray(psd, dtype=np.float64))

    psds = np.asarray(psds, dtype=np.float64)

    # Smooth PSDs with rolling median
    mesa_half_seg = int(getattr(config, "mesaHalfSeg", 0))
    if mesa_half_seg > 0 and psds.shape[0] > 1:
        logger.info("Smoothing PSD estimates with rolling median over %d segments", mesa_half_seg * 2 + 1)
        psds = rolling_median(psds, half_size=mesa_half_seg)

    # Reindex PSDs to exclude glitch-contaminated estimates
    if bool(getattr(config, "mesaReindex", False)) and psds.shape[0] > 5 and freqs is not None:
        logger.warning("Reindexing PSD estimates with IsolationForest")
        psds = reindex_psds(psds, np.asarray(freqs, dtype=np.float64))

    # Planck taper window for whitening
    taper = planck_taper_window(window)

    # Whiten the data in time domain
    whitened = np.array(data, copy=True)
    n_win = n_windows + 1

    for i in range(n_win):
        start = i * stride
        stop = start + window

        # Apply taper and FFT
        chunk = data[start:stop] * taper
        chunk_fft = np.fft.rfft(chunk)
        psd = psds[i, :chunk_fft.size]
        chunk_w = np.fft.irfft(chunk_fft / np.sqrt(psd), n=window) * np.sqrt(1.0 / sample_rate)

        # Overlap-save stitching (matching whitening_mesa.py logic)
        if i == 0:
            whitened[start:stop - stride] = chunk_w[:-stride]
        elif i == n_windows:
            whitened[start + stride:stop] = chunk_w[stride:]
        else:
            whitened[start + stride:stop - stride] = chunk_w[stride:-stride]

    # Determine the valid data length (matching whitening_mesa.py)
    final_stop = n_windows * stride + window

    # Slice to valid length for TF transform
    data_sliced = data[:final_stop].copy()
    whitened_sliced = whitened[:final_stop].copy()

    # Create WDM TF maps
    layers = int(layers)
    wdm = WDM(M=layers, K=layers, beta_order=beta_order, precision=precision)

    tf_raw = wdm.t2w(data_sliced, sample_rate=sample_rate, t0=float(h_ts.t0), MM=-1)
    tf_white = wdm.t2w(whitened_sliced, sample_rate=sample_rate, t0=float(h_ts.t0), MM=-1)

    # Compute nRMS as ratio of raw/whitened TF coefficients
    coeff_raw = np.asarray(tf_raw.data, dtype=np.complex128)
    coeff_white = np.asarray(tf_white.data, dtype=np.complex128)

    # Debug: log coefficient statistics
    logger.info("TF map shape: %s", coeff_raw.shape)

    # Compute ratio using magnitudes
    # Use a more reasonable floor to avoid division issues
    abs_white = np.abs(coeff_white)
    abs_raw = np.abs(coeff_raw)
    
    # Floor based on median to avoid extreme ratios from near-zero values
    floor_value = max(_NRMS_DIV_FLOOR, np.median(abs_white) * 1e-6)
    nrms_matrix = abs_raw / np.maximum(abs_white, floor_value)
    
    # Set low-frequency bins to 1.0 BEFORE median computation (matching whitening_mesa.py)
    # This is different from the bandpass applied later!
    wdm_df = float(tf_white.df)
    low_freq_cutoff = int(16.0 / wdm_df) + 1
    nrms_matrix[:low_freq_cutoff, :] = 1.0
    

    # Compute anchor points using the same logic as _estimate_nrms_cwb_mode0
    nrms_anchor = _compute_nrms_anchors_cwb_style(
        nrms_matrix,
        tf_dt=float(tf_white.dt),
        window_length=float(getattr(config, "whiteWindow", 60.0)),
        stride=float(getattr(config, "whiteStride", 20.0)),
        edge_length=float(getattr(config, "segEdge", 10.0)),
    )

    # Apply cWB bandpass constant to the nRMS TF map data
    nrms_anchor = _apply_cwb_bandpass_constant(
        nrms_anchor,
        f1=16.0,
        f2=0.0,
        a=1.0,
        df=float(tf_white.df),
        f_low_map=float(config.fLow),
        f_high_map=float(config.fHigh),
    )

    nrms_tf = TimeFrequencyMap(
        data=np.asarray(nrms_anchor, dtype=np.float64),
        dt=float(tf_white.dt),
        df=float(tf_white.df),
        t0=float(tf_white.t0),
        len_timeseries=int(tf_white.len_timeseries),
        wdm_params=dict(tf_white.wdm_params),
    )
    conditioned_strain = TimeSeries(data=whitened, dt=h_ts.dt, t0=h_ts.t0) 
    logger.info("Conditioned strain length: %d", len(conditioned_strain))
    logger.info("nRMS TF map shape: %s", nrms_tf.data.shape)

    return conditioned_strain, nrms_tf


def _compute_nrms_anchors_cwb_style(nrms_matrix, tf_dt, window_length=60.0, stride=60.0, edge_length=10.0):
    """
    Compute nRMS anchor points using the same batching logic as cWB's white() mode=0.

    This matches the anchor-point computation in whitening_py._estimate_nrms_cwb_mode0,
    but uses the input nrms_matrix (from ratio computation) instead of power statistics.

    Parameters
    ----------
    nrms_matrix : np.ndarray
        2D array of nRMS values (n_freq, n_time)
    tf_dt : float
        Time resolution of TF map
    window_length : float
        Window length in seconds for median computation
    stride : float
        Stride in seconds between anchor points
    edge_length : float
        Edge length to exclude from computation

    Returns
    -------
    np.ndarray
        nRMS anchor points with shape (n_freq, K+1)
    """
    if nrms_matrix.ndim != 2:
        raise ValueError("Expected 2D nRMS matrix")

    n_freq, n_time = nrms_matrix.shape
    tf_rate = 1.0 / tf_dt
    seg_t = n_time / tf_rate

    # Match cWB logic for window/stride defaults
    if window_length <= 0.0:
        window_length = seg_t - 2.0 * edge_length
    if stride > window_length or stride <= 0.0:
        stride = window_length

    # Offset computation (matching cWB)
    offset = int(edge_length * tf_rate + 0.5)
    if offset & 1:
        offset -= 1
    offset = max(0, offset)

    # Number of anchor intervals
    K = int((seg_t - 2.0 * edge_length) / stride)
    if K < 1:
        K = 1

    n_usable = n_time - 2 * offset
    if n_usable < 4:
        # Not enough data, return median
        nrms_const = np.sqrt(np.nanmedian(nrms_matrix ** 2, axis=1, keepdims=True))
        return nrms_const

    # Samples per segment
    k = n_usable // K
    if k & 1:
        k -= 1
    if k < 2:
        k = 2

    # Window size in samples
    m = int(window_length * tf_rate + 0.5)
    if m < 3:
        m = 3

    mm = m // 2
    jL = (n_time - k * K) // 2
    jR = n_time - offset - m
    jj = jL - mm

    nrms_anchor = np.zeros((n_freq, K + 1), dtype=np.float64)

    for j in range(K + 1):
        if jj < offset:
            p_start = offset
        elif jj >= jR:
            p_start = jR
        else:
            p_start = jj
        jj += k

        p_start = max(0, min(p_start, max(0, n_time - m)))
        p_end = min(n_time, p_start + m)

        window_data = nrms_matrix[:, p_start:p_end]

        if window_data.shape[1] < 3:
            # Not enough data in window, use global median
            sorted_all = np.sort(nrms_matrix, axis=1)
            median_vals = sorted_all[:, nrms_matrix.shape[1] // 2]
        else:
            # Use median of squared values, then sqrt (matching RMS computation)
            # This is analogous to sqrt(median(power) * 0.7191) but for ratio-based nRMS
            sorted_w = np.sort(window_data ** 2, axis=1)
            median_vals = np.sqrt(sorted_w[:, mm])

        nrms_anchor[:, j] = np.maximum(median_vals, 0.0)

    return nrms_anchor


def rolling_median(psds, half_size):
    """
    Smooths the PSDs estimates with a rolling median filter.

    Parameters
    ----------
    psds : np.ndarray
        Array of shape (N_segments, N_frequencies) containing all PSDs
    half_size : int
        Half size of the rolling median window

    Returns
    -------
    np.ndarray
        Smoothed PSDs
    """
    out = np.empty_like(psds)
    n_segments = psds.shape[0]

    for i in range(n_segments):
        left = i - half_size
        right = i + half_size + 1

        # Extend window if we run off edges
        if left < 0:
            deficit = -left
            left = 0
            right = min(n_segments, right + deficit)

        if right > n_segments:
            deficit = right - n_segments
            right = n_segments
            left = max(0, left - deficit)

        out[i] = np.median(psds[int(left):int(right)], axis=0)

    return out


def reindex_psds(psds, f):
    """
    Uses Isolation Forest to exclude PSDs with large deviation from median.

    Parameters
    ----------
    psds : np.ndarray
        Array of shape (N_segments, N_frequencies) containing PSDs
    f : np.ndarray
        Array of sampling frequencies

    Returns
    -------
    np.ndarray
        Reindexed PSDs
    """
    mask_lf = (f > 16.0) & (f < 128.0)
    mask_hf = f > 128.0
    if not np.any(mask_lf) or not np.any(mask_hf):
        return psds

    psd_median = np.median(psds, axis=0)

    dist_lf = np.sqrt(np.mean((np.log(psds[:, mask_lf] / psd_median[mask_lf])) ** 2, axis=1))
    dist_hf = np.sqrt(np.mean((np.log(psds[:, mask_hf] / psd_median[mask_hf])) ** 2, axis=1))
    dist = np.stack([dist_lf, dist_hf], axis=1)

    predictions = IsolationForest(n_estimators=100, contamination=0.15).fit_predict(dist)

    median_lf, median_hf = np.median([dist_lf, dist_hf], axis=1)
    above_median = (dist_lf > median_lf) | (dist_hf > median_hf)
    outlier_mask = np.logical_and(predictions == -1, above_median)

    outliers_idx = np.where(outlier_mask)[0]
    inliers_idx = np.where(~outlier_mask)[0]
    if len(inliers_idx) == 0:
        return psds

    out = np.array(psds, copy=True)
    for idx in outliers_idx:
        new_idx = inliers_idx[np.argmin(np.abs(inliers_idx - idx))]
        out[idx] = out[new_idx]

    logger.info("Reindexed %d PSD estimates out of %d segments", len(outliers_idx), psds.shape[0])
    return out


def planck_taper_window(n_samples, eps=0.15):
    """
    Generate a Planck-taper window.

    Parameters
    ----------
    n_samples : int
        Number of samples
    eps : float
        Taper fraction (default 0.15)

    Returns
    -------
    np.ndarray
        Window array
    """
    k = np.arange(n_samples, dtype=np.float64)
    window = np.zeros(n_samples, dtype=np.float64)

    if n_samples <= 2:
        return window

    left_edge = eps * (n_samples - 1)
    right_edge = (1.0 - eps) * (n_samples - 1)

    left_mask = (k > 0) & (k < left_edge)
    mid_mask = (k >= left_edge) & (k <= right_edge)
    right_mask = (k > right_edge) & (k < n_samples - 1)

    if np.any(left_mask):
        kl = k[left_mask]
        za = eps * (n_samples - 1) * (1.0 / kl + 1.0 / (kl - left_edge))
        window[left_mask] = expit(-za)

    window[mid_mask] = 1.0

    if np.any(right_mask):
        kr = k[right_mask]
        zb = eps * (n_samples - 1) * (1.0 / (n_samples - 1 - kr) + 1.0 / (right_edge - kr))
        window[right_mask] = expit(-zb)

    return window


def _apply_cwb_bandpass_constant(nrms_map, f1, f2, a, df, f_low_map, f_high_map):
    """
    Mirror cWB `WSeries::bandpass(f1, f2, a)` behavior on TF rows.

    For `bandpass(16., 0., 1.)`, rows below the low edge are set to 1.

    Parameters
    ----------
    nrms_map : np.ndarray
        2D array of nRMS values
    f1, f2 : float
        Frequency bounds
    a : float
        Value to set for out-of-band rows
    df : float
        Frequency resolution
    f_low_map, f_high_map : float
        Map frequency bounds

    Returns
    -------
    np.ndarray
        Modified nRMS map
    """
    if nrms_map.ndim != 2:
        raise ValueError("Expected 2D RMS map")

    out = np.array(nrms_map, copy=True)
    n_freq = out.shape[0]
    if n_freq == 0:
        return out

    dF = float(df)
    fl = abs(float(f1)) if abs(float(f1)) > 0.0 else float(f_low_map)
    fh = abs(float(f2)) if abs(float(f2)) > 0.0 else float(f_high_map)

    n = int((fl + dF / 2.0) / dF + 0.1)
    m = int((fh + dF / 2.0) / dF + 0.1) - 1

    if n > m:
        return out

    n = max(0, min(n, n_freq - 1))
    m = max(0, min(m, n_freq - 1))

    indices = np.arange(n_freq)

    keep = np.zeros(n_freq, dtype=bool)
    if f1 >= 0 and f2 >= 0:
        keep = (indices > n) & (indices <= m)
    elif f1 < 0 and f2 < 0:
        keep = (indices < n) | (indices > m)
    elif f1 < 0 and f2 >= 0:
        keep = (indices < n)
    elif f1 >= 0 and f2 < 0:
        keep = (indices >= m)

    out[~keep, :] = float(a)
    return out