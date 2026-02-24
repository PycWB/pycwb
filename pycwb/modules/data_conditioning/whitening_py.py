"""
Pure-Python whitening without ROOT dependencies.
"""

import logging

import numpy as np
from scipy import signal as scipy_signal
from wdm_wavelet.types.time_frequency_map import TimeFrequencyMap
from wdm_wavelet.wdm import WDM

logger = logging.getLogger(__name__)

_NRMS_DIV_FLOOR = 1.0e-30


def whitening_python(config, h):
    """
    Noise whitening via WDM (pure-Python implementation).

    Returns
    -------
    tuple[pycbc.types.timeseries.TimeSeries, TimeFrequencyMap]
        `(conditioned_strain, nRMS_tf_map)`.
    """
    import pycbc.types

    if not isinstance(h, pycbc.types.TimeSeries):
        h_ts = pycbc.types.TimeSeries(h.value, delta_t=h.dt.value, epoch=h.t0.value)
    else:
        h_ts = h

    layers = 2 ** config.l_white if getattr(config, "l_white", 0) > 0 else 2 ** config.l_high
    beta_order = getattr(config, "WDM_beta_order", 6)
    precision = getattr(config, "WDM_precision", 10)

    white_window = (
        getattr(config, "whiteWindow", 60.0)
        if hasattr(config, "whiteWindow") and config.whiteWindow is not None
        else 60.0
    )
    edge_length = getattr(config, "segEdge", 10.0)
    f_low = float(config.fLow)
    f_high = float(config.fHigh)

    signal_data = np.array(h_ts.data, dtype=np.float64)
    sample_rate = float(h_ts.sample_rate)
    t0 = float(h_ts.start_time)

    logger.info(
        "Python whitening: M=%d, beta=%s, prec=%s, Window=%ss, Edge=%ss",
        layers,
        beta_order,
        precision,
        white_window,
        edge_length,
    )

    wdm = WDM(M=layers, K=layers, beta_order=beta_order, precision=precision)
    tf_map = wdm.t2w(signal_data, sample_rate=sample_rate, t0=t0, MM=-1)

    nRMS_anchor, nRMS_interp = _estimate_nrms_cwb_mode0(
        tf_map,
        window_length=white_window,
        stride=getattr(config, "whiteStride", white_window),
        edge_length=edge_length,
        return_interpolated=True,
    )
    nRMS_anchor = _apply_cwb_bandpass_constant(
        nRMS_anchor,
        f1=16.0,
        f2=0.0,
        a=1.0,
        df=float(tf_map.df),
        f_low_map=float(config.fLow),
        f_high_map=float(config.fHigh),
    )
    nRMS_interp = _apply_cwb_bandpass_constant(
        nRMS_interp,
        f1=16.0,
        f2=0.0,
        a=1.0,
        df=float(tf_map.df),
        f_low_map=float(config.fLow),
        f_high_map=float(config.fHigh),
    )

    coeff = np.asarray(tf_map.data, dtype=np.complex128)
    whitened = np.zeros_like(coeff)

    freqs = np.arange(coeff.shape[0], dtype=np.float64) * float(tf_map.df)
    in_band = (freqs >= f_low) & (freqs <= f_high)

    safe_nrms = np.maximum(nRMS_interp, _NRMS_DIV_FLOOR)
    whitened[in_band, :] = coeff[in_band, :] / safe_nrms[in_band, :]

    tf_map.data = whitened

    nrms_tf = TimeFrequencyMap(
        data=nRMS_anchor,
        dt=float(tf_map.dt),
        df=float(tf_map.df),
        t0=float(tf_map.t0),
        len_timeseries=int(tf_map.len_timeseries),
        wdm_params=dict(tf_map.wdm_params),
    )

    whitened_ts_gwpy = wdm.w2t(tf_map)
    whitened_data = np.array(whitened_ts_gwpy.value, dtype=np.float64)
    conditioned_strain = pycbc.types.TimeSeries(
        whitened_data,
        delta_t=h_ts.delta_t,
        epoch=h_ts.start_time,
    )

    logger.info("  Conditioned strain length: %d", len(conditioned_strain))
    logger.info("  nRMS TF map shape: %s", nrms_tf.data.shape)

    return conditioned_strain, nrms_tf


def _estimate_nrms_cwb_mode0(tf_map, window_length=60.0, stride=60.0, edge_length=10.0, return_interpolated=False):
    """
    Estimate TF nRMS map consistent with cWB `white(..., mode=0)` behavior.

    In cWB mode=0, per-layer input is power (00^2 + 90^2), and
    nRMS anchor values are `sqrt(median(power) * 0.7191)`.
    """
    coeff = np.asarray(tf_map.data)
    if coeff.ndim != 2:
        raise ValueError("Expected 2D TF coefficient array")

    power = np.asarray(coeff.real * coeff.real + coeff.imag * coeff.imag, dtype=np.float64)
    n_freq, n_time = power.shape

    tf_rate = 1.0 / float(tf_map.dt)
    seg_t = n_time / tf_rate

    if window_length <= 0.0:
        window_length = seg_t - 2.0 * edge_length
    if stride > window_length or stride <= 0.0:
        stride = window_length

    offset = int(edge_length * tf_rate + 0.5)
    if offset & 1:
        offset -= 1
    offset = max(0, offset)

    K = int((seg_t - 2.0 * edge_length) / stride)
    if K < 1:
        K = 1

    n_usable = n_time - 2 * offset
    if n_usable < 4:
        median0 = np.median(power, axis=1, keepdims=True)
        nrms_const = np.sqrt(np.maximum(median0 * 0.7191, 1.0e-12))
        if return_interpolated:
            return nrms_const, nrms_const * np.ones_like(power)
        return nrms_const

    k = n_usable // K
    if k & 1:
        k -= 1
    if k < 2:
        k = 2

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

        window_data = power[:, p_start:p_end]
        if window_data.shape[1] < 3:
            median_vals = np.median(power, axis=1)
        else:
            median_vals = np.median(window_data, axis=1)

        nrms_anchor[:, j] = np.sqrt(np.maximum(median_vals * 0.7191, 0.0))

    nrms_interp = np.zeros((n_freq, n_time), dtype=np.float64)

    head = max(0, jL)
    if head > 0:
        nrms_interp[:, :head] = nrms_anchor[:, [0]]

    for j in range(K):
        seg_start = jL + j * k
        seg_end = min(n_time, seg_start + k)
        if seg_start >= n_time or seg_end <= max(seg_start, 0):
            continue

        s = max(seg_start, 0)
        e = seg_end
        i = np.arange(e - s, dtype=np.float64)
        interp = (nrms_anchor[:, [j + 1]] * i + nrms_anchor[:, [j]] * (k - i)) / k
        nrms_interp[:, s:e] = interp

    tail_start = min(n_time, jL + K * k)
    if tail_start < n_time:
        nrms_interp[:, tail_start:] = nrms_anchor[:, [K]]

    nrms_anchor = np.maximum(nrms_anchor, 0.0)
    nrms_interp = np.maximum(nrms_interp, 0.0)
    if return_interpolated:
        return nrms_anchor, nrms_interp
    return nrms_anchor


def _bandpass_rms_frequency(nrms_map, f_low, f_high, sample_rate, df=None):
    """
    Approximate WSeries::bandpass in TF by flattening out-of-band rows to nearest in-band values.
    """
    if nrms_map.ndim != 2:
        raise ValueError("Expected 2D RMS map")

    n_freq, _ = nrms_map.shape
    if n_freq < 2:
        return np.maximum(nrms_map, 1.0e-12)

    if df is None:
        nyquist = sample_rate / 2.0
        freq_bins = np.linspace(0.0, nyquist, n_freq, endpoint=False)
    else:
        freq_bins = np.arange(n_freq, dtype=np.float64) * float(df)

    if f_low is None:
        f_low = 0.0
    if f_high is None or f_high <= 0.0:
        f_high = float(np.max(freq_bins))

    out = np.array(nrms_map, copy=True)
    in_band = np.where((freq_bins >= float(f_low)) & (freq_bins <= float(f_high)))[0]
    if in_band.size == 0:
        return np.maximum(out, 0.0)

    low_idx = int(in_band[0])
    high_idx = int(in_band[-1])

    if low_idx > 0:
        out[:low_idx, :] = out[low_idx:low_idx + 1, :]
    if high_idx < n_freq - 1:
        out[high_idx + 1:, :] = out[high_idx:high_idx + 1, :]

    return np.maximum(out, 0.0)


def _apply_cwb_bandpass_constant(nrms_map, f1, f2, a, df, f_low_map, f_high_map):
    """
    Mirror cWB `WSeries::bandpass(f1, f2, a)` behavior on TF rows.

    For `bandpass(16., 0., 1.)`, rows below the low edge are set to 1.
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
    if (f1 >= 0 and f2 >= 0):
        keep = (indices > n) & (indices <= m)
    elif (f1 < 0 and f2 < 0):
        keep = (indices < n) | (indices > m)
    elif (f1 < 0 and f2 >= 0):
        keep = (indices < n)
    elif (f1 >= 0 and f2 < 0):
        keep = (indices >= m)

    out[~keep, :] = float(a)
    return out


# Backward-compatible helpers kept for existing imports/tests.
def _estimate_noise_rms(tf_map, edge_length=1.0, block_size=4096):
    del block_size
    _, nrms_interp = _estimate_nrms_cwb_mode0(
        tf_map,
        window_length=60.0,
        stride=60.0,
        edge_length=edge_length,
        return_interpolated=True,
    )
    return nrms_interp


def _estimate_noise_rms_cwb(tf_coeff, original_length, window_length=60.0, stride=60.0, edge_length=10.0, sample_rate=2048.0):
    del original_length, sample_rate
    nrms = _estimate_nrms_cwb_mode0(
        tf_coeff,
        window_length=window_length,
        stride=stride,
        edge_length=edge_length,
    )
    median_like = np.maximum((nrms ** 2) / 0.7191, 0.0)
    norm50_like = np.array(nrms, copy=True)
    return median_like, norm50_like


def _bandpass_rms(nrms_map, f_low, f_high, sample_rate):
    return _bandpass_rms_frequency(nrms_map, f_low=f_low, f_high=f_high, sample_rate=sample_rate)


def _whiten_coefficients(tf_coeff, nrms_map, regularization=1.0e-6):
    coeff = np.asarray(tf_coeff.data, dtype=np.complex128)
    nrms_safe = np.maximum(nrms_map, max(regularization, _NRMS_DIV_FLOOR))
    tf_coeff.data = coeff / nrms_safe
    return tf_coeff


def _apply_wiener_filter(tf_coeff, nrms_map, snr_threshold=1.0):
    del snr_threshold
    coeff = np.asarray(tf_coeff.data, dtype=np.complex128)
    power = np.abs(coeff) ** 2
    noise_power = np.maximum(nrms_map, 1.0e-12) ** 2
    weights = power / (power + noise_power + 1.0e-12)
    tf_coeff.data = coeff * weights
    return tf_coeff


def _average_phases(tf_coeff):
    coeff = np.asarray(tf_coeff.data, dtype=np.complex128)
    if coeff.ndim != 2:
        return tf_coeff

    _, T = coeff.shape
    if T % 2 != 0:
        logger.warning("Odd time length; averaging both phases may produce unexpected results")
        return tf_coeff

    T_half = T // 2
    phase_0 = coeff[:, :T_half]
    phase_90 = coeff[:, T_half:]
    averaged = (phase_0 + phase_90) / 2.0
    tf_coeff.data = np.hstack([averaged, averaged])
    return tf_coeff
