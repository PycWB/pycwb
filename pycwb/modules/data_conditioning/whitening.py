"""
Pure-Python whitening without ROOT dependencies.
"""

from pycwb.constants.execution_profile import wdm_options
from .whitening_common import _apply_cwb_bandpass_constant

import logging

import numpy as np
from wdm_wavelet.wdm import WDM
from pycwb.types.noise_rms import make_noise_rms_map

logger = logging.getLogger(__name__)

_NRMS_DIV_FLOOR = 1.0e-30


def whiten_wavelet(config, h, *, apply_bandpass=True):
    """
    Noise whitening via WDM (pure-Python implementation).

    Returns
    -------
    tuple[pycwb.types.time_series.TimeSeries, NoiseRMSMap]
        `(conditioned_strain, nRMS_tf_map)`.
    """
    from pycwb.types.time_series import TimeSeries

    if not isinstance(h, TimeSeries):
        h_ts = TimeSeries.from_input(h)
    else:
        h_ts = h

    layers = 2**config.l_white if getattr(config, "l_white", 0) > 0 else 2**config.l_high
    beta_order = getattr(config, "WDM_beta_order", 6)
    precision = getattr(config, "WDM_precision", 10)

    white_window = (
        getattr(config, "whiteWindow", 60.0)
        if hasattr(config, "whiteWindow") and config.whiteWindow is not None
        else 60.0
    )
    edge_length = getattr(config, "segEdge", 10.0)

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

    wdm = WDM(M=layers, K=layers, beta_order=beta_order, precision=precision, **wdm_options(config))
    tf_map = wdm.t2w(signal_data, sample_rate=sample_rate, t0=t0, MM=-1)

    nRMS_anchor, nRMS_interp = _estimate_nrms_cwb_mode0(
        tf_map,
        window_length=white_window,
        stride=getattr(config, "whiteStride", white_window),
        edge_length=edge_length,
        return_interpolated=True,
    )
    # ReadData target-SNR estimation uses the original noise anchors; the
    # conditioning stage applies bandpass(16, 0, 1) afterward in cWB.
    if apply_bandpass:
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

    # C++ white(nRMS, 1) whitens ALL layers; out-of-band nRMS is already 1.0
    # from bandpass(16., 0., 1), so dividing keeps those coefficients unchanged.
    safe_nrms = np.maximum(nRMS_interp, _NRMS_DIV_FLOOR)
    whitened = coeff / safe_nrms

    tf_map.data = whitened

    nrms_tf = make_noise_rms_map(tf_map, nRMS_anchor, edge_length)

    # C++ inverts BOTH 00° and 90° phases and averages:
    #   tf_map.Inverse()      → time series from 00° phase
    #   wtmp.Inverse(-2)      → time series from 90° phase
    #   output = (00_inv + 90_inv) / 2
    ts_00 = wdm.w2t(tf_map)
    ts_90 = wdm.w2tQ(tf_map)
    whitened_data = 0.5 * (np.array(ts_00.value, dtype=np.float64) + np.array(ts_90.value, dtype=np.float64))
    conditioned_strain = TimeSeries(
        data=whitened_data,
        dt=h_ts.dt,
        t0=h_ts.t0,
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
            # C++ waveSplit picks element at index m//2 (not average of two middle)
            sorted_all = np.sort(power, axis=1)
            median_vals = sorted_all[:, power.shape[1] // 2]
        else:
            # C++ waveSplit(pp, 0, m-1, mm) with mm = m//2: selects the mm-th
            # smallest element.  np.median averages two middle values for even m.
            sorted_w = np.sort(window_data, axis=1)
            median_vals = sorted_w[:, mm]

        nrms_anchor[:, j] = np.sqrt(np.maximum(median_vals * 0.7191, 0.0))

    # Interpolation matching C++ WSeries::white(nRMS, mode) exactly.
    # C++ formula: r = (na[k-1]*(dT-T+t) + na[k]*(T-t))/dT
    # This is REVERSED linear interpolation: left anchor gets weight proportional
    # to distance-from-left, right anchor gets weight proportional to distance-from-right.
    nrms_interp = np.zeros((n_freq, n_time), dtype=np.float64)

    j_arr = np.arange(n_time)

    # Head: j <= jL → anchor[0]  (C++: t <= To)
    nrms_interp[:, j_arr <= jL] = nrms_anchor[:, [0]]

    # Middle: jL < j < jL + K*k → reversed interpolation between anchors
    mid_mask = (j_arr > jL) & (j_arr < jL + K * k)
    j_mid = j_arr[mid_mask]
    seg_k = (j_mid - jL - 1) // k + 1  # 1-based segment index (matching C++ k after increment)
    T_idx = jL + seg_k * k  # right boundary index
    d_left = j_mid - (T_idx - k)  # distance from left anchor
    d_right = T_idx - j_mid  # distance from right anchor
    nrms_interp[:, mid_mask] = (
        nrms_anchor[:, seg_k - 1] * d_left[np.newaxis, :] + nrms_anchor[:, seg_k] * d_right[np.newaxis, :]
    ) / k

    # Tail: j >= jL + K*k → anchor[K]  (C++: t >= To+K*dT)
    nrms_interp[:, j_arr >= jL + K * k] = nrms_anchor[:, [K]]

    nrms_anchor = np.maximum(nrms_anchor, 0.0)
    nrms_interp = np.maximum(nrms_interp, 0.0)
    if return_interpolated:
        return nrms_anchor, nrms_interp
    return nrms_anchor
