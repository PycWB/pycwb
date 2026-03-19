"""
Pure-Python PSD variability correction using wdm_wavelet.
"""

import logging

import numpy as np
from wdm_wavelet.wdm import WDM
from pycwb.types.time_series import TimeSeries

logger = logging.getLogger(__name__)


def psd_correction_python(
    config,
    h,
    high_frequency_threshold: float = 513.0,
    layer_rate: float = 32.0,
    smooth_seconds: float = 6.0,
):
    """
    Apply cWB-like PSD variability correction in pure Python.

    Parameters
    ----------
    config : Config
        Configuration object with `segEdge`, `WDM_beta_order`, `WDM_precision`.
    h : pycwb.types.time_series.TimeSeries | gwpy.timeseries.TimeSeries
        Input conditioned/whitened strain.
    high_frequency_threshold : float, optional
        Minimum frequency (Hz) used to build PSD variability envelope.
    layer_rate : float, optional
        Target WDM layer rate used in the original plugin (`R=32`).
    smooth_seconds : float, optional
        Smoothing scale in seconds for envelope cleaning.

    Returns
    -------
    pycwb.types.time_series.TimeSeries
        Corrected time-domain strain.
    """
    h_ts = _as_pycwb_timeseries(h)
    sample_rate = float(h_ts.sample_rate)

    layers = int(sample_rate / float(layer_rate) + 0.1)
    if layers < 1:
        logger.warning("PSD correction skipped: invalid WDM layers=%d", layers)
        return h_ts

    beta_order = int(getattr(config, "WDM_beta_order", 6))
    precision = int(getattr(config, "WDM_precision", 10))
    edge_seconds = float(getattr(config, "segEdge", 0.0) or 0.0)

    wdm = WDM(M=layers, K=layers, beta_order=beta_order, precision=precision)
    signal_data = np.asarray(h_ts.data, dtype=np.float64)
    t0 = float(h_ts.t0)
    tf_map = wdm.t2w(signal_data, sample_rate=sample_rate, t0=t0, MM=-1)

    coeff = np.asarray(tf_map.data, dtype=np.complex128)
    if coeff.ndim != 2 or coeff.shape[1] < 5:
        logger.warning("PSD correction skipped: invalid TF map shape %s", coeff.shape)
        return h_ts

    n_freq, n_time = coeff.shape
    df = float(tf_map.df)
    freq_axis = np.arange(n_freq, dtype=np.float64) * df
    selected = freq_axis >= float(high_frequency_threshold)
    n_selected = int(np.count_nonzero(selected))
    if n_selected == 0:
        logger.info("PSD correction skipped: no layers above %.1f Hz", high_frequency_threshold)
        return h_ts

    edge_bins = int(edge_seconds * float(layer_rate))
    core_start = max(0, min(edge_bins, n_time - 1))
    core_end = max(core_start + 1, n_time - core_start)

    amp = np.abs(coeff[selected, :])
    layer_median = np.array([_median_core(row, core_start, core_end) for row in amp], dtype=np.float64)
    layer_cap = 4.0 * layer_median[:, None]
    capped = np.minimum(amp, layer_cap)

    u = np.mean(capped, axis=0)
    q = np.mean(amp, axis=0)
    v = _smooth_envelope(u, smooth_seconds=smooth_seconds, rate=layer_rate, edge_seconds=edge_seconds)

    um = float(_median_core(u, core_start, core_end))
    vm = float(_median_core(v, core_start, core_end))
    if um <= 0.0 or not np.isfinite(um):
        logger.warning("PSD correction skipped: invalid envelope median %.6e", um)
        return h_ts

    v = v + (um - vm)
    v = np.maximum(v, 0.0)

    w = np.ones(n_time, dtype=np.float64)
    for i in range(n_time):
        uu = float(u[i])
        qq = float(q[i])
        aa = uu / um - 1.0
        aa = np.sqrt(aa) if aa > 0.0 else 0.0
        aa = np.sqrt(aa) if aa < 1.0 else 1.0
        aa = np.sqrt(aa)

        if qq > 0.0:
            aa = (uu - v[i]) * aa * uu * uu / (qq * qq)
        else:
            aa = 0.0
        aa = uu - (aa if aa > 0.0 else 0.0)
        aa = aa / uu if uu > um and uu > 0.0 else 1.0
        w[i] = 1.0 if aa > 0.97 else aa + 0.03

    v1 = w.copy()
    q2 = q * q
    w2 = w.copy()
    for i in range(2, n_time - 2):
        aa = 0.0
        uu = 0.0
        for j in range(-2, 3):
            aa += (1.0 - v1[i + j]) * q2[i + j]
            uu = max(uu, q2[i + j])
        if uu > 0.0:
            base = max(0.0, 1.0 - aa / uu / 5.0)
            w2[i] = np.power(base, 2.5)
        else:
            w2[i] = 1.0

    w_final = np.maximum(w2, 0.0)
    tf_map.data = coeff * w_final[None, :]

    corrected_00 = _to_numpy_1d(wdm.w2t(tf_map))
    corrected_90 = _to_numpy_1d(wdm.w2tQ(tf_map))
    corrected = 0.5 * (corrected_00 + corrected_90)

    logger.info(
        "PSD correction applied: layers=%d selected=%d edge=%.2fs",
        layers,
        n_selected,
        edge_seconds,
    )

    return TimeSeries(data=corrected, t0=float(h_ts.t0), dt=float(h_ts.dt))


def _as_pycwb_timeseries(h):
    ts = TimeSeries.from_input(h)
    if not isinstance(ts.data, np.ndarray) or ts.data.dtype != np.float64:
        ts = TimeSeries(data=np.asarray(ts.data, dtype=np.float64), t0=float(ts.t0), dt=float(ts.dt))
    return ts


def _to_numpy_1d(x):
    if hasattr(x, "value"):
        return np.asarray(x.value, dtype=np.float64)
    if hasattr(x, "data"):
        return np.asarray(x.data, dtype=np.float64)
    return np.asarray(x, dtype=np.float64)


def _median_core(x, start, end):
    start = int(max(0, start))
    end = int(min(len(x), end))
    if end <= start:
        return float(np.median(x))
    return float(np.median(np.asarray(x[start:end], dtype=np.float64)))


def _smooth_envelope(x, smooth_seconds, rate, edge_seconds):
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n <= 2:
        return x.copy()

    width = int(round(float(smooth_seconds) * float(rate)))
    if width < 3:
        width = 3
    if width % 2 == 0:
        width += 1

    pad = width // 2
    x_pad = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(width, dtype=np.float64) / float(width)
    y = np.convolve(x_pad, kernel, mode="valid")

    edge_bins = int(max(0.0, float(edge_seconds)) * float(rate))
    if edge_bins > 0 and 2 * edge_bins < n:
        y[:edge_bins] = x[:edge_bins]
        y[n - edge_bins:] = x[n - edge_bins:]

    return y


__all__ = ["psd_correction_python"]
