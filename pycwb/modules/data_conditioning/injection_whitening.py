"""Pure-Python whitening of injection strain using a pre-computed nRMS map."""

from pycwb.config.processing import wdm_options
import logging

import numpy as np

logger = logging.getLogger(__name__)


def whiten_injection_strain(config, strain, noise_rms):
    """
    Whiten signal-only injection strain using a pre-computed noise RMS map.

    This uses the ``wdm_wavelet`` Python library for the wavelet transform.

    Parameters
    ----------
    config : Config
        Configuration object (must expose ``l_white``/``l_high``, ``WDM_beta_order``,
        ``WDM_precision``, ``fLow``, ``fHigh``).
    strain : pycwb.types.time_series.TimeSeries or gwpy.TimeSeries
        Signal-only injection strain for a single detector.
    noise_rms : wdm_wavelet.types.time_frequency_map.TimeFrequencyMap
        Per-frequency noise RMS, as returned by
        :func:`~pycwb.modules.data_conditioning.whitening.whiten_wavelet`.
        Its ``data`` array has shape ``(n_freq, K+1)`` (anchor columns spanning the segment).

    Returns
    -------
    tuple[TimeFrequencySeries, TimeFrequencySeries]
        ``(whitened_injection_strain, unwhitened_injection_strain)`` matching
        the interface expected by
        :func:`~pycwb.modules.reconstruction.getINJwaveform.get_INJ_waveform`.
    """
    from pycwb.types.time_series import TimeSeries
    from wdm_wavelet.wdm import WDM as WDMpy
    from wdm_wavelet.types.time_frequency_map import TimeFrequencyMap

    layers = 2**config.l_white if getattr(config, "l_white", 0) > 0 else 2**config.l_high
    beta_order = getattr(config, "WDM_beta_order", 6)
    precision = getattr(config, "WDM_precision", 10)

    if not isinstance(strain, TimeSeries):
        h_ts = TimeSeries.from_input(strain)
    else:
        h_ts = strain

    sample_rate = float(h_ts.sample_rate)
    t0 = float(h_ts.start_time)
    signal_data = np.asarray(h_ts.data, dtype=np.float64)

    logger.info(
        "Whitening injection strain: M=%d, beta=%s, prec=%s, sample_rate=%s Hz",
        layers,
        beta_order,
        precision,
        sample_rate,
    )

    wdm = WDMpy(M=layers, K=layers, beta_order=beta_order, precision=precision, **wdm_options(config))
    tf_map = wdm.t2w(signal_data, sample_rate=sample_rate, t0=t0, MM=-1)

    original_gwpy = wdm.w2t(tf_map)
    original_ts = TimeSeries(
        data=np.asarray(original_gwpy.value, dtype=np.float64),
        dt=h_ts.delta_t,
        t0=h_ts.start_time,
    )

    coeff = np.asarray(tf_map.data, dtype=np.complex128)
    n_freq, n_time = coeff.shape

    if isinstance(noise_rms, TimeFrequencyMap):
        nrms_anchor = np.asarray(noise_rms.data, dtype=np.float64)
    else:
        nrms_anchor = np.asarray(noise_rms, dtype=np.float64)

    if nrms_anchor.ndim == 1:
        nrms_anchor = nrms_anchor[:, np.newaxis]

    K1 = nrms_anchor.shape[1]
    if hasattr(noise_rms, "noise_start"):
        # WSeries::white uses the interior noise-anchor lattice and reversed
        # interpolation, including the left interval at an exact anchor.
        step = 1.0 / (float(noise_rms.noise_rate) * float(tf_map.dt))
        first = (float(noise_rms.noise_start)-t0) / float(tf_map.dt)
        x = (np.arange(n_time, dtype=float)-first)/step
        right = np.clip(np.ceil(x).astype(int), 1, K1-1)
        frac = x-(right-1)
        nrms_interp = (nrms_anchor[:, right-1]*frac + nrms_anchor[:, right]*(1-frac))
        nrms_interp[:, x <= 0] = nrms_anchor[:, [0]]
        nrms_interp[:, x >= K1-1] = nrms_anchor[:, [-1]]
    elif K1 != n_time:
        anchor_pos = np.linspace(0.0, float(n_time - 1), K1)
        full_pos = np.arange(n_time, dtype=np.float64)
        nrms_interp = np.array([np.interp(full_pos, anchor_pos, row) for row in nrms_anchor])
    else:
        nrms_interp = nrms_anchor

    f_low = float(config.fLow)
    f_high = float(config.fHigh)
    freqs = np.arange(n_freq, dtype=np.float64) * float(tf_map.df)
    in_band = (freqs >= f_low) & (freqs <= f_high)

    safe_nrms = np.maximum(nrms_interp, 1.0e-30)
    whitened_coeff = np.zeros_like(coeff)
    whitened_coeff[in_band, :] = coeff[in_band, :] / safe_nrms[in_band, :]

    original_coeff = tf_map.data
    tf_map.data = whitened_coeff
    whitened_gwpy = wdm.w2t(tf_map)
    quadrature_gwpy = wdm.w2tQ(tf_map)
    tf_map.data = original_coeff

    whitened_ts = TimeSeries(
        data=0.5 * (np.asarray(whitened_gwpy.value, dtype=np.float64)
                    + np.asarray(quadrature_gwpy.value, dtype=np.float64)),
        dt=h_ts.delta_t,
        t0=h_ts.start_time,
    )

    return whitened_ts, original_ts
