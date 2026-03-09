import warnings

import numpy as np

try:
    import ROOT
except ImportError:
    ROOT = None
    warnings.warn(
        "ROOT module not found. CWB conversions will not work. This warning will be removed in future versions when ROOT is no longer a dependency.",
        ImportWarning,
        stacklevel=2,
    )
import logging
from pycwb.modules.cwb_conversions import convert_to_wavearray, convert_wseries_to_time_frequency_series, convert_to_wseries
from pycwb.types.time_frequency_series import TimeFrequencySeries
from pycwb.types.wdm import WDM

logger = logging.getLogger(__name__)



def whitening_mdc(config, h, nRMS):
    """
    Performs whitening on the given strain data with provided nRMS

    :param config: config object
    :type config: Config
    :param h: strain data
    :type h: pycbc.types.timeseries.TimeSeries or gwpy.timeseries.TimeSeries or ROOT.wavearray(np.double)
    :param nRMS: noise rms data
    :type nRMS: pycwb.types.time_frequency_series.TimeFrequencySeries or ROOT.WSeries<double>
    :return: (whitened strain, original strain)
    :rtype: tuple[pycwb.types.time_frequency_series.TimeFrequencySeries, pycwb.types.time_frequency_series.TimeFrequencySeries]
    """
    layers_white = 2 ** config.l_white if config.l_white > 0 else 2 ** config.l_high
    wdm_white = WDM(layers_white, layers_white, config.WDM_beta_order, config.WDM_precision)

    # TODO: check the length of data and white parameters to prevent freezing
    # check if whitening WDM filter lenght is less than cwb scratch
    wdmFlen = wdm_white.m_H / config.rateANA
    if wdmFlen > config.segEdge + 0.001:
        logger.error("Error - filter scratch must be <= cwb scratch!!!")
        logger.error(f"filter length : {wdmFlen} sec")
        logger.error(f"cwb   scratch : {config.segEdge} sec")
        raise ValueError("Filter scratch must be <= cwb scratch!!!")
    else:
        logger.info(f"WDM filter max length = {wdmFlen} (sec)")
    
    tf_map = ROOT.WSeries(np.double)(convert_to_wavearray(h), wdm_white.wavelet)
    tf_map.Forward()
    tf_map.setlow(config.fLow)
    tf_map.sethigh(config.fHigh)

    if isinstance(nRMS, TimeFrequencySeries):
        nRMS = convert_to_wseries(nRMS)
            
    # save original data
    hot = ROOT.WSeries(np.double)(tf_map)
    hot.Inverse()

    # whiten  0 phase WSeries
    tf_map.white(nRMS, 1)
    # whiten 90 phase WSeries
    tf_map.white(nRMS, -1)

    wtmp = ROOT.WSeries(np.double)(tf_map)
    # average 00 and 90 phase
    tf_map.Inverse()
    wtmp.Inverse(-2)
    tf_map += wtmp
    tf_map *= 0.5

    tf_map_whitened = convert_wseries_to_time_frequency_series(tf_map)
    HoT = convert_wseries_to_time_frequency_series(hot)
    wtmp.resize(0)

    return tf_map_whitened, HoT


def whitening_mdc_py(config, h, nRMS):
    """
    Pure-Python whitening of MDC strain using a pre-computed nRMS map (no ROOT required).

    Mirrors :func:`whitening_mdc` but uses the ``wdm_wavelet`` Python library for the
    wavelet transform instead of ROOT/WSeries.

    Parameters
    ----------
    config : Config
        Configuration object (must expose ``l_white``/``l_high``, ``WDM_beta_order``,
        ``WDM_precision``, ``fLow``, ``fHigh``).
    h : pycbc.types.TimeSeries or gwpy.TimeSeries
        MDC strain data for a single detector.
    nRMS : wdm_wavelet.types.time_frequency_map.TimeFrequencyMap
        Per-frequency noise RMS, as returned by the pure-Python
        :func:`~pycwb.modules.data_conditioning.whitening_py.whitening_python`.
        Its ``data`` array has shape ``(n_freq, K+1)`` (anchor columns spanning the segment).

    Returns
    -------
    tuple[TimeFrequencySeries, TimeFrequencySeries]
        ``(whitened_mdc, original_mdc)`` — both wrap a pycbc ``TimeSeries`` in their
        ``.data`` attribute, matching the interface expected by
        :func:`~pycwb.modules.reconstruction.getINJwaveform.get_INJ_waveform`.
    """
    import pycbc.types
    from wdm_wavelet.wdm import WDM as WDMpy
    from wdm_wavelet.types.time_frequency_map import TimeFrequencyMap

    layers = 2 ** config.l_white if getattr(config, "l_white", 0) > 0 else 2 ** config.l_high
    beta_order = getattr(config, "WDM_beta_order", 6)
    precision = getattr(config, "WDM_precision", 10)

    # ------------------------------------------------------------------
    # Normalise input to pycbc TimeSeries
    # ------------------------------------------------------------------
    if not isinstance(h, pycbc.types.TimeSeries):
        signal_data = h.value if hasattr(h, "value") else np.asarray(h)
        dt = h.dt.value if hasattr(h, "dt") and hasattr(h.dt, "value") else float(h.dt)
        t0 = h.t0.value if hasattr(h, "t0") and hasattr(h.t0, "value") else float(h.t0)
        h_ts = pycbc.types.TimeSeries(signal_data.astype(np.float64), delta_t=dt, epoch=t0)
    else:
        h_ts = h

    sample_rate = float(h_ts.sample_rate)
    t0 = float(h_ts.start_time)
    signal_data = np.asarray(h_ts.data, dtype=np.float64)

    logger.info(
        "Python whitening_mdc: M=%d, beta=%s, prec=%s, sample_rate=%s Hz",
        layers, beta_order, precision, sample_rate,
    )

    # ------------------------------------------------------------------
    # Forward WDM transform
    # ------------------------------------------------------------------
    wdm = WDMpy(M=layers, K=layers, beta_order=beta_order, precision=precision)
    tf_map = wdm.t2w(signal_data, sample_rate=sample_rate, t0=t0, MM=-1)

    # ------------------------------------------------------------------
    # Save original (un-whitened) inverse transform → HoT
    # ------------------------------------------------------------------
    original_gwpy = wdm.w2t(tf_map)
    original_ts = pycbc.types.TimeSeries(
        np.asarray(original_gwpy.value, dtype=np.float64),
        delta_t=h_ts.delta_t,
        epoch=h_ts.start_time,
    )

    # ------------------------------------------------------------------
    # Interpolate anchor nRMS  (n_freq, K+1)  →  (n_freq, n_time)
    # ------------------------------------------------------------------
    coeff = np.asarray(tf_map.data, dtype=np.complex128)
    n_freq, n_time = coeff.shape

    if isinstance(nRMS, TimeFrequencyMap):
        nrms_anchor = np.asarray(nRMS.data, dtype=np.float64)
    else:
        nrms_anchor = np.asarray(nRMS, dtype=np.float64)

    # Safety: make sure nrms_anchor is 2-D
    if nrms_anchor.ndim == 1:
        nrms_anchor = nrms_anchor[:, np.newaxis]

    K1 = nrms_anchor.shape[1]
    if K1 != n_time:
        anchor_pos = np.linspace(0.0, float(n_time - 1), K1)
        full_pos = np.arange(n_time, dtype=np.float64)
        nrms_interp = np.empty((n_freq, n_time), dtype=np.float64)
        for fi in range(n_freq):
            nrms_interp[fi] = np.interp(full_pos, anchor_pos, nrms_anchor[fi])
    else:
        nrms_interp = nrms_anchor

    # ------------------------------------------------------------------
    # Whiten within the analysis band
    # ------------------------------------------------------------------
    f_low = float(config.fLow)
    f_high = float(config.fHigh)
    freqs = np.arange(n_freq, dtype=np.float64) * float(tf_map.df)
    in_band = (freqs >= f_low) & (freqs <= f_high)

    safe_nrms = np.maximum(nrms_interp, 1.0e-30)
    whitened_coeff = np.zeros_like(coeff)
    whitened_coeff[in_band, :] = coeff[in_band, :] / safe_nrms[in_band, :]

    # ------------------------------------------------------------------
    # Inverse transform → whitened time series
    # ------------------------------------------------------------------
    original_coeff = tf_map.data          # keep reference
    tf_map.data = whitened_coeff
    whitened_gwpy = wdm.w2t(tf_map)
    tf_map.data = original_coeff          # restore (non-destructive)

    whitened_ts = pycbc.types.TimeSeries(
        np.asarray(whitened_gwpy.value, dtype=np.float64),
        delta_t=h_ts.delta_t,
        epoch=h_ts.start_time,
    )

    return whitened_ts, original_ts
