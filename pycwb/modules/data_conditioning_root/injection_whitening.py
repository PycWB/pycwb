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



def whiten_injection_strain(config, strain, noise_rms):
    """
    Whiten signal-only injection strain using the provided noise RMS.

    :param config: config object
    :type config: Config
    :param strain: signal-only injection strain
    :type strain: pycwb.types.time_series.TimeSeries or gwpy.timeseries.TimeSeries or ROOT.wavearray(np.double)
    :param noise_rms: noise RMS data
    :type noise_rms: pycwb.types.time_frequency_series.TimeFrequencySeries or ROOT.WSeries<double>
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
    
    tf_map = ROOT.WSeries(np.double)(convert_to_wavearray(strain), wdm_white.wavelet)
    tf_map.Forward()
    tf_map.setlow(config.fLow)
    tf_map.sethigh(config.fHigh)

    if isinstance(noise_rms, TimeFrequencySeries):
        noise_rms = convert_to_wseries(noise_rms)
            
    # Preserve the unwhitened injection strain for amplitude measurements.
    unwhitened_series = ROOT.WSeries(np.double)(tf_map)
    unwhitened_series.Inverse()

    # whiten  0 phase WSeries
    tf_map.white(noise_rms, 1)
    # whiten 90 phase WSeries
    tf_map.white(noise_rms, -1)

    wtmp = ROOT.WSeries(np.double)(tf_map)
    # average 00 and 90 phase
    tf_map.Inverse()
    wtmp.Inverse(-2)
    tf_map += wtmp
    tf_map *= 0.5

    whitened_strain = convert_wseries_to_time_frequency_series(tf_map)
    unwhitened_strain = convert_wseries_to_time_frequency_series(
        unwhitened_series
    )
    wtmp.resize(0)

    return whitened_strain, unwhitened_strain
