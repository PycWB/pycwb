import numpy as np
import ROOT
import logging
from pycwb.modules.cwb_conversions import convert_to_wavearray, convert_wseries_to_time_frequency_series, convert_to_wseries
from pycwb.types.time_frequency_series import TimeFrequencySeries
from pycwb.types.wdm import WDM

logger = logging.getLogger(__name__)


def whitening_cwb(config, h):
    """
    Performs whitening on the given strain data

    :param config: config object
    :type config: Config
    :param h: strain data
    :type h: pycbc.types.timeseries.TimeSeries or gwpy.timeseries.TimeSeries or ROOT.wavearray(np.double)
    :return: (whitened strain, nRMS)
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
    ##########################################
    # cWB2G whitening method
    ##########################################
    tf_map = ROOT.WSeries(np.double)(convert_to_wavearray(h), wdm_white.wavelet)
    tf_map.Forward()
    tf_map.setlow(config.fLow)
    tf_map.sethigh(config.fHigh)
    # calculate noise rms
    # FIXME: should here be tf_map?
    # FIXME: check the length of data and white parameters to prevent freezing
    nRMS = tf_map.white(config.whiteWindow, 0, config.segEdge,
                        config.whiteStride)

    # high pass filtering at 16Hz
    nRMS.bandpass(16., 0., 1)

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

    # hw = ut.convert_wseries_to_wavearray(tf_map)
    tf_map_whitened = convert_wseries_to_time_frequency_series(tf_map)
    n_rms = convert_wseries_to_time_frequency_series(nRMS)
    wtmp.resize(0)
    ##########################################

    return tf_map_whitened, n_rms

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