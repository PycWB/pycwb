import numpy as np
import ROOT
import logging
from pycwb.config import Config
from pycwb.conversions import convert_to_wavearray, convert_wseries_to_time_frequency_series
from pycwb.types import TimeFrequencySeries

logger = logging.getLogger(__name__)


def whitening(config, wdm_white, h):
    """
    Performs whitening on the given strain data

    :param config: config object
    :type config: Config
    :param wdm_white: WDM used for whitening
    :type wdm_white: WDM
    :param h: strain data
    :type h: pycbc.types.timeseries.TimeSeries or gwpy.timeseries.TimeSeries or ROOT.wavearray(np.double)
    :return: (whitened strain, nRMS)
    :rtype: tuple[TimeFrequencySeries, TimeFrequencySeries]
    """

    # tf_map = TimeFrequencySeries(data=h, wavelet=wdm_white)
    # tf_map.forward()
    # tf_map.f_low = config.fLow
    # tf_map.f_high = config.fHigh
    # tf_map = convert_time_frequency_series_to_wseries(tf_map)

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
    tf_map.Inverse()
    wtmp.Inverse(-2)
    tf_map += wtmp
    tf_map *= 0.5

    # hw = ut.convert_wseries_to_wavearray(tf_map)
    tf_map_whitened = convert_wseries_to_time_frequency_series(tf_map)
    n_rms = convert_wseries_to_time_frequency_series(nRMS)
    ##########################################

    return tf_map_whitened, n_rms