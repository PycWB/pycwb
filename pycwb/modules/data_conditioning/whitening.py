from gwpy.timeseries import TimeSeries
from pycwb import utils as ut
import numpy as np
import copy
import ROOT
import logging
from pycwb.config import Config

logger = logging.getLogger(__name__)


def whitening(config: Config, h: TimeSeries):
    """
        
    Input
    -----
    h: data to whiten
    edge: extra data to avoid artifacts
    f_min: (int) minimum frequency
    f_max: (int) maximum frequency
    white_window: (float) time window dT. if = 0 - dT=T, where T is wavearray duration - 2*offset
    white_stride: (float) noise sampling interval (window stride), the number of measurements is
                          k=int((T-2*offset)/stride) if stride=0, then stride is set to dT
    
    Output
    ------
    hw: whitened data
    """

    layers_white = 2 ** config.l_white if config.l_white > 0 else 2 ** config.l_high

    wdm_white = ROOT.WDM(np.double)(layers_white,
                                    layers_white, 6, 10)

    tf_map = ROOT.WSeries(np.double)(h, wdm_white)
    tf_map.Forward()
    tf_map.setlow(config.fLow)
    tf_map.sethigh(config.fHigh)

    # calculate noise rms
    logger.info('calculate noise rms')
    # FIXME: should here be tf_map?
    # FIXME: check the length of data and white parameters to prevent freezing
    nRMS = tf_map.white(config.whiteWindow, 0, config.segEdge,
                        config.whiteStride)

    # high pass filtering at 16Hz
    logger.info('high pass filtering at 16Hz')
    # nRMS.bandpass(16., 0., 1)

    # whiten  0 phase WSeries
    logger.info('whiten  0 phase WSeries')
    tf_map.white(nRMS, 1)
    # whiten 90 phase WSeries
    logger.info('whiten 90 phase WSeries')
    tf_map.white(nRMS, -1)

    wtmp = copy.deepcopy(tf_map)
    tf_map.Inverse()
    wtmp.Inverse(-2)
    tf_map += wtmp
    tf_map *= 0.5

    # hw = ut.convert_wseries_to_wavearray(tf_map)

    return {"TFmap": tf_map, 'nRMS': nRMS}
