import numpy as np
import ROOT
import logging
from pyburst.config import Config

logger = logging.getLogger(__name__)


def whitening(config, wdm_white, h):
    """
    Performs whitening on the given strain data

    :param config: config object
    :type config: Config
    :param wdm_white: WDM used for whitening
    :type wdm_white: ROOT.WDM
    :param h: strain data
    :type h: ROOT.wavearray(np.double)
    :return: (whitened strain, nRMS)
    :rtype: tuple[ROOT.wavearray(np.double), float]
    """

    tf_map = ROOT.WSeries(np.double)(h, wdm_white)
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

    return tf_map, nRMS
