from gwpy.timeseries import TimeSeries
from pycwb import utils as ut
import numpy as np
import copy
import ROOT


def whitening(h: TimeSeries, edge, white_window: float, white_stride: float, f_min: int, f_max: int):
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

    layers_high = 1 << 9

    WDMwhite = ROOT.WDM(np.double)(layers_high,
                                   layers_high, 6, 10)

    tf_map = ROOT.WSeries(np.double)(h, WDMwhite)
    tf_map.Forward()
    tf_map.setlow(f_min)
    tf_map.sethigh(f_max)
    # // calculate noise rms
    nRMS = tf_map.white(white_window, 0, edge, white_stride)

    # // whiten  0 phase WSeries
    tf_map.white(nRMS, 1)
    tf_map.white(nRMS, -1)

    wtmp = copy.deepcopy(tf_map)
    tf_map.Inverse()
    wtmp.Inverse(-2)
    tf_map += wtmp
    tf_map *= 0.5

    hw = ut.convert_wseries_to_wavearray(tf_map)

    return hw
