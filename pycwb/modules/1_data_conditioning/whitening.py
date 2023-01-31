from gwpy.timeseries import TimeSeries
from pycwb.utils import utils as ut
import numpy as np
import copy
import ROOT

ROOT.gSystem.Load("cwb.so")

def whitening(h: TimeSeries, Edge, whiteWindow, whiteStride, F1: int, F2: int): 
    """
        
    Input
    -----
    h: data to whiten
    Edge: extra data to avoid artifacts
    F1: (int) minimum frequency
    F2: (int) maximum frequency
    
    Output
    ------
    hw: whitened data
    """

    layers_high = 1 << 9

    WDMwhite = ROOT.WDM(np.double)(layers_high,
                                   layers_high, 6, 10)

    tf_map = ROOT.WSeries(np.double)(h, WDMwhite)
    tf_map.Forward()
    tf_map.setlow(F1)
    tf_map.sethigh(F2)
    #// calculate noise rms
    nRMS = tf_map.white(whiteWindow, 0, Edge, whiteStride)

    #// whiten  0 phase WSeries
    tf_map.white(nRMS,1);                                     
    tf_map.white(nRMS,-1);

    wtmp = copy.deepcopy(tf_map);
    tf_map.Inverse();
    wtmp.Inverse(-2);
    tf_map += wtmp;
    tf_map *= 0.5;

    hw = ut.convert_wseries_to_wavearray(tf_map)
    
    return hw

