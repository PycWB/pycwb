from gwpy.timeseries import TimeSeries
from pycwb.constants import REGRESSION_FILTER_LENGTH, \
    REGRESSION_MATRIX_FRACTION, \
    REGRESSION_SOLVE_EIGEN_THR, REGRESSION_SOLVE_EIGEN_NUM, \
    REGRESSION_SOLVE_REGULATOR, REGRESSION_APPLY_THR, \
    WDM_BETAORDER, WDM_PRECISION
from pycwb.config import Config
import ROOT
import numpy as np


def regression(config: Config, h: TimeSeries):
    """
        Clean data with cWB regression method.
    Input
    ------
    
    config: (dict) configuration dictionary
    h: (TimeSeries) data
    
    Output
    ------
    hh: (ROOT wavearray) cleaned data 
    
    """
    layers = int(config.rateANA / 8)
    wdm = ROOT.WDM(np.double)(layers, layers, WDM_BETAORDER, WDM_PRECISION)
    tf_map = ROOT.WSeries(np.double)(h, wdm)
    tf_map.Forward()

    # TOOD: consideration?
    r = ROOT.regression(tf_map, "target", 1., config.fHigh)
    r.add(h, "target")

    # Calculate prediction
    r.setFilter(REGRESSION_FILTER_LENGTH)  # length of filter
    r.setMatrix(config.segEdge, REGRESSION_MATRIX_FRACTION)
    r.solve(REGRESSION_SOLVE_EIGEN_THR,
            REGRESSION_SOLVE_EIGEN_NUM,
            REGRESSION_SOLVE_REGULATOR)
    r.apply(REGRESSION_APPLY_THR)

    # cleaned data
    hh = r.getClean()

    return hh
