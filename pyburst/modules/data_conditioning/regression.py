from gwpy.timeseries import TimeSeries
from pyburst.constants import REGRESSION_FILTER_LENGTH, \
    REGRESSION_MATRIX_FRACTION, \
    REGRESSION_SOLVE_EIGEN_THR, REGRESSION_SOLVE_EIGEN_NUM, \
    REGRESSION_SOLVE_REGULATOR, REGRESSION_APPLY_THR, \
    WDM_BETAORDER, WDM_PRECISION
from pyburst.config import Config
import ROOT
import numpy as np


def regression(config: Config, wdm: ROOT.WDM, h: TimeSeries):
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
    tf_map = ROOT.WSeries(np.double)(h, wdm)
    tf_map.Forward()

    # Construct regression from WSeries, add target channel, set low and high frequencies
    r = ROOT.regression(tf_map, "target", 1., config.fHigh) # TODO: consideration for flow=1.?
    # Add witness channel to the regression list, set low and high frequencies to 0 by default
    # using Bi-othogonal wavelet transforms?
    r.add(h, "target")

    # Calculate prediction
    # set Wiener filter structure
    r.setFilter(REGRESSION_FILTER_LENGTH)  # length of filter
    # set system of linear equations: M * F = V (M = matrix array, V =  vector of free coefficients, F = filters)
    r.setMatrix(config.segEdge, REGRESSION_MATRIX_FRACTION)
    # solve for eigenvalues and calculate Wiener filters
    r.solve(REGRESSION_SOLVE_EIGEN_THR,
            REGRESSION_SOLVE_EIGEN_NUM,
            REGRESSION_SOLVE_REGULATOR)
    # apply filter to target channel and produce noise TS
    r.apply(REGRESSION_APPLY_THR)

    # cleaned data
    hh = r.getClean()

    return hh
