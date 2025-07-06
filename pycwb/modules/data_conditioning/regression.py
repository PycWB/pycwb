import ROOT
import numpy as np

from pycwb.modules.cwb_conversions import convert_to_wavearray, convert_wavearray_to_pycbc_timeseries
from pycwb.types.wdm import WDM


def regression(config, h):
    """
    Clean data with cWB regression method.

    :param config: config object
    :type config: Config
    :param wdm: WDM transform for regression
    :type wdm: WDM
    :param h: data to be cleaned
    :type h: pycbc.types.timeseries.TimeSeries or gwpy.timeseries.TimeSeries or ROOT.wavearray(np.double)
    :return: cleaned data
    :rtype: pycbc.types.timeseries.TimeSeries
    """
    layers = int(config.rateANA / 8)
    wdm = WDM(layers, layers, config.WDM_beta_order, config.WDM_precision)

    ##########################################
    # cWB2G regression method
    ##########################################
    h = convert_to_wavearray(h)

    tf_map = ROOT.WSeries(np.double)(h, wdm.wavelet)
    tf_map.Forward()

    # Construct regression from WSeries, add target channel, set low and high frequencies
    r = ROOT.regression(tf_map, "target", 1., config.fHigh)  # TODO: consideration for flow=1.?
    # Add witness channel to the regression list, set low and high frequencies to 0 by default
    # using Bi-othogonal wavelet transforms?
    r.add(h, "target")

    # Calculate prediction
    # set Wiener filter structure
    r.setFilter(config.REGRESSION_FILTER_LENGTH)  # length of filter
    # set system of linear equations: M * F = V (M = matrix array, V =  vector of free coefficients, F = filters)
    r.setMatrix(config.segEdge, config.REGRESSION_MATRIX_FRACTION)
    # solve for eigenvalues and calculate Wiener filters
    r.solve(config.REGRESSION_SOLVE_EIGEN_THR,
            config.REGRESSION_SOLVE_EIGEN_NUM,
            config.REGRESSION_SOLVE_REGULATOR)
    # apply filter to target channel and produce noise TS
    r.apply(config.REGRESSION_APPLY_THR)

    # cleaned data
    hh = r.getClean()
    strain = convert_wavearray_to_pycbc_timeseries(hh)
    tf_map.resize(0)
    ##########################################

    return strain
