from gwpy.timeseries import TimeSeries
from pycwb import utils as ut
import numpy as np
import copy
import ROOT


def max_energy(h: TimeSeries, pwdm, m_tau: float, up_N: int, pattern: int):
    """
    produce TF maps with max over the sky energy
    Input
    :param h: input time series
    :param pwdm: wavelet used for the transformation
    :param m_tau: range of time delays
    :param up_N: downsample factor to obtain coarse TD steps
    :param pattern: clustering pattern
    :return:
    """
