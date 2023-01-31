import numpy as np
from gwpy.timeseries import TimeSeries
from .data_check import data_check
from ligo.segments import segment, segmentlist


def read_from_gwf(detector, sample_rate, filename, channel, start, end):
    # Read data from GWF file
    data = TimeSeries.read(filename, channel, start, end)

    # Check data
    data_check(data, sample_rate)

    # data shift
    # SLAG
    # DC correction
    # resampling
    # rescaling
    # return data
    return data


def read_from_online(detector, sample_rate, channel, start, end):
    data = segmentlist([segment(start, end)])
    # Check data

    # data shift
    # SLAG
    # DC correction
    # resampling
    # rescaling
    # return data
    return data