import numpy as np
from gwpy.timeseries import TimeSeries
from .data_check import data_check
from ligo.segments import segment, segmentlist
import pycbc.catalog


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


def read_from_catalog(catalog: str, event: str, detector: str, time_slice: tuple = None):
    # Read data from catalog
    m = pycbc.catalog.Merger(event, source=catalog)
    data = m.strain(detector)

    if time_slice:
        data = data.time_slice(time_slice[0], time_slice[1])

    return data, m
