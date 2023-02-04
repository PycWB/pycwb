from pycwb.utils import convert_pycbc_timeseries_to_wavearray
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


def read_from_catalog(catalog: str, event: str, detectors: list, time_slice: tuple = None):
    # Read data from catalog
    m = pycbc.catalog.Merger(event, source=catalog)
    data = [m.strain(ifo) for ifo in detectors]

    if time_slice:
        for i in range(len(data)):
            data[i] = data[i].crop(time_slice[0], time_slice[1])

    wavearray = [convert_pycbc_timeseries_to_wavearray(d) for d in data]

    return data, m, wavearray
