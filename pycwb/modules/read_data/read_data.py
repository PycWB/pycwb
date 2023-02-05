from pycwb.utils import convert_pycbc_timeseries_to_wavearray
from gwpy.timeseries import TimeSeries
from .data_check import data_check
from ligo.segments import segment, segmentlist
import pycbc.catalog


def read_from_gwf(ifo_index, config, filename, channel, start=None, end=None):
    # Read data from GWF file
    data = TimeSeries.read(filename, channel, start, end)

    # Check data
    data_check(data, config.inRate)
    data = data.to_pycbc()
    # TODO: complete the following
    # data shift
    # SLAG
    # DC correction
    # if config.dcCal[ifo_index] > 0:
    #     data.data *= config.dcCal[config.ifo.indexof(ifo_index)]

    # resampling
    if config.fResample > 0:
        data = data.resample(1.0 / config.fResample)

    new_sample_rate = data.sample_rate / (1 << config.levelR)
    data = data.resample(1.0 / new_sample_rate)

    # rescaling
    data.data *= (2 ** config.levelR) ** 0.5

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
