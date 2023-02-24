from pycwb.utils import convert_pycbc_timeseries_to_wavearray
from gwpy.timeseries import TimeSeries
from .data_check import data_check
from ligo.segments import segment, segmentlist
import pycbc.catalog
import logging
from multiprocessing import Pool
import time
logger = logging.getLogger(__name__)


def read_from_gwf(ifo_index, config, filename, channel, start=None, end=None):
    # Read data from GWF file
    if start or end:
        logger.info(f'Reading data from {filename} from {channel} from {start} to {end}')
    else:
        logger.info(f'Reading data from {filename} from {channel}')

    data = TimeSeries.read(filename, channel, start, end)

    # TODO: Check data
    data_check(data, config.inRate)
    data = data.to_pycbc()
    # TODO: complete the following
    # data shift
    # SLAG
    # DC correction
    if config.dcCal[ifo_index] > 0 and config.dcCal[ifo_index] != 1.0:
        data.data *= config.dcCal[config.ifo.indexof(ifo_index)]

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
    from gwpy.timeseries import TimeSeriesDict

    channels = ['H1:DCS-CALIB_STRAIN_CLEAN_SUB60HZ_C01',
                'L1:DCS-CALIB_STRAIN_CLEAN_SUB60HZ_C01',
                'V1:Hrec_hoft_16384Hz']

    data_dict = TimeSeriesDict.get(
        channels,
        1242442967 - 300,
        1242442967 + 300,
    )

    data = [data_dict[c] for c in channels]
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


def read_from_config(config):
    # timer
    timer_start = time.perf_counter()

    # data = []
    with Pool(processes=len(config.ifo)) as pool:
        data = pool.starmap(_read_from_config_wrapper, [(config, i) for i in range(len(config.ifo))])
    # for i in range(len(config.ifo)):
    #     # read path string from the files in config.frFiles
    #     filenames = ""
    #     with open(config.frFiles[i], 'r') as f:
    #         filenames = f.read()
    #     # read data from the files
    #     data.append(read_from_gwf(i, config, filenames, config.channelNamesRaw[i]))

    # timer
    timer_end = time.perf_counter()
    logger.info(f'Read data from config in {timer_end - timer_start} seconds')
    return data


def _read_from_config_wrapper(config, i):
    with open(config.frFiles[i], 'r') as f:
        filenames = f.read()
    # read data from the files
    return read_from_gwf(i, config, filenames, config.channelNamesRaw[i])
