from pycwb.utils import convert_pycbc_timeseries_to_wavearray
from gwpy.timeseries import TimeSeries
from .data_check import data_check
from ligo.segments import segment, segmentlist
import pycbc.catalog
import logging
from multiprocessing import Pool
import time

from ..job_segment import WaveSegment

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

    # timer
    timer_end = time.perf_counter()
    logger.info(f'Read data from config in {timer_end - timer_start} seconds')
    return data


def _read_from_config_wrapper(config, i):
    with open(config.frFiles[i], 'r') as f:
        filenames = f.read()
    # read data from the files
    return read_from_gwf(i, config, filenames, config.channelNamesRaw[i])


def read_from_job_segment(config, job_seg: WaveSegment):
    timer_start = time.perf_counter()

    with Pool(processes=len(job_seg.frames)) as pool:
        data = pool.starmap(_read_from_job_segment_wrapper, [
            (config, frame, job_seg) for frame in job_seg.frames
        ])

    merged_data = []

    ifo_frames = [[i for i, frame in enumerate(job_seg.frames) if frame.ifo == ifo] for ifo in config.ifo]

    for frames in ifo_frames:
        if len(frames) == 1:
            ifo_data = data[frames[0]]
        else:
            ifo_data = TimeSeries.from_pycbc(data[frames[0]])
            for i in frames[1:]:
                ifo_data.append(data[i], gap='raise')

            ifo_data = ifo_data.to_pycbc()
    # for ifo in config.ifo:
    #     ifo_data = None
    #     # merge data if there are more than one file for one ifo
    #     for i, frame in enumerate(job_seg.frames):
    #
    #         if frame.ifo == ifo:
    #             if ifo_data is None:
    #                 ifo_data = data[i]
    #             else:
    #                 ifo_data.append(data[i], gap='raise')

        # check if data range match with job segment
        if ifo_data.start_time != job_seg.start_time - config.segEdge or ifo_data.end_time != job_seg.end_time + config.segEdge:
            logger.error(f'Job segment {job_seg} not match with data {ifo_data}, '
                         f'the gwf data start at {ifo_data.start_time} and end at {ifo_data.end_time}')
            raise ValueError(f'Job segment {job_seg} not match with data {ifo_data}')
        
        merged_data.append(ifo_data)

    timer_end = time.perf_counter()
    logger.info(f'Read data from job segment in {timer_end - timer_start} seconds')
    return merged_data


def _read_from_job_segment_wrapper(config, frame, job_seg: WaveSegment):
    start = job_seg.start_time - config.segEdge
    end = job_seg.end_time + config.segEdge

    if frame.start_time > start:
        start = frame.start_time

    if frame.end_time < end:
        end = frame.end_time

    i = config.ifo.index(frame.ifo)
    return read_from_gwf(i, config, frame.path, config.channelNamesRaw[i], start=start, end=end)

