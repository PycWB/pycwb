from gwpy.timeseries import TimeSeries
from .data_check import check_and_resample
import pycbc.catalog
import logging
from multiprocessing import Pool
import time

from ..job_segment import WaveSegment

logger = logging.getLogger(__name__)


def read_from_gwf(filename, channel, start=None, end=None):
    """
    Read data from GWF file

    :param ifo_index: the index of the ifo in the config.ifo list for processing with user defined parameters
    :type ifo_index: int
    :param config: user configuration
    :type config: Config
    :param filename: path to the gwf file
    :type filename: str
    :param channel: channel name to read from the gwf file
    :type channel: str
    :param start: start time to read from the gwf file, defaults to reading from the beginning of the file
    :type start: float, optional
    :param end: end time to read from the gwf file, defaults to reading to the end of the file
    :type end: float, optional
    :return: strain data
    :rtype: pycbc.types.timeseries.TimeSeries
    """
    # Read data from GWF file
    if start or end:
        logger.info(f'Reading data from {filename} ({channel}) from {start} to {end}')
    else:
        logger.info(f'Reading data from {filename} ({channel})')

    # Read gwf file
    data = TimeSeries.read(filename, channel, start, end)

    return data


def read_from_online(channels, start, end):
    """
    Read data from data server with gwpy

    :param channels: list of channels to read
    :type channels: list[str]
    :param start: start time to read from the data server
    :type start: float
    :param end: end time to read from the data server
    :type end: float
    :return: list of strain data
    :rtype: list[gwpy.timeseries.TimeSeries]
    """
    from gwpy.timeseries import TimeSeriesDict

    data_dict = TimeSeriesDict.get(channels, start, end)

    data = [data_dict[c] for c in channels]

    return data


def read_from_catalog(catalog, event, detectors, time_slice = None):
    """
    Read data from catalog

    :param catalog: the name of the catalog to read from
    :type catalog: str
    :param event: the name of the event
    :type event: str
    :param detectors: list of detectors
    :type detectors: list[str]
    :param time_slice: time slice for cropping the data, defaults to None
    :type time_slice: tuple, optional
    :return: (time series data, merger object)
    :rtype: tuple[list[pycbc.types.timeseries.TimeSeries], pycbc.catalog.Merger]
    """
    # Read data from catalog
    m = pycbc.catalog.Merger(event, source=catalog)
    data = [m.strain(ifo) for ifo in detectors]

    if time_slice:
        for i in range(len(data)):
            data[i] = data[i].crop(time_slice[0], time_slice[1])

    return data, m


# def _read_from_config(config):
#     # timer
#     timer_start = time.perf_counter()
#
#     # data = []
#     with Pool(processes=min(config.nproc, config.nIFO)) as pool:
#         data = pool.starmap(_read_from_config_wrapper, [(config, i) for i in range(len(config.ifo))])
#
#     # timer
#     timer_end = time.perf_counter()
#     logger.info(f'Read data from config in {timer_end - timer_start} seconds')
#     return data
#
#
# def _read_from_config_wrapper(config, i):
#     with open(config.frFiles[i], 'r') as f:
#         filenames = f.read()
#     # read data from the files
#     return read_from_gwf(i, config, filenames, config.channelNamesRaw[i])


def read_from_job_segment(config, job_seg: WaveSegment):
    """
    Read data from the frame files in job segment in parallel
    and merge them if there are more than one frame files for each ifo

    :param config: user configuration
    :type config: Config
    :param job_seg: job segment
    :type job_seg: WaveSegment
    :return: list of strain data
    :rtype: list[pycbc.types.timeseries.TimeSeries]
    """
    timer_start = time.perf_counter()

    # read data from the files in parallel
    with Pool(processes=min(config.nproc, len(job_seg.frames))) as pool:
        data = pool.starmap(_read_from_job_segment_wrapper, [
            (config, frame, job_seg) for frame in job_seg.frames
        ])

    merged_data = []

    # split data by ifo for next step of merging
    ifo_frames = [[i for i, frame in enumerate(job_seg.frames) if frame.ifo == ifo] for ifo in config.ifo]

    for frames in ifo_frames:
        if len(frames) == 1:
            # if there is only one frame, no need to merge
            ifo_data = data[frames[0]]
        else:
            # merge gw frames
            ifo_data = TimeSeries.from_pycbc(data[frames[0]])
            # free memory
            data[frames[0]] = None

            for i in frames[1:]:
                # use append method from gwpy, raise error if there is a gap
                ifo_data.append(TimeSeries.from_pycbc(data[i]), gap='raise')
                # free memory
                data[i] = None

            # convert back to pycbc
            ifo_data = ifo_data.to_pycbc()

        # check if data range match with job segment
        if ifo_data.start_time != job_seg.start_time - config.segEdge or \
                ifo_data.end_time != job_seg.end_time + config.segEdge:
            logger.error(f'Job segment {job_seg} not match with data {ifo_data}, '
                         f'the gwf data start at {ifo_data.start_time} and end at {ifo_data.end_time}')
            raise ValueError(f'Job segment {job_seg} not match with data {ifo_data}')

        # append to final data
        merged_data.append(ifo_data)

    timer_end = time.perf_counter()
    logger.info(f'Read data from job segment in {timer_end - timer_start} seconds')
    return merged_data


def _read_from_job_segment_wrapper(config, frame, job_seg: WaveSegment):
    # should read data with segment edge
    start = job_seg.start_time - config.segEdge
    end = job_seg.end_time + config.segEdge

    # for each frame, if the frame start time is later than the job segment start time, use the frame start time
    if frame.start_time > start:
        start = frame.start_time

    # for each frame, if the frame end time is earlier than the job segment end time, use the frame end time
    if frame.end_time < end:
        end = frame.end_time

    i = config.ifo.index(frame.ifo)
    data = read_from_gwf(frame.path, config.channelNamesRaw[i], start=start, end=end)
    if int(data.sample_rate.value) != int(config.inRate):
        data = data.resample(config.inRate)
        logger.info(f'Resample data from {data.sample_rate.value} to {config.inRate}')
    return check_and_resample(data, config, i)
