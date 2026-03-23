from gwpy.timeseries import TimeSeries
import numpy as np
import os
import logging
from multiprocessing import Pool
import time

from pycwb.types.time_series import TimeSeries as PycwbTimeSeries
from ..cwb_conversions import convert_to_wavearray, convert_wavearray_to_timeseries
from ..job_segment import WaveSegment

logger = logging.getLogger(__name__)


def read_from_gwf(filename, channel, start=None, end=None) -> TimeSeries:
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
    :rtype: gwpy.timeseries.TimeSeries
    """
    # Read data from GWF file
    if start or end:
        logger.info(f'Reading data from {filename} ({channel}) from {start} to {end}')
    else:
        logger.info(f'Reading data from {filename} ({channel})')

    # Read gwf file
    # if framefile is an osdf url, check if the file exist in current path or not, if so, read the local transferred file
    if filename.startswith("osdf://"):
        local_filename = os.path.basename(filename)
        if os.path.exists(local_filename):
            filename = local_filename
            logger.info(f'Using local transferred file: {local_filename}')
    data = TimeSeries.read(filename, channel, start, end)

    # check if data contains NaN values
    if np.isnan(np.sum(data)):
        if start and end:
            raise ValueError(f'Data from {filename} ({channel}) from {start} to {end} contains NaN values')
        else:
            raise ValueError(f'Data from {filename} ({channel}) contains NaN values')
    
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


def read_from_catalog(catalog, event, detectors, time_slice=None):
    """
    Read data from catalog via GWOSC (Gravitational-Wave Open Science Center).

    :param catalog: the name of the catalog (unused, kept for API compatibility)
    :type catalog: str
    :param event: the name of the event (e.g. 'GW150914')
    :type event: str
    :param detectors: list of detectors
    :type detectors: list[str]
    :param time_slice: time slice for cropping the data, defaults to None
    :type time_slice: tuple, optional
    :return: (time series data, event GPS time)
    :rtype: tuple[list[pycwb.types.time_series.TimeSeries], float]
    """
    from gwosc.datasets import event_gps
    gps = event_gps(event)
    # Fetch 32 s of open data centred on the event
    data = []
    for ifo in detectors:
        ts = TimeSeries.fetch_open_data(ifo, gps - 16, gps + 16)
        data.append(PycwbTimeSeries.from_gwpy(ts))

    if time_slice:
        for i in range(len(data)):
            data[i] = data[i].time_slice(time_slice[0], time_slice[1])

    return data, gps


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
    :rtype: list[pycwb.types.time_series.TimeSeries]
    """
    timer_start = time.perf_counter()

    # read data from the files in parallel
    if config.nproc > 1:
        logger.info(f'Read data from job segment {job_seg} in parallel')
        with Pool(processes=min(config.nproc, len(job_seg.frames))) as pool:
            data = pool.starmap(read_single_frame_from_job_segment, [
                (config, frame, job_seg) for frame in job_seg.frames
            ])
    else:
        data = [read_single_frame_from_job_segment(config, frame, job_seg) for frame in job_seg.frames]

    # merge the data
    merged_data = merge_frames(job_seg, data, config.segEdge)

    timer_end = time.perf_counter()
    logger.info(f'Read data from job segment in {timer_end - timer_start} seconds')
    return merged_data


def merge_frames(job_seg, data, seg_edge):
    logger.info(f'Merging data from job segment {job_seg.index}')
    merged_data = []

    # split data by ifo for next step of merging
    ifo_frames = [[i for i, frame in enumerate(job_seg.frames) if frame.ifo == ifo] for ifo in job_seg.ifos]

    for ifo, frames in zip(job_seg.ifos, ifo_frames):
        if len(frames) == 0:
            raise ValueError(
                f"No frame files found for IFO '{ifo}' in job segment {job_seg.index}. "
                f"Available frames cover: "
                f"{[f.ifo for f in job_seg.frames]}"
            )
        if len(frames) == 1:
            # if there is only one frame, no need to merge
            ifo_data = data[frames[0]]
        else:
            # sort frames by start time so gwpy.append receives them in chronological order
            frames = sorted(frames, key=lambda i: data[i].start_time)

            # merge gw frames via gwpy append
            ifo_data = data[frames[0]].to_gwpy()
            # free memory
            data[frames[0]] = None

            for i in frames[1:]:
                # use append method from gwpy, raise error if there is a gap
                ifo_data.append(data[i].to_gwpy(), gap='raise')
                # free memory
                data[i] = None

            # convert to pycwb TimeSeries
            ifo_data = PycwbTimeSeries.from_gwpy(ifo_data)

        # check if data range match with job segment
        if ifo_data.start_time != job_seg.padded_start or \
                ifo_data.end_time != job_seg.padded_end:
            raise ValueError(f'Job segment {job_seg} not match with data {ifo_data}, '
                             f'the gwf data start at {ifo_data.start_time} and end at {ifo_data.end_time}')

        logger.info(
            f'data info: start={ifo_data.start_time}, duration={ifo_data.duration}, rate={ifo_data.sample_rate}')
        # append to final data
        merged_data.append(ifo_data)

    return merged_data


def read_single_frame_from_job_segment(config, frame, job_seg: WaveSegment):
    """
    Read data from a single frame file in job segment, this functions also handles the shift defined in the WaveSegment.
    It will read the data with the physical start and end time of the frame, and shift the data to the data start time.
    The segment edge of the job segment will be added to the start and end time of the data.
    Additionally, it will resample the data if the sample rate of the data is different from the job segment sample rate.

    Parameters
    ----------
    config : Config
        user configuration
    frame : FrameFile
        frame file to read
    job_seg : WaveSegment
        job segment

    Returns
    -------
    TimeSeries
        strain data
    """
    # get the physical start and end time of the frame
    ifo = frame.ifo

    # should read data with segment edge
    start = job_seg.physical_padded_starts[ifo]
    end = job_seg.physical_padded_ends[ifo]
    data_start = job_seg.padded_start
    data_end = job_seg.padded_end

    # for each frame, if the frame start time is later than the job segment start time, use the frame start time
    if frame.start_time > start:
        # sync the data start time with the offset of frame start time
        data_start += frame.start_time - start
        start = frame.start_time

    # for each frame, if the frame end time is earlier than the job segment end time, use the frame end time
    if frame.end_time < end:
        # sync the data end time with the offset of frame end time
        data_end -= end - frame.end_time
        end = frame.end_time

    i = job_seg.ifos.index(frame.ifo)
    data = read_from_gwf(frame.path, job_seg.channels[i], start=start, end=end)
    logger.info(f'Read data: start={data.t0}, duration={data.duration}, rate={data.sample_rate}')
    # shift the time label of the physical data to the data start time
    data.shift(data_start - start)
    logger.info(f'Shift data: start={data.t0}, duration={data.duration}, rate={data.sample_rate}')
    if int(data.sample_rate.value) != int(job_seg.sample_rate):
        raise ValueError(f'Sample rate of the data {data.sample_rate} is different from the job segment sample rate {job_seg.sample_rate}, '
                         f'please check the frame file {frame.path} and the job segment {job_seg.index}')
        # sample_rate_old = data.sample_rate.value
        # w = convert_to_wavearray(data)
        # w.Resample(job_seg.sample_rate)
        # data = convert_wavearray_to_timeseries(w)
        # # data = data.resample(config.inRate)
        # logger.info(f'Resample data from {sample_rate_old} to {job_seg.sample_rate}')
    # return check_and_resample(data, config, i) # move this to the final step
    return PycwbTimeSeries.from_gwpy(data)
