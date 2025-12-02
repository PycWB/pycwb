import logging
from warnings import warn
import numpy as np
import orjson

from .dq_segment import read_seg_list, get_job_list, merge_seg_list
from .frame import get_frame_meta, select_frame_list, get_frame_files_from_gwdatafind
from pycwb.types.job import WaveSegment
from ..superlag import generate_slags
from ...utils.module import import_helper
from pycwb.modules.gracedb import get_superevent_t0
from pycwb.modules.injection.injection import generate_injection_list_from_config
from pycwb.modules.injection.par_generator import get_injection_list_from_parameters
logger = logging.getLogger(__name__)


def create_job_segment_from_config(config):
    """
    Create job segments based on the configuration file. Currently, the following cases are supported:
    1. pure simulation mode: only injection parameters are specified
    2. gps_start and gps_end are specified, only one job segment will be created
    3. gps_center, time_left, and time_right are specified, only one job segment will be created
    4. superevent, time_left, and time_right are specified, only one job segment will be created
    5. DQ files are specified

    The frame files will be attached to the job segments if fr_files are specified.
    Otherwise, the start and end times of the job segments can be used with the channel names to fetch the data.

    :param config: The configuration object.
    :type config: Config
    :return:
    :rtype: list[WaveSegment]
    """
    logger.info("-" * 80)
    logger.info("Initializing job segments")


    job_segments = None
    periods = None

    ############################################
    ## generate job segments based on the configuration if the DQ files or periods are specified
    ############################################

    ## generate the period with given gps times or superevent id
    # case 1: gps_start and gps_end are specified
    if config.gps_start and config.gps_end:
        periods = ([config.gps_start], [config.gps_end])
    # case 2: gps_center, time_left, and time_right are specified
    elif config.gps_center:
        gps_center = int(config.gps_center)
        if not config.time_left and not config.time_right:
            raise ValueError("Please specify either time_left or time_right for the job segment")
        periods = ([gps_center - config.time_left], [gps_center + config.time_right])
    # case 3: superevent, time_left, and time_right are specified
    elif config.superevent:
        if not config.time_left and not config.time_right:
            raise ValueError("Please specify either time_left or time_right for the superevent")
        gps_center = int(get_superevent_t0(config.superevent))
        periods = ([gps_center - config.time_left], [gps_center + config.time_right])

    ## create job segments
    if config.dq_files or periods:
        # get the job segments from the DQ files
        job_segments = job_segment_from_dq(config.dq_files, config.ifo,
                                           config.segLen, config.segMLS, config.segEdge, config.segOverlap,
                                           config.rateANA, config.l_high, config.inRate, periods=periods,
                                           slag_size=config.slagSize, slag_off=config.slagOff,
                                           slag_min=config.slagMin, slag_max=config.slagMax)

    ############################################
    ## injections on job segments
    ############################################

    ## Case 1 (Backward compatibility): if only simulation mode is specified without DQ files or periods,
    # create job segments based on the injection parameters
    if config.injection and job_segments is None:
        # TODO: split out the injection part for other job types
        job_segments = create_job_segment_from_injection(config.ifo, config.simulation, config.injection,
                                                         config.inRate, config.segEdge)
    ## Case 2: when the DQ files and simulation both are specified, inject the parameters into the job segments
    elif config.injection and job_segments is not None:
        start_gps_time = min([job_seg.start_time for job_seg in job_segments])
        end_gps_time = max([job_seg.end_time for job_seg in job_segments])
        injections, n_trails = generate_injection_list_from_config(config.injection, start_gps_time, end_gps_time)
        add_injections_into_job_segments(job_segments, injections)
        # add noise settings to the job segments if specified
        noise = config.injection.get('noise', None)
        if noise is not None:
            for job_seg in job_segments:
                job_seg.noise = {
                    'type': noise['type'],
                    'psds': [noise['psds'][ifo] for ifo in config.ifo],
                    'seeds': [noise['delta_seeds'][ifo] + job_seg.physical_start_times[ifo] for ifo in config.ifo],
                }

    ############################################
    ## Load the frames if frFiles or gwdatafind specified
    ############################################
    # attach the frame files to the job segments if defined
    if config.frFiles:
        # if frFiles exist, attach the given framefiles to the job segments
        attach_frame_files_to_job_segments(job_segments, config.ifo, config.frFiles, config.segEdge)

    if config.gwdatafind:
        # if gwdatafind specified, use gwdatafind to fetch the frame files for each job segment
        gwdatafind_frames_for_job_segments(job_segments, config.ifo, config.gwdatafind, config.segEdge)

    # attach the channel names to the job segments from config for self-contained information
    if config.channelNamesRaw:
        for job_seg in job_segments:
            job_seg.channels = config.channelNamesRaw

    ############################################
    ## TODO: check if the job segments are valid
    logger.info(f"Number of segments: {len(job_segments)}")
    logger.info("-" * 80)
    return job_segments


def job_segment_from_dq(dq_file_list, ifos, seg_len, seg_mls, seg_edge, seg_overlap, rateANA, l_high, sample_rate,
                        periods=None,
                       slag_size=0, slag_off=0, slag_min=0, slag_max=0, slag_site=0, slag_file=0):
    """Select a job segment from the database.

    :param dq_file_list: The list of DQ files.
    :type dq_file_list: list[DQFile]
    :param ifos: The list of interferometers.
    :type ifos: list[str]
    :param seg_len: The segment length.
    :type seg_len: int
    :param seg_mls: The minimum segment length after DQ_CAT1.
    :type seg_mls: int
    :param seg_edge: The wavelet boundary offset.
    :type seg_edge: int
    :param seg_overlap: The segment overlap.
    :type seg_overlap: int
    :param rateANA: The rate of the analysis.
    :type rateANA: int
    :param l_high: The high frequency cutoff.
    :type l_high: int
    :param sample_rate: The sample rate.
    :type sample_rate: int
    :param periods: Given start and stop periods, will be added with the dq_file_list
    :type periods: tuple[list[int], list[int]]
    :param slag_size: The super lag size.
    :type slag_size: int, optional
    :param slag_off: The super lag offset.
    :type slag_off: int, optional
    :param slag_min: The super lag minimum.
    :type slag_min: int, optional
    :param slag_max: The super lag maximum.
    :type slag_max: int, optional
    :param slag_site: The super lag site.
    :type slag_site: int, optional
    :param slag_file: The super lag file.
    :type slag_file: int, optional
    :return: The job segment.
    :rtype: WaveSegment
    """
    # merge the DQ segments for each interferometer
    seg_lists = []
    for ifo in ifos:
        dq_files = [dq_file for dq_file in dq_file_list if dq_file.ifo == ifo]
        cat1_list = read_seg_list(dq_files, 'CWB_CAT1', periods)
        seg_lists.append(cat1_list)

    # for zero super lag, merge the segments, and get the standard job segments
    merged_seg_list = seg_lists[0]
    for seg_list in seg_lists[1:]:
        merged_seg_list = merge_seg_list(merged_seg_list, seg_list)

    job_segments = []
    # for super lag, shift the cat1 list and merge the segments
    if slag_size > 0:
        # TODO: add checks on maximum super lag size
        slags = generate_slags(len(ifos), slag_min, slag_max, slag_off, slag_size)

        for slag in slags:
            slag_seg_lists = []
            for ifo, j in enumerate(slag):
                slag_seg_lists.append(np.array(seg_lists[ifo]) + j * seg_len)

            merged_slag_seg_list = slag_seg_lists[0]
            for seg_list in slag_seg_lists[1:]:
                merged_slag_seg_list = merge_seg_list(merged_slag_seg_list, seg_list)

            if len(merged_slag_seg_list) == 0 or len(merged_slag_seg_list[0]) == 0 or len(merged_slag_seg_list[1]) == 0:
                logger.warning(f"No segments found for super lag {slag}")
                raise ValueError(f"No segments found for super lag {slag}, please check the DQ files or periods")

            print('live time', merged_slag_seg_list[1][0] - merged_slag_seg_list[0][0])
            job_segments += get_job_list(ifos, merged_slag_seg_list, seg_len, seg_mls,
                                         seg_edge=seg_edge, sample_rate=sample_rate,
                                         shift=np.array(slag) * seg_len, index_start=len(job_segments))
    else:
        job_segments += get_job_list(ifos, merged_seg_list, seg_len, seg_mls, seg_edge, sample_rate)


    # cat1_list = read_seg_list(dq_file_list, 'CWB_CAT1', periods)
        #
        # # Get number/list of available super lag jobs
        # # Compute the available segments with length=segLen contained between the interval [min,max]
        # # Where min,max are the minimum and macimum times in the cat1List list
        # # The start time of each segment is forced to be a multiple of segLen
        # slag_job_list = get_slag_job_list(cat1_list, seg_len)
        #
        # slag_segments = len(slag_job_list)
        #
        # # Get super lag list : slagList
        # # Is the list of available segment shifts according to the slag configuration parameters
        # slag_list = get_slag_list(slag_segments, slag_size, slag_segments, slag_off, slag_min, slag_max, slag_site,
        #                           slag_file)
        #
        # for slag in slag_list:
        #     logger.info(f"SuperLag={slag.slag_id[0]} jobID={slag.job_id}")
        #     for n in range(len(ifos)):
        #         logger.info(f"segID[{slag.slag_id[1]}]={slag.seg_id[n]}")
        # raise Exception("Not finished")

    rate_min = rateANA >> l_high
    for job_seg in job_segments:
        # Check if seg length is compatible with WDM parity (only for 2G)
        # This condition is necessary to avoid mixing between odd
        # and even pixels when circular buffer is used for lag shift
        # The MRAcatalog distinguish odd and even pixels
        # If not compatible then the length is modified according the requirements
        length = job_seg.end_time - job_seg.start_time
        if int(length * rate_min + 0.001) & 1:
            job_seg.end_time -= 1

        # add segOverlap to the dataector's segments stop for this job
        job_seg.end_time += seg_overlap

        logger.debug(f"job segment gps range = {job_seg.start_time} - {job_seg.end_time}")
    logger.info(f"Number of job segments = {len(job_segments)}")

    return job_segments


def attach_frame_files_to_job_segments(job_segments, ifos, fr_files, seg_edge):
    # Get frame file list
    frame_files = {}
    for i, ifo in enumerate(ifos):
        frame_files[ifo] = get_frame_meta(fr_files[i], ifo)

    # Select frame files for each job segment
    for job_seg in job_segments:
        if job_seg.frames is None:
            job_seg.frames = []
        for ifo in ifos:
            job_seg.frames += select_frame_list(frame_files[ifo],
                                                job_seg.physical_start_times[ifo],
                                                job_seg.physical_end_times[ifo],
                                                seg_edge)
        # job_seg.frames = select_frame_list(frame_files, job_seg.start_time, job_seg.end_time, seg_edge)


def gwdatafind_frames_for_job_segments(job_segments, ifos, gwdatafind, seg_edge):
    """
    Use gwdatafind to get the frame files for the job segments

    :param job_segments: The job segments.
    :type job_segments: list[WaveSegment]
    :param ifos: The list of interferometers.
    :type ifos: list[str]
    :param gwdatafind: The gwdatafind setting.
    :type gwdatafind: dict
    :param seg_edge: The wavelet boundary offset.
    :type seg_edge: int

    :return: None
    """
    from gwdatafind import find_urls

    # prepare the gwdatafind.find_urls arguments
    if 'site' not in gwdatafind:
        gwdatafind['site'] = [ifo[0] for ifo in ifos]
    host = gwdatafind.get('host')

    for job_seg in job_segments:
        if job_seg.frames is None:
            job_seg.frames = []
        for i, ifo in enumerate(ifos):
            job_seg.frames += get_frame_files_from_gwdatafind(ifo, gwdatafind['site'][i], gwdatafind['frametype'][i],
                                                            job_seg.physical_start_times[ifo],
                                                            job_seg.physical_end_times[ifo],
                                                            seg_edge, host=host)
        # job_seg.frames = get_frame_files_from_gwdatafind(ifos, gwdatafind['site'], gwdatafind['frametype'],
        #                                                  job_seg.start_time, job_seg.end_time, seg_edge, host=host)


def create_job_segment_from_injection(ifo, simulation_mode, injection, sample_rate, seg_edge):
    warn("This function will be deprecated.", DeprecationWarning)
    # TODO: keep the job generation from noise but remove the injection insertion
    injections = get_injection_list_from_parameters(injection)

    # get the noise settings
    if 'noise' in injection['segment'] and injection['segment']['noise'] is not None:
        noise = injection['segment']['noise']
    else:
        noise = None

    # create the job segments with specified simulation mode
    if simulation_mode == "all_inject_in_one_segment":
        # inject all the parameters in one job segment
        job_segments = [WaveSegment(0, ifo, injection['segment']['start'], injection['segment']['end'],
                                    sample_rate=sample_rate, seg_edge=seg_edge,
                                    noise=noise, injections=injections)]
    elif simulation_mode == "one_inject_in_one_segment":
        # repeat the injection N times for the same job segment
        repeat = len(injections) #injection['segment']['repeat']
        # if len(injections) != repeat:
        #     raise ValueError(f"The number of injections ({len(injections)}) does not match the number of repeats ({repeat})")

        job_segments = [WaveSegment(i, ifo, injection['segment']['start'], injection['segment']['end'],
                                    sample_rate=sample_rate, seg_edge=seg_edge,
                                    noise=noise, injections=[injections[i]])
                        for i in range(repeat)]
    else:
        raise ValueError(f"Invalid simulation mode: {simulation_mode}")

    return job_segments


def add_injections_into_job_segments(job_segments, injections):
    """
    Associates injections with job segments based on overlapping time intervals.
    
    Parameters:
    job_segments (list): A list of job segment objects with start_time, end_time, and injections list.
    injections (list): A list of injection dictionaries with start_time and end_time.
    
    Returns:
    None: The function modifies job_segments in place.
    """
    for injection in injections:
        injection['start_time'] = injection['gps_time'] + injection['t_start']
        injection['end_time'] = injection['gps_time'] + injection['t_end']
    # Sort job segments and injections by start_time
    job_segments.sort(key=lambda x: x.start_time)
    injections.sort(key=lambda x: x["start_time"])
    
    injection_index = 0
    num_injections = len(injections)
    
    for segment in job_segments:
        segment.injections = []
        # Move injection index to the first relevant injection
        while injection_index < num_injections and injections[injection_index]["end_time"] < segment.start_time:
            injection_index += 1
        
        # Collect all overlapping injections
        i = injection_index
        while i < num_injections and injections[i]["start_time"] <= segment.end_time:
            if injections[i]["end_time"] >= segment.start_time:
                segment.injections.append(injections[i])
            i += 1


def save_job_segments_to_json(job_segments, output_file) -> None:
    """Save the job segments to a JSON file.

    :param job_segments: The job segments.
    :type job_segments: list[WaveSegment]
    :param output_file: The output file.
    :type output_file: str
    """
    with open(output_file, 'wb') as f:
        f.write(orjson.dumps(job_segments))


def load_job_segments_from_json(input_file: str) -> list[WaveSegment]:
    """Load the job segments from a JSON file.

    :param input_file: The input file.
    :type input_file: str
    :return: The job segments.
    :rtype: list[WaveSegment]
    """
    with open(input_file, 'rb') as f:
        data = orjson.loads(f.read())
    return data
