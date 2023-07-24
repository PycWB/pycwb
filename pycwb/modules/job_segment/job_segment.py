import logging
from .super_lag import get_slag_job_list, get_slag_list
from .dq_segment import read_seg_list, get_seg_list, get_job_list
from .frame import get_frame_meta, select_frame_list
from pycwb.types.job import WaveSegment
from ...utils.module import import_helper

logger = logging.getLogger(__name__)


def create_job_segment_from_config(config):
    if not config.simulation:
        logger.info("-" * 80)
        logger.info("Initializing job segments")
        job_segments = select_job_segment(config.dq_files, config.ifo, config.frFiles,
                                          config.segLen, config.segMLS, config.segEdge, config.segOverlap,
                                          config.rateANA, config.l_high)

        # log number of segments
        logger.info(f"Number of segments: {len(job_segments)}")
        logger.info("-" * 80)
    else:
        job_segments = create_job_segment_from_injection(config.ifo, config.simulation, config.injection)
        for job_seg in job_segments:
            logger.info(job_seg)
    return job_segments


def select_job_segment(dq_file_list, ifos, fr_files, seg_len, seg_mls, seg_edge, seg_overlap, rateANA, l_high,
                       slag_size=0, slag_off=0, slag_min=0, slag_max=0, slag_site=0, slag_file=0):
    """Select a job segment from the database.

    :param dq_file_list: The list of DQ files.
    :type dq_file_list: list[DQFile]
    :param ifos: The list of interferometers.
    :type ifos: list[str]
    :param fr_files: The list of frame files.
    :type fr_files: list[FrameFile]
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
    if slag_size > 0:
        # TODO: not finished
        cat1_list = read_seg_list(dq_file_list, 'CWB_CAT1')

        # Get number/list of available super lag jobs
        # Compute the available segments with length=segLen contained between the interval [min,max]
        # Where min,max are the minimum and macimum times in the cat1List list
        # The start time of each segment is forced to be a multiple of segLen
        slag_job_list = get_slag_job_list(cat1_list, seg_len)

        slag_segments = len(slag_job_list)

        # Get super lag list : slagList
        # Is the list of available segment shifts according to the slag configuration parameters
        slag_list = get_slag_list(slag_segments, slag_size, slag_segments, slag_off, slag_min, slag_max, slag_site,
                                  slag_file)

        for slag in slag_list:
            logger.info(f"SuperLag={slag.slag_id[0]} jobID={slag.job_id}")
            for n in range(len(ifos)):
                logger.info(f"segID[{slag.slag_id[1]}]={slag.seg_id[n]}")

        raise Exception("Not finished")

    else:
        cat1_list = read_seg_list(dq_file_list, 'CWB_CAT1')

        job_segments = get_job_list(ifos, cat1_list, seg_len, seg_mls, seg_edge)

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

    # Get frame file list
    frame_files = []
    for i in range(len(ifos)):
        frame_files += get_frame_meta(fr_files[i], ifos[i])

    # Select frame files for each job segment
    # TODO: this is only for simple job segment
    for job_seg in job_segments:
        job_seg.frames = select_frame_list(frame_files, job_seg.start_time, job_seg.end_time, seg_edge)

    return job_segments


def create_job_segment_from_injection(ifo, simulation_mode, injection):
    # get the injection parameters
    if 'parameters' in injection:
        if isinstance(injection['parameters'], list):
            injections = injection['parameters']
        else:
            injections = [injection['parameters']]
    elif 'parameters_from_python' in injection:
        # remove the .py extension if it exists
        module = import_helper(injection['parameters_from_python']['file'], "wf_gen")

        # get the injection parameters
        injections = getattr(module, injection['parameters_from_python']['function'])()

        if not isinstance(injections, list):
            raise ValueError('The function get_injection_parameters() should return a list of injection parameters')
    else:
        raise ValueError('No injection parameters specified, '
                         'please specify either parameters or parameters_from_python')

    # get the noise settings
    if 'noise' in injection['segment'] and injection['segment']['noise'] is not None:
        noise = injection['segment']['noise']
    else:
        noise = None

    # create the job segments with specified simulation mode
    if simulation_mode == "all_inject_in_one_segment":
        # inject all the parameters in one job segment
        job_segments = [WaveSegment(0, ifo, injection['segment']['start'], injection['segment']['end'],
                                    noise=noise, injections=injections)]
    elif simulation_mode == "one_inject_in_one_segments":
        # repeat the injection N times for the same job segment
        repeat = len(injections) #injection['segment']['repeat']
        # if len(injections) != repeat:
        #     raise ValueError(f"The number of injections ({len(injections)}) does not match the number of repeats ({repeat})")

        job_segments = [WaveSegment(i, ifo, injection['segment']['start'], injection['segment']['end'],
                                    noise=noise, injections=[injections[i]])
                        for i in range(repeat)]
    else:
        raise ValueError(f"Invalid simulation mode: {simulation_mode}")

    return job_segments

