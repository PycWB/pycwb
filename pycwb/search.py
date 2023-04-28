import os, time
import multiprocessing
import pycwb
from pycwb.modules.plot.cluster_statistics import plot_statistics
from pycwb.types import Network
from pycwb.utils import logger_init
from pycwb.config import Config
from pycwb.modules.plot import plot_event_on_spectrogram
from pycwb.modules.read_data import read_from_job_segment, generate_injection
from pycwb.modules.data_conditioning import data_conditioning
from pycwb.modules.coherence import coherence
from pycwb.modules.super_cluster import supercluster
from pycwb.modules.likelihood import likelihood, save_likelihood_data
from pycwb.modules.job_segment import select_job_segment, create_job_segment_from_injection
from pycwb.modules.catalog import create_catalog
from pycwb.types.job import WaveSegment
import logging

logger = logging.getLogger(__name__)


def analyze_job_segment(config, job_seg):
    """Analyze one job segment with the given configuration

    This function includes the following stages:

    1. Read data from job segment (pycwb.modules.read_data.read_from_job_segment) \n
    2. Data conditioning (pycwb.modules.data_conditioning.data_conditioning) \n
    3. Create network (pycwb.modules.coherence.create_network) \n
    4. Coherence (pycwb.modules.coherence.coherence) \n
    5. Supercluster (pycwb.modules.super_cluster.supercluster) \n
    6. Likelihood (pycwb.modules.likelihood.likelihood) \n

    The results will be saved to the output directory in json format on likelihood stage

    :param config: configuration
    :type config: Config
    :param job_seg: job segment
    :type job_seg: WaveSegment
    """
    # config, job_seg = args
    start_time = time.perf_counter()

    job_id = job_seg.index
    # log job info
    logger.info(f"Job ID: {job_id}")
    logger.info(f"Start time: {job_seg.start_time}")
    logger.info(f"End time: {job_seg.end_time}")
    logger.info(f"Duration: {job_seg.end_time - job_seg.start_time}")
    if config.simulation != 0:
        data = generate_injection(config, job_seg)
    else:
        data = read_from_job_segment(config, job_seg)

    # data conditioning
    tf_maps, nRMS_list = data_conditioning(config, data)

    # calculate coherence
    # TODO: Merge resolution here?
    fragment_clusters = coherence(config, tf_maps, nRMS_list)

    # create network
    network = Network(config, tf_maps, nRMS_list)

    # supercluster
    pwc_list = supercluster(config, network, fragment_clusters, tf_maps)

    # likelihood
    events, clusters = likelihood(config, network, pwc_list)

    # save the results
    for i, event in enumerate(events):
        save_likelihood_data(job_id, i+1, config.outputDir, event, clusters[i])

    for i, tf_map in enumerate(tf_maps):
        plot_event_on_spectrogram(tf_map, events, filename=f'{config.outputDir}/events_{job_id}_all_{i}.png')

    # plot the likelihood map
    for i, cluster in enumerate(clusters):
        if cluster.cluster_status != -1:
            continue
        plot_statistics(cluster, 'likelihood', filename=f'{config.outputDir}/likelihood_map_{job_id}_{i+1}.png')
        plot_statistics(cluster, 'null', filename=f'{config.outputDir}/null_map_{job_id}_{i+1}.png')

    # calculate the performance
    end_time = time.perf_counter()
    logger.info("-" * 80)
    logger.info(f"Job {job_id} finished in {round(end_time - start_time, 1)} seconds")
    logger.info(f"Speed factor: {round((job_seg.end_time - job_seg.start_time) / (end_time - start_time), 1)}X")
    logger.info("-" * 80)


def search(user_parameters='./user_parameters.yaml', log_file=None, log_level='INFO', no_subprocess=False):
    """Main function to run the search

    This function will read the user parameters, select the job segments, create the catalog,
    copy the html and css files and run the search in subprocesses by default to avoid memory leak.

    :param user_parameters: path to user parameters file
    :type user_parameters: str
    :param log_file: path to log file, defaults to None
    :type log_file: str, optional
    :param log_level: log level, defaults to 'INFO'
    :type log_level: str, optional
    :param no_subprocess: run the search in the main process, defaults to False (Set to True for macOS development)
    :type no_subprocess: bool, optional
    """
    logger_init(log_file, log_level)
    logger.info("Logging initialized")
    logger.info("Logging level: " + log_level)
    logger.info("Logging file: " + str(log_file))

    # set env HOME_WAT_FILTERS
    if not os.environ.get('HOME_WAT_FILTERS'):
        pycwb_path = os.path.dirname(os.path.abspath(pycwb.__file__))
        os.environ['HOME_WAT_FILTERS'] = f"{os.path.abspath(pycwb_path)}/vendor"
        logger.info(f"Set HOME_WAT_FILTERS to {os.environ['HOME_WAT_FILTERS']}")

    # read config
    logger.info("Reading user parameters")
    config = Config(user_parameters)

    # create folder for output and log
    logger.info(f"Output folder: {config.outputDir}")
    logger.info(f"Log folder: {config.logDir}")
    if not os.path.exists(config.outputDir):
        os.makedirs(config.outputDir)
    if not os.path.exists(config.logDir):
        os.makedirs(config.logDir)

    if config.simulation == 0:
        logger.info("-" * 80)
        logger.info("Initializing job segments")
        job_segments = select_job_segment(config.dq_files, config.ifo, config.frFiles,
                                          config.segLen, config.segMLS, config.segEdge, config.segOverlap,
                                          config.rateANA, config.l_high)

        # log number of segments
        logger.info(f"Number of segments: {len(job_segments)}")
        logger.info("-" * 80)
    else:
        job_segments = create_job_segment_from_injection(config.simulation, config.injection)
        for job_seg in job_segments:
            logger.info(job_seg)

    # create catalog
    logger.info("Creating catalog file")
    create_catalog(f"{config.outputDir}/catalog.json", config, job_segments)

    # copy all files in web_viewer to output folder
    logger.info("Copying web_viewer files to output folder")
    import shutil
    web_viewer_path = os.path.dirname(os.path.abspath(pycwb.__file__)) + '/web_viewer'
    for file in os.listdir(web_viewer_path):
        shutil.copy(f'{web_viewer_path}/{file}', config.outputDir)

    # analyze job segments
    logger.info("Start analyzing job segments")
    for job_seg in job_segments:
        if no_subprocess:
            analyze_job_segment(config, job_seg)
            # gc.collect()
        else:
            process = multiprocessing.Process(target=analyze_job_segment, args=(config, job_seg))
            process.start()
            process.join()
