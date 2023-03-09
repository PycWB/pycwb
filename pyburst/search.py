import os, time
from concurrent.futures import ProcessPoolExecutor as Pool
import multiprocessing
import pyburst
from pyburst.utils import logger_init
from pyburst.config import Config
from pyburst.modules.plot import plot_spectrogram
from pyburst.modules.read_data import read_from_job_segment
from pyburst.modules.data_conditioning import data_conditioning
from pyburst.modules.network import create_network
from pyburst.modules.coherence import coherence
from pyburst.modules.super_cluster import supercluster
from pyburst.modules.likelihood import likelihood
from pyburst.modules.job_segment import select_job_segment
from pyburst.modules.catalog import create_catalog
import logging

logger = logging.getLogger(__name__)


def analyze_job_segment(config, job_seg):
    """Analyze one job segment with the given configuration

    This function includes the following stages:

    1. Read data from job segment (pyburst.modules.read_data.read_from_job_segment) \n
    2. Data conditioning (pyburst.modules.data_conditioning.data_conditioning) \n
    3. Create network (pyburst.modules.coherence.create_network) \n
    4. Coherence (pyburst.modules.coherence.coherence) \n
    5. Supercluster (pyburst.modules.super_cluster.supercluster) \n
    6. Likelihood (pyburst.modules.likelihood.likelihood) \n

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

    data = read_from_job_segment(config, job_seg)

    # data conditioning
    tf_maps, nRMS_list = data_conditioning(config, data)

    # initialize network
    net, wdm_list = create_network(1, config, tf_maps, nRMS_list)

    # calculate coherence
    sparse_table_list, cluster_list = coherence(config, net, tf_maps, wdm_list)

    # supercluster
    pwc_list = supercluster(config, net, cluster_list, sparse_table_list)

    # likelihood
    events = likelihood(job_id, config, net, sparse_table_list, pwc_list, wdm_list)

    import matplotlib.pyplot as plt
    plot = plot_spectrogram(tf_maps[0], figsize=(24, 6), gwpy_plot=True)

    # plot boxes on the plot
    i = 0
    boxes = [[e.start[i], e.stop[i], e.low[i], e.high[i]] for e in events if len(e.start) > 0]

    for box in boxes:
        ax = plot.gca()
        ax.add_patch(plt.Rectangle((box[0], box[2]), box[1] - box[0], box[3] - box[2], linewidth=0.5, fill=False,
                                   color='red'))

    # save to png
    plot.savefig(f'{config.outputDir}/events_{job_id}_all.png')

    # save event to json
    # for i, event in enumerate(events):
    #     try:
    #         output = event.json()
    #         with open(f'{config.outputDir}/event_{job_id}_{i + 1}.json', 'w') as f:
    #             f.write(output)
    #     except Exception as e:
    #         logger.error(e)

    # save event to catalog
    # event_summary = [event.summary(job_id, i+1) for i, event in enumerate(events)]
    # try:
    #     add_events_to_catalog(f"{config.outputDir}/catalog.json", event_summary)
    # except Exception as e:
    #     logger.error(e)

    # del data, tf_maps, nRMS_list, net, wdm_list, sparse_table_list, cluster_list, pwc_list, events

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
        pyburst_path = os.path.dirname(os.path.abspath(pyburst.__file__))
        os.environ['HOME_WAT_FILTERS'] = f"{os.path.abspath(pyburst_path)}/vendor"
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

    logger.info("-" * 80)
    logger.info("Initializing job segments")
    job_segments = select_job_segment(config.dq_files, config.ifo, config.frFiles,
                                      config.segLen, config.segMLS, config.segEdge, config.segOverlap,
                                      config.rateANA, config.l_high)

    # log number of segments
    logger.info(f"Number of segments: {len(job_segments)}")
    logger.info("-" * 80)

    # create catalog
    logger.info("Creating catalog file")
    create_catalog(f"{config.outputDir}/catalog.json", config, job_segments)

    # copy all files in web_viewer to output folder
    logger.info("Copying web_viewer files to output folder")
    import shutil
    web_viewer_path = os.path.dirname(os.path.abspath(pyburst.__file__)) + '/web_viewer'
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
