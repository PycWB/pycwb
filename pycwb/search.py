import os, time
import multiprocessing
import logging
import pickle
import shutil
import click
import pycwb
import matplotlib.pyplot as plt

from pycwb.utils.dep_check import check_dependencies

if check_dependencies(['autoencoder', 'reconstruction', 'logger', 'read_data', 'data_conditioning', 'coherence',
                       'super_cluster', 'likelihood', 'job_segment', 'catalog', 'plot', 'plot_map', 'web_viewer']):
    exit(1)

from pycwb.config import Config
from pycwb.types.network import Network
from pycwb.modules.autoencoder import get_glitchness
from pycwb.modules.reconstruction import get_network_MRA_wave
from pycwb.modules.logger import logger_init
from pycwb.modules.read_data import read_from_job_segment, generate_injection
from pycwb.modules.data_conditioning import data_conditioning
from pycwb.modules.coherence import coherence
from pycwb.modules.super_cluster import supercluster
from pycwb.modules.likelihood import likelihood, save_likelihood_data
from pycwb.modules.job_segment import create_job_segment_from_config
from pycwb.modules.catalog import create_catalog, add_events_to_catalog
from pycwb.modules.plot.cluster_statistics import plot_statistics
from pycwb.modules.web_viewer.create import create_web_viewer
from pycwb.modules.plot_map.world_map import plot_world_map, plot_skymap_contour

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

    # read data
    data = None
    if job_seg.frames:
        data = read_from_job_segment(config, job_seg)
    if job_seg.injections:
        data = generate_injection(config, job_seg, data)

    # data conditioning
    tf_maps, nRMS_list = data_conditioning(config, data)

    # calculate coherence
    fragment_clusters = coherence(config, tf_maps, nRMS_list)

    # create network
    network = Network(config, tf_maps, nRMS_list)

    # supercluster
    pwc_list = supercluster(config, network, fragment_clusters, tf_maps)

    import ROOT
    import healpy as hp
    from pycwb.modules.cwb_conversions import convert_fragment_clusters_to_netcluster

    network.set_delay_index(config.TDRate)
    pwc = ROOT.netcluster()
    pwc.cpf(convert_fragment_clusters_to_netcluster(pwc_list[0].dump_cluster(0)), False)
    pwc.setcore(False, 1)
    pwc.loadTDampSSE(network.net, 'a', config.BATCH, config.BATCH)  # attach TD amp to pixels

    # skymap* nSkyStat, skymap* nSensitivity, skymap* nAlignment, skymap* nDisbalance,
    # skymap* nLikelihood, skymap* nNullEnergy, skymap* nCorrEnergy, skymap* nCorrelation,
    # skymap* nEllipticity, skymap* nPolarisation, skymap* nNetIndex, skymap* nAntenaPrior,
    # skymap* nProbability,
    keys = ['nSkyStat', 'nSensitivity', 'nAlignment', 'nDisbalance', 'nLikelihood', 'nNullEnergy', 'nCorrEnergy', 'nCorrelation', 'nEllipticity', 'nPolarisation', 'nNetIndex', 'nAntenaPrior', 'nProbability']
    skymaps = {key: ROOT.skymap(7) for key in keys}
    ifos = [network.get_ifo(i) for i in range(len(config.ifo))]
    skymap_index_size = hp.nside2npix(2**7)

    # double netCC, bool EFEC, double precision, double gamma, bool optim, double netRHO, double delta, double acor,
    count = ROOT.likelihoodWP(pwc, skymaps['nSkyStat'], skymaps['nSensitivity'], skymaps['nAlignment'], skymaps['nDisbalance'], skymaps['nLikelihood'], skymaps['nNullEnergy'], skymaps['nCorrEnergy'], skymaps['nCorrelation'], skymaps['nEllipticity'],
                      skymaps['nPolarisation'], skymaps['nNetIndex'], skymaps['nAntenaPrior'], skymaps['nProbability'], len(ifos), ifos, skymap_index_size,
                      network.net.skyMask.data, network.net.skyMaskCC.data, network.net.wdmMRA,
                      network.net.netCC, network.net.EFEC, network.net.precision, network.net.gamma, network.net.optim, network.net.netRHO, network.net.delta, network.net.acor,
                      config.search, 1, config.Search)
    return count
    # likelihood
    events, clusters, skymap_statistics = likelihood(config, network, pwc_list)

    # save the results
    for i, event in enumerate(events):
        save_likelihood_data(job_id, i+1, config.outputDir, event, clusters[i])
        # save event to catalog
        add_events_to_catalog(f"{config.outputDir}/catalog.json", event.summary(job_id, i+1))

    # for i, tf_map in enumerate(tf_maps):
    #     plot_event_on_spectrogram(tf_map, events, filename=f'{config.outputDir}/events_{job_id}_all_{i}.png')

    # reconstruct the waveforms
    for i, cluster in enumerate(clusters):
        if cluster.cluster_status != -1:
            continue
        reconstructed_waves = get_network_MRA_wave(config, cluster, config.rateANA, config.nIFO, network.net.rTDF,
                                                  'signal', 0, True)
        # plot the reconstructed wave
        for j, reconstructed_wave in enumerate(reconstructed_waves):
            plt.plot(reconstructed_wave.sample_times, reconstructed_wave.data)
            plt.xlim(events[i].left[0], events[i].left[0] + events[i].stop[0] - events[i].start[0])
            plt.savefig(f'{config.outputDir}/reconstructed_wave_job_{job_id}_cluster_{i+1}_ifo_{j+1}.png')
            plt.clf()

        # calculate the glitchness
        glitchness = get_glitchness(config, reconstructed_waves, events[i].sSNR, events[i].likelihood)
        # TODO: save to event file
        print(f"Glitchness: {glitchness}")

    # plot the likelihood map
    for i, cluster in enumerate(clusters):
        if cluster.cluster_status != -1:
            continue
        plot_statistics(cluster, 'likelihood', filename=f'{config.outputDir}/likelihood_map_{job_id}_{i+1}.png')
        plot_statistics(cluster, 'null', filename=f'{config.outputDir}/null_map_{job_id}_{i+1}.png')

    for i, event in enumerate(events):
        if event.nevent == 0:
            continue
        # plot_world_map(event.phi[0], event.theta[0], filename=f'{config.outputDir}/world_map_{job_id}_{i+1}.png')
        # TODO: plot in parallel
        for key in skymap_statistics[i].keys():
            plot_skymap_contour(skymap_statistics[i],
                                key=key,
                                reconstructed_loc=(event.phi[0], event.theta[0]),
                                detector_loc=(event.phi[3], event.theta[3]),
                                resolution=1,
                                filename=f'{config.outputDir}/{key}_{job_id}_{i+1}.png')
        # save the skymap statistics as pickle file
        with open(f'{config.outputDir}/skymap_statistics_{job_id}_{i+1}.pkl', 'wb') as f:
            pickle.dump(skymap_statistics[i], f)

    # calculate the performance
    end_time = time.perf_counter()
    logger.info("-" * 80)
    logger.info(f"Job {job_id} finished in {round(end_time - start_time, 1)} seconds")
    logger.info(f"Speed factor: {round((job_seg.end_time - job_seg.start_time) / (end_time - start_time), 1)}X")
    logger.info("-" * 80)


def search(user_parameters='./user_parameters.yaml', working_dir=".", log_file=None, log_level='INFO',
           no_subprocess=False, overwrite=False, nproc=None):
    """Main function to run the search

    This function will read the user parameters, select the job segments, create the catalog,
    copy the html and css files and run the search in subprocesses by default to avoid memory leak.

    Parameters
    ----------
    user_parameters : str, optional
        path to user parameters file, by default './user_parameters.yaml'
    working_dir : str, optional
        working directory, by default "."
    log_file : str, optional
        path to log file, by default None
    log_level : str, optional
        log level, by default 'INFO'
    no_subprocess : bool, optional
        run the search in the main process, by default False (Set to True for macOS development)
    overwrite : bool, optional
        overwrite the existing results, by default False
    nproc : int, optional
        number of threads to use, by default None (use the value in user parameters)
    """
    # create working directory
    working_dir = os.path.abspath(working_dir)
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)
    os.chdir(working_dir)

    # initialize logger
    logger_init(log_file, log_level)
    logger.info(f"Working directory: {os.path.abspath(working_dir)}")

    # set env HOME_WAT_FILTERS
    if not os.environ.get('HOME_WAT_FILTERS'):
        logger.warning("HOME_WAT_FILTERS is not set, default to pycwb/vendor")
        logger.warning("Please download the latest version of cwb config and set HOME_WAT_FILTERS to the path of folder XTALKS")
        pycwb_path = os.path.dirname(os.path.abspath(pycwb.__file__))
        os.environ['HOME_WAT_FILTERS'] = f"{os.path.abspath(pycwb_path)}/vendor"
        logger.info(f"Set HOME_WAT_FILTERS to {os.environ['HOME_WAT_FILTERS']}")

    # read config
    logger.info("Reading user parameters")
    config = Config(user_parameters)

    # overwrite threads if it is set
    if nproc:
        config.nproc = nproc

    # Safety Check: if output is not empty, ask for confirmation
    if os.path.exists(config.outputDir) and os.listdir(config.outputDir):
        if overwrite:
            logger.info(f"Overwrite output directory {config.outputDir}")
        elif not click.confirm(f"Output directory {config.outputDir} is not empty, do you want to continue?"):
            logger.info("Search stopped")
            return

    # create folder for output and log
    logger.info(f"Output folder: {working_dir}/{config.outputDir}")
    logger.info(f"Log folder: {working_dir}/{config.logDir}")
    if not os.path.exists(config.outputDir):
        os.makedirs(config.outputDir)
    if not os.path.exists(config.logDir):
        os.makedirs(config.logDir)

    # copy user parameters to output folder if it is not there
    if not os.path.exists(f"{config.outputDir}/user_parameters.yaml"):
        shutil.copyfile(user_parameters, f"{config.outputDir}/user_parameters.yaml")
    else:
        logger.warning(f"User parameters file already exists in {working_dir}/{config.outputDir}")

    # select job segments
    job_segments = create_job_segment_from_config(config)

    # create catalog
    logger.info("Creating catalog file")
    create_catalog(f"{config.outputDir}/catalog.json", config, job_segments)

    # copy all files in web_viewer to output folder
    create_web_viewer(config.outputDir)

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
