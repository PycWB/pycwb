import logging
import os
import psutil
from copy import copy
from pycwb.config import Config
from pycwb.modules.super_cluster.super_cluster import supercluster_wrapper
from pycwb.modules.super_cluster.supercluster import supercluster
from pycwb.modules.xtalk.monster import load_catalog
from pycwb.modules.coherence.coherence import coherence
from pycwb.modules.read_data import generate_injections, generate_noise_for_job_seg, read_from_job_segment, check_and_resample
from pycwb.modules.data_conditioning import data_conditioning
from pycwb.types.job import WaveSegment
from pycwb.types.network import Network
from pycwb.modules.workflow_utils.job_setup import print_job_info

logger = logging.getLogger(__name__)


def process_job_segment(working_dir: str, config: Config, job_seg: WaveSegment, compress_json: bool = True,
                        catalog_file: str = None, queue=None, production_mode: bool = False, skip_lags: list = None):
    """
    The core workflow to process single job segment with trails or lags. 

    Parameters
    ----------
    working_dir : str
        The working directory for the run
    config : Config
        The configuration object
    job_seg : WaveSegment
        The job segment to process
    compress_json : bool
        Whether to compress the json files
    catalog_file : str
        The catalog file to save the triggers
    queue : Queue
        The queue to send the triggers to the collector for saving in production mode
    production_mode : bool
        Whether to run in production mode, if True, the triggers will be sent to the queue instead of saving them in this function
    skip_lags : list
        The options to skip certain lags. It is used for resuming the processing after a crash
    
    """
    print_job_info(job_seg)

    if not job_seg.frames and not job_seg.noise and not job_seg.injections:
        raise ValueError("No data to process")

    base_data = None

    if job_seg.frames:
        base_data = read_from_job_segment(config, job_seg)
    if job_seg.noise:
        base_data = generate_noise_for_job_seg(job_seg, config.inRate, data=base_data)
    
    data = base_data
    
    # get all the trail_idx from the injections, if there is no injections, use 0
    trail_idxs = {0}
    if job_seg.injections:
        trail_idxs = set([injection.get('trail_idx', 0) for injection in job_seg.injections])
        
    # loop over all the trail_idx, if there is no injections or only one trail, only loop once for trail_idx=0
    for trail_idx in trail_idxs:
        if job_seg.injections:
            # use sub_job_seg for each trail_idx to avoid passing the trail_idx to the following functions,
            sub_job_seg = copy(job_seg)
            sub_job_seg.injections = [injection for injection in job_seg.injections if injection.get('trail_idx', 0) == trail_idx]
            logger.info(f"Processing trail_idx: {trail_idx} with {len(sub_job_seg.injections)} injections: {sub_job_seg.injections}")
            data = generate_injections(config, sub_job_seg, base_data)
        else:
            logger.info(f"Processing trail_idx: {trail_idx} without injections")
            sub_job_seg = job_seg
        # add trail_idx to the sub_job_seg
        sub_job_seg.trail_idx = trail_idx

        # check and resample the data
        data = [check_and_resample(data[i], config, i) for i in range(len(job_seg.ifos))]
        logger.info("Memory usage: %f.2 MB", psutil.Process().memory_info().rss / 1024 / 1024)

        # data conditioning
        tf_maps, nRMS_list = data_conditioning(config, data)
        logger.info("Memory usage: %f.2 MB", psutil.Process().memory_info().rss / 1024 / 1024)

        # initialize network object 
        network = Network(config, tf_maps, nRMS_list)

        # coherence
        fragment_clusters = coherence(config, tf_maps, nRMS_list, net=network)
        logger.info("Memory usage: %f.2 MB", psutil.Process().memory_info().rss / 1024 / 1024)

        # supercluster
        if config.use_root_supercluster:
            super_fragment_clusters = supercluster(config, network, fragment_clusters, tf_maps)
        else:
            xtalk_coeff, xtalk_lookup_table, layers, _ = load_catalog(config.MRAcatalog)
            super_fragment_clusters = supercluster_wrapper(config, network, fragment_clusters, tf_maps,
                                                        xtalk_coeff, xtalk_lookup_table, layers)
        logger.info("Memory usage: %f.2 MB", psutil.Process().memory_info().rss / 1024 / 1024)

        # only dump the first supercluster
        d = super_fragment_clusters[0]

        from pycwb.modules.cwb_conversions import convert_fragment_clusters_to_netcluster, convert_netcluster_to_fragment_clusters

        pwc = network.get_cluster(0)
        wdm_list = network.get_wdm_list()
        for wdm in wdm_list:
            wdm.setTDFilter(config.TDSize, config.upTDF)

        # load delay index
        network.set_delay_index(config.TDRate)

        # load time delay data
        pwc.cpf(convert_fragment_clusters_to_netcluster(d.dump_cluster(0)), False)
        pwc.setcore(False, 1)
        pwc.loadTDampSSE(network.net, 'a', config.BATCH, config.BATCH)


        from pycwb.modules.likelihoodWP.likelihood import load_data_from_ifo
        import numpy as np

        acor = network.net.acor
        network_energy_threshold = 2 * acor * acor * config.nIFO
        gamma_regulator = network.net.gamma * network.net.gamma * 2 / 3
        delta_regulator = abs(network.net.delta) if abs(network.net.delta) < 1 else 1
        REG = [delta_regulator * np.sqrt(2), 0, 0]
        netEC_threshold = network.net.netRHO * network.net.netRHO * 2

        n_sky = network.net.index.size()

        ml, FP, FX = load_data_from_ifo(network, config.nIFO)

        cluster_test = convert_netcluster_to_fragment_clusters(pwc)
        pixels = cluster_test.clusters[0].pixels

        # save FP, FX, rms, n_sky, gamma_regulator, network_energy_threshold to pickle
        test_data = {
            'FP': FP,
            'FX': FX,
            'cluster': cluster_test.clusters[0],
            'pixels': pixels,
            'n_ifo': config.nIFO,
            'ml': ml,
            'n_sky': n_sky,
            'delta_regulator': delta_regulator,
            'gamma_regulator': gamma_regulator,
            'netEC_threshold': netEC_threshold,
            'network_energy_threshold': network_energy_threshold,
            'netCC': network.net.netCC,
        }
        import pickle
        with open('test_data.pkl', 'wb') as f:
            pickle.dump(test_data, f)