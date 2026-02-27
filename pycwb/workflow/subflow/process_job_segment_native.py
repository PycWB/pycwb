import logging
import os
import time
import psutil
import numpy as np
from copy import copy
from pycwb.config import Config
from pycbc.types import TimeSeries
from pycwb.modules.super_cluster.super_cluster import supercluster_wrapper
from pycwb.modules.xtalk.monster import load_catalog
from pycwb.modules.cwb_coherence import coherence
from pycwb.modules.read_data import generate_strain_from_injection, generate_noise_for_job_seg, read_from_job_segment
from pycwb.modules.read_data.data_check import check_and_resample_py
from pycwb.modules.data_conditioning.data_conditioning_python import data_conditioning
from pycwb.modules.likelihoodWP.likelihood import likelihood
from pycwb.types.job import WaveSegment
from pycwb.modules.workflow_utils.job_setup import print_job_info
from pycwb.workflow.subflow.process_job_segment import create_single_trigger_folder, save_trigger, add_event_to_catalog

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
        base_data = generate_noise_for_job_seg(job_seg, config.inRate, f_low=config.fLow, data=base_data)

    # get all the trail_idx from the injections, if there is no injections, use 0
    trail_idxs = {0}
    if job_seg.injections:
        trail_idxs = set([injection.get('trail_idx', 0) for injection in job_seg.injections])

    wave_file = os.path.basename(catalog_file).replace('catalog', 'wave').replace('.json','.h5') if catalog_file is not None else None
    # loop over all the trail_idx, if there is no injections or only one trail, only loop once for trail_idx=0
    for trail_idx in trail_idxs:
        #creates new "clean" data for each trail_idx to avoid mixing different trail idxs at different cycles 
        data = base_data
        if job_seg.injections:
            # use sub_job_seg for each trail_idx to avoid passing the trail_idx to the following functions. 
            sub_job_seg = copy(job_seg)
            sub_job_seg.injections = [injection for injection in job_seg.injections if injection.get('trail_idx', 0) == trail_idx]
            logger.info(f"Processing trail_idx: {trail_idx} with {len(sub_job_seg.injections)} injections: {sub_job_seg.injections}")
            
            mdc = [TimeSeries(np.zeros(int(base_data[i].duration * base_data[i].sample_rate)), epoch = base_data[i].start_time, delta_t = 1/base_data[i].sample_rate) for i in range(len(sub_job_seg.ifos))]

            for injection in sub_job_seg.injections:
                inj = generate_strain_from_injection(injection, config, base_data[0].sample_rate, sub_job_seg.ifos) 
                #default argument copy = True prevents original base_data to be modified due to aliasing 
                mdc = [mdc[i].inject(inj[i]) for i in range(len(sub_job_seg.ifos))]
                data = [data[i].inject(inj[i]) for i in range(len(sub_job_seg.ifos))] 
        else:
            logger.info(f"Processing trail_idx: {trail_idx} without injections")
            sub_job_seg = job_seg
        # add trail_idx to the sub_job_seg
        sub_job_seg.trail_idx = trail_idx

        # check and resample the data
        data = [check_and_resample_py(data[i], config, i) for i in range(len(job_seg.ifos))]
        logger.info("Memory usage: %f.2 MB", psutil.Process().memory_info().rss / 1024 / 1024)

        # data conditioning
        stage_timer = time.perf_counter()
        strains, nRMS = data_conditioning(config, data)
        logger.info("Data conditioning time: %.2f s", time.perf_counter() - stage_timer)
        logger.info("Memory usage: %f.2 MB", psutil.Process().memory_info().rss / 1024 / 1024)

        # coherence
        stage_timer = time.perf_counter()
        fragment_clusters = coherence(config, strains)
        logger.info("Coherence time: %.2f s", time.perf_counter() - stage_timer)
        logger.info("Memory usage: %f.2 MB", psutil.Process().memory_info().rss / 1024 / 1024)

        stage_timer = time.perf_counter()
        xtalk_coeff, xtalk_lookup_table, layers, _ = load_catalog(config.MRAcatalog)
        # return_td_cache=True: always returns (clusters, td_inputs_cache) tuple
        super_fragment_clusters, wdm_td_cache = supercluster_wrapper(
            config, None, fragment_clusters, strains,
            xtalk_coeff, xtalk_lookup_table, layers,
            return_td_cache=True,
        )
        logger.info("Native supercluster_wrapper time: %.2f s", time.perf_counter() - stage_timer)
        logger.info("Memory usage: %f.2 MB", psutil.Process().memory_info().rss / 1024 / 1024)

        if super_fragment_clusters is None:
            logger.warning("No supercluster results for job segment %s trail_idx=%s", job_seg.index, trail_idx)
            continue

        checked = False
        likelihood_timer = time.perf_counter()
        for lag, fragment_cluster in enumerate(super_fragment_clusters):
            if skip_lags and lag in skip_lags:
                logger.info("Skipping lag %d due to skip_lags", lag)
                continue

            for k, selected_cluster in enumerate(fragment_cluster.clusters):
                if selected_cluster.cluster_status > 0:
                    continue

                result_cluster = likelihood(
                    config.nIFO,
                    selected_cluster,
                    config.MRAcatalog,
                    strains=strains,
                    config=config,
                    cluster_id=k + 1,
                    wdm_td_cache=wdm_td_cache,
                )
                if result_cluster is not None:
                    logger.info(
                        "likelihood td_amp reloaded: %s",
                        getattr(result_cluster, "_td_amp_reloaded", False),
                    )
                else:
                    logger.info(
                        "likelihood rejected cluster; td_amp reloaded: %s",
                        getattr(selected_cluster, "_td_amp_reloaded", False),
                    )
                checked = True
                break

            if checked:
                break

        logger.info("Native likelihood check time: %.2f s", time.perf_counter() - likelihood_timer)
        if not checked:
            logger.warning("No accepted clusters found for likelihood check")
