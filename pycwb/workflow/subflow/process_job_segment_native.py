import gc
import logging
import os
import time
import psutil
import numpy as np
from copy import copy
from pycwb.config import Config
from pycbc.types import TimeSeries
from pycwb.modules.super_cluster.super_cluster import setup_supercluster, supercluster_single_lag
from pycwb.modules.xtalk.type import XTalk
from pycwb.modules.cwb_coherence import coherence
from pycwb.modules.cwb_coherence.coherence import setup_coherence, coherence_single_lag
from pycwb.modules.read_data import generate_strain_from_injection, generate_noise_for_job_seg, read_from_job_segment
from pycwb.modules.read_data.data_check import check_and_resample_py
from pycwb.modules.data_conditioning.data_conditioning_python import data_conditioning
from pycwb.modules.cwb_interop import create_cwb_workdir
from pycwb.modules.likelihoodWP.likelihood import likelihood, setup_likelihood
from pycwb.modules.qveto.qveto import get_qveto
from pycwb.modules.reconstruction import estimate_snr
from pycwb.types.job import WaveSegment
from pycwb.types.network_event import Event
from pycwb.modules.workflow_utils.job_setup import print_job_info
from pycwb.modules.catalog import Catalog
from pycwb.modules.workflow_utils import create_single_trigger_folder, save_trigger, add_event_to_catalog
from pycwb.utils.memory import release_memory
from pycwb.workflow.subflow.postprocess_and_plots import (
    plot_trigger_flow, reconstruct_waveforms_flow,
    reconstruct_INJwaveforms_flow, plot_skymap_flow,
)

logger = logging.getLogger(__name__)


def process_job_segment(working_dir: str, config: Config, job_seg: WaveSegment, compress_json: bool = True,
                        catalog_file: str = None, queue=None, production_mode: bool = False, skip_lags: list = None):
    """
    The core workflow to process single job segment with trails or lags. 

    .. versionchanged:: 0.31.2
       Memory optimizations (~560 MB reduction, ~28 % faster):

       * **malloc_trim after conditioning** – ``gc.collect()`` +
         ``ctypes.CDLL("libc.so.6").malloc_trim(0)`` returns fragmented
         glibc heap pages to the OS after data conditioning.
       * **Early MDC resampling & whitening** – MDC channels are resampled
         together with the data and whitened *before* the heavy
         coherence / supercluster setup, freeing ~315 MB earlier.
       * **Combined coherence → supercluster TD extraction** –
         ``setup_coherence(..., extract_td=True)`` extracts TD inputs from
         the complex TF maps before ``max_energy`` destroys them.  The
         resulting ``td_inputs_cache`` is passed to
         ``setup_supercluster(..., td_inputs_cache=...)``, eliminating a
         duplicate WDM transform pass (~275 MB temporary, ~3 s wall time).
       * **Float32 TD padded planes** – ``prepare_td_inputs()`` now stores
         padded planes in float32 instead of float64, halving the TD cache
         from ~565 MB to ~282 MB with no precision loss (Numba accumulates
         in float64 internally).

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

    if catalog_file is not None:
        base = os.path.basename(catalog_file)
        stem, _ = os.path.splitext(base)
        wave_file = stem.replace('catalog', 'wave') + '.h5'
    else:
        wave_file = None
    # loop over all the trail_idx, if there is no injections or only one trail, only loop once for trail_idx=0
    for trail_idx in trail_idxs:
        #creates new "clean" data for each trail_idx to avoid mixing different trail idxs at different cycles 
        data = base_data
        # if only one trail_idxs, remove the base_data reference to save memory
        if len(trail_idxs) == 1:
            base_data = None
        if job_seg.injections:
            # use sub_job_seg for each trail_idx to avoid passing the trail_idx to the following functions. 
            sub_job_seg = copy(job_seg)
            sub_job_seg.injections = [injection for injection in job_seg.injections if injection.get('trail_idx', 0) == trail_idx]
            logger.info(f"Processing trail_idx: {trail_idx} with {len(sub_job_seg.injections)} injections: {sub_job_seg.injections}")
            
            mdc = [TimeSeries(np.zeros(int(sub_job_seg.duration * sub_job_seg.sample_rate)), epoch = sub_job_seg.start_time, delta_t = 1/sub_job_seg.sample_rate) for i in range(len(sub_job_seg.ifos))]

            for injection in sub_job_seg.injections:
                inj = generate_strain_from_injection(injection, config, sub_job_seg.sample_rate, sub_job_seg.ifos) 
                #default argument copy = True prevents original base_data to be modified due to aliasing 
                mdc = [mdc[i].inject(inj[i]) for i in range(len(sub_job_seg.ifos))]
                data = [data[i].inject(inj[i]) for i in range(len(sub_job_seg.ifos))] 
        else:
            logger.info(f"Processing trail_idx: {trail_idx} without injections")
            sub_job_seg = job_seg
            mdc = None
        # add trail_idx to the sub_job_seg
        sub_job_seg.trail_idx = trail_idx

        # Export raw (injected) data + equivalent user_parameters.C for cWB comparison.
        # Must be called BEFORE check_and_resample_py so data is still at config.inRate.
        if getattr(config, 'cwb_compare', False):
            _cwb_compare_dir = getattr(config, 'cwb_compare_dir', '') or None
            create_cwb_workdir(working_dir, config, sub_job_seg, data,
                               cwb_compare_dir=_cwb_compare_dir)

        # check and resample the data
        data = [check_and_resample_py(data[i], config, i) for i in range(len(job_seg.ifos))]
        # Resample MDC at the same time; frees ~315 MB of 16 kHz data early
        if mdc is not None:
            mdc = [check_and_resample_py(mdc[i], config, i) for i in range(len(sub_job_seg.ifos))]
        logger.info("Memory usage: %f.2 MB", psutil.Process().memory_info().rss / 1024 / 1024)

        # data conditioning
        stage_timer = time.perf_counter()
        strains, nRMS = data_conditioning(config, data)
        data = None  # remove reference to original data to save memory
        release_memory()
        logger.info("Data conditioning time: %.2f s", time.perf_counter() - stage_timer)
        logger.info("Memory usage: %f.2 MB", psutil.Process().memory_info().rss / 1024 / 1024)

        # pre-condition the mdc (whitened injection maps) EARLY to free resampled MDC
        # before the memory-intensive coherence+supercluster setup.
        mdc_maps = None
        HoT_list = None
        if mdc is not None and hasattr(sub_job_seg, 'injections') and sub_job_seg.injections:
            from pycwb.modules.data_conditioning.whitening_mdc import whitening_mdc_py
            mdc_maps, HoT_list = zip(*[whitening_mdc_py(config, m, nrms) for m, nrms in zip(mdc, nRMS)])
            del mdc
            release_memory()

        # ---- One-time lag-independent setup ----------------------------------------
        # Use extract_td=True so coherence setup extracts TD inputs from the complex
        # TF maps *before* max_energy, avoiding a duplicate WDM pass in supercluster.
        stage_timer = time.perf_counter()
        coherence_setup = setup_coherence(config, strains, extract_td=True, job_seg=sub_job_seg)
        logger.info("Coherence setup time: %.2f s", time.perf_counter() - stage_timer)
        logger.info("Memory usage: %f.2 MB", psutil.Process().memory_info().rss / 1024 / 1024)

        # Build td_inputs_cache from coherence setup (avoids duplicate WDM in supercluster)
        td_inputs_cache = {}
        for setup in coherence_setup:
            level = setup["level"]
            layers = 2 ** level if level > 0 else 0
            wdm_layers = max(1, int(layers))
            td_inputs = setup.pop("td_inputs", None)
            if td_inputs is not None:
                td_inputs_cache[int(wdm_layers)] = td_inputs
                td_inputs_cache[int(wdm_layers) + 1] = td_inputs

        n_lag = sub_job_seg.n_lag
        logger.info("Lag plan: n_lag=%d (from job segment duration=%.2f s)", n_lag, sub_job_seg.duration)

        stage_timer = time.perf_counter()
        xtalk = XTalk.load(config.MRAcatalog)
        sc_setup = setup_supercluster(config, strains, td_inputs_cache=td_inputs_cache)
        wdm_td_cache = sc_setup["td_inputs_cache"]
        logger.info("Supercluster setup time: %.2f s", time.perf_counter() - stage_timer)
        logger.info("Memory usage: %f.2 MB", psutil.Process().memory_info().rss / 1024 / 1024)

        stage_timer = time.perf_counter()
        # Reuse sky arrays already computed by setup_supercluster; use the full-resolution
        # arrays (ml_likelihood/FP_likelihood/FX_likelihood) so the likelihood sky scan runs
        # at config.healpix resolution, not the reduced MIN_SKYRES_HEALPIX resolution.
        lh_setup = setup_likelihood(config, strains, config.nIFO,
                                    ml=sc_setup.get("ml_likelihood", sc_setup["ml"]),
                                    FP=sc_setup.get("FP_likelihood", sc_setup["FP"]),
                                    FX=sc_setup.get("FX_likelihood", sc_setup["FX"]))
        logger.info("Likelihood setup time: %.2f s", time.perf_counter() - stage_timer)
        logger.info("Memory usage: %f.2 MB", psutil.Process().memory_info().rss / 1024 / 1024)

        # ---- Streaming per-lag loop ---------------------------------------------------
        # For each lag: coherence (pixel selection + clustering) →
        #               supercluster (TD amps + subnet cut) →
        #               likelihood + output.
        # Expensive lag-independent work (TF maps, WDM decomposition, sky patterns)
        # was done once above and is reused here.
        likelihood_timer = time.perf_counter()
        for lag in range(n_lag):
            # timer for each lag iteration, not including the one-time setup above
            lag_timer = time.perf_counter()
            lag_shifts = sub_job_seg.lag_shifts[lag]
            lag_shift_str = ", ".join(
                f"{ifo}={shift:.3f}s"
                for ifo, shift in zip(sub_job_seg.ifos, lag_shifts)
            )
            logger.info("Processing lag %d / %d  [%s]", lag, n_lag - 1, lag_shift_str)
            if skip_lags and lag in skip_lags:
                logger.info("Skipping lag %d due to skip_lags", lag)
                continue

            # coherence for this lag only
            frag_clusters_this_lag = coherence_single_lag(coherence_setup, lag)

            # supercluster for this lag only
            fragment_cluster = supercluster_single_lag(sc_setup, config, frag_clusters_this_lag, lag, xtalk=xtalk)

            if fragment_cluster is None:
                logger.warning(
                    "No supercluster results for lag %d (job segment %s trail_idx=%s)",
                    lag, job_seg.index, trail_idx,
                )
                continue

            events_data = []
            for k, selected_cluster in enumerate(fragment_cluster.clusters):
                if selected_cluster.cluster_status > 0:
                    continue

                # Tag the cluster with its sequential 1-based ID for the event record
                selected_cluster.cluster_id = k + 1

                result_cluster, sky_stats = likelihood(
                    config.nIFO,
                    selected_cluster,
                    config.MRAcatalog,
                    cluster_id=k + 1,
                    wdm_td_cache=wdm_td_cache,
                    nRMS=nRMS,
                    setup=lh_setup,
                    xtalk=xtalk,
                    config=config,
                )

                if result_cluster is None or result_cluster.cluster_status != -1:
                    logger.info("likelihood rejected cluster %d in lag %d", k + 1, lag)
                    continue

                logger.info("likelihood accepted cluster %d in lag %d", k + 1, lag)

                # Construct Event from native cluster
                event = Event()
                event.output_py(sub_job_seg, result_cluster, config)
                event.job_id = sub_job_seg.index

                # Associate injections by GPS time overlap
                if sub_job_seg.injections:
                    for injection in sub_job_seg.injections:
                        # FIXME: for overlap signal, this won't work
                        if event.start[0] - 0.1 < injection['gps_time'] < event.stop[0] + 0.1:
                            event.injection = injection

                events_data.append((event, result_cluster, sky_stats))

            logger.info("Memory usage: %f.2 MB", psutil.Process().memory_info().rss / 1024 / 1024)

            #################### Save triggers and post-process ####################
            trigger_folders = []
            for trigger in events_data:
                trigger_folder = create_single_trigger_folder(working_dir, config.trigger_dir, sub_job_seg, trigger)
                trigger_folders.append(trigger_folder)
                save_trigger(trigger_folder=trigger_folder, trigger_data=trigger,
                             save_cluster=config.save_cluster, save_sky_map=config.save_sky_map)

            logger.info("Memory usage: %f.2 MB", psutil.Process().memory_info().rss / 1024 / 1024)

            # post-process and plot
            for trigger_folder, trigger in zip(trigger_folders, events_data):
                event, cluster_out, event_skymap_statistics = trigger

                # estimate reconstructed waveforms
                reconst_data = reconstruct_waveforms_flow(
                    trigger_folder, config, sub_job_seg.ifos,
                    event, cluster_out, epoch=sub_job_seg.start_time,
                    wave_file=wave_file, save=config.save_waveform, plot=config.plot_waveform,
                )

                # if injection, estimate injected waveforms and calculate statistics
                if event.injection:
                    injected_data = reconstruct_INJwaveforms_flow(
                        trigger_folder, config, sub_job_seg.ifos, event,
                        HoT_list, mdc_maps, config.iwindow / 2, config.segEdge, config.inRate,
                        wave_file=wave_file, save=config.save_injection, plot=config.plot_injection,
                    )
                    event.hrss      += injected_data['hrss']
                    event.time      += injected_data['central_time']
                    event.iSNR       = injected_data['snr']
                    event.frequency += injected_data['central_freq']
                    event.bandwidth += injected_data['bandwidth']
                    event.duration  += injected_data['duration']

                    inj_waveforms = injected_data['whitened_injected_waveform']
                    rec_waveforms = [reconst_data[f'{ifo}_wf_REC_whiten'] for ifo in sub_job_seg.ifos]
                    event.oSNR  = [estimate_snr(rec) for rec in rec_waveforms]
                    event.ioSNR = [
                        estimate_snr(inj, rec)
                        if (inj is not None) and (rec is not None) else None
                        for inj, rec in zip(inj_waveforms, rec_waveforms)
                    ]
                    del injected_data, inj_waveforms, rec_waveforms

                # Q-veto and Q-factor
                try:
                    min_qveto = 1e23
                    min_qfactor = 1e23
                    for ifo in sub_job_seg.ifos:
                        for a_type in ['DAT', 'REC']:
                            [qveto, qfactor] = get_qveto(reconst_data[f'{ifo}_wf_{a_type}_whiten'])
                            min_qveto = min(min_qveto, qveto)
                            min_qfactor = min(min_qfactor, qfactor)
                    event.Qveto = [min_qveto, min_qfactor]
                    event.qveto = min_qveto
                    event.qfactor = min_qfactor
                    logger.info("Qveto for event %s: %s, Qfactor: %s",
                                event.hash_id, event.qveto, event.qfactor)
                except Exception as e:
                    logger.error("Error calculating Qveto for event %s: %s", event.hash_id, e)

                if config.plot_trigger:
                    plot_trigger_flow(trigger_folder, event, cluster_out)

                if config.plot_sky_map:
                    plot_skymap_flow(trigger_folder, event, event_skymap_statistics)

                del reconst_data

            #################### Add events to catalog ####################
            for trigger in events_data:
                catalog_file = add_event_to_catalog(working_dir, config.catalog_dir,
                                                    trigger_data=trigger,
                                                    catalog_file=catalog_file)
            logger.info("-------------------------------------------")
            logger.info("Lag %d processing time: %.2f s", lag, time.perf_counter() - lag_timer)
            logger.info("Memory usage: %f.2 MB", psutil.Process().memory_info().rss / 1024 / 1024)
            logger.info("-------------------------------------------")

            # Release JAX device buffers and Python objects from this lag before
            # the next iteration.  JAX kernels hold device memory until GC runs;
            # with 100+ background lags this accumulates to GB of device memory.
            del frag_clusters_this_lag, fragment_cluster, events_data, trigger_folders
            gc.collect()

        logger.info("Native likelihood loop time: %.2f s", time.perf_counter() - likelihood_timer)
