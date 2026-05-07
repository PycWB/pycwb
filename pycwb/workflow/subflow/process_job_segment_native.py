import gc
import logging
import os
import time
import psutil
import numpy as np
from copy import copy
from pycwb.config import Config
from pycwb.types.time_series import TimeSeries
from pycwb.modules.super_cluster.super_cluster import setup_supercluster, supercluster_single_lag
from pycwb.utils.td_vector_batch import build_td_inputs_cache
from pycwb.modules.xtalk.type import XTalk
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
from pycwb.modules.workflow_utils.job_setup import print_job_info, print_node_info
from pycwb.modules.workflow_utils import create_single_trigger_folder, save_trigger
from pycwb.types.trigger import Trigger
from pycwb.utils.memory import release_memory
from pycwb.modules.job_segment import build_injection_veto_windows, intersect_intervals
from pycwb.workflow.subflow.postprocess_and_plots import (
    plot_trigger_flow, reconstruct_waveforms_flow,
    reconstruct_INJwaveforms_flow, plot_skymap_flow,
)

logger = logging.getLogger(__name__)


def process_job_segment(working_dir: str, config: Config, job_seg: WaveSegment, compress_json: bool = True,
                        catalog_file: str = None, queue=None, production_mode: bool = False,
                        skip_lags: dict[int, set[int]] | None = None):
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
    # ─────────────────────────────────────────────────────────────────────────
    # HIGH-LEVEL WORKFLOW OVERVIEW
    # ─────────────────────────────────────────────────────────────────────────
    # For each trial_idx (injection realisation, or 0 if no injections):
    #
    #   1. DATA LOADING     – read frames / generate noise / inject signals
    #   2. CONDITIONING     – resample → whiten (data + MDC injections)
    #   3. ONE-TIME SETUP   – coherence, TD-input cache, supercluster, likelihood
    #                         (expensive; computed once, reused across all lags)
    #   4. PER-LAG LOOP     – for each time-slide lag:
    #        a. Coherence      → pixel selection + fragment clustering
    #        b. Supercluster   → TD amplitudes + subnet veto
    #        c. Likelihood     → sky scan + reconstruction veto
    #        d. Post-process   → waveforms, injections, Q-veto, plots
    #        e. Catalog        → persist triggers and release lag memory
    # ─────────────────────────────────────────────────────────────────────────
    print_job_info(job_seg)
    print_node_info()
    job_timer = time.perf_counter()  # total wall-time for this job segment

    if not job_seg.frames and not job_seg.noise and not job_seg.injections:
        raise ValueError("No data to process")

    base_data = None

    if job_seg.frames:
        base_data = read_from_job_segment(config, job_seg)
    if job_seg.noise:
        base_data = generate_noise_for_job_seg(job_seg, config.inRate, f_low=config.fLow, data=base_data)

    # get all the trial_idx from the injections, if there is no injections, use 0
    trial_idxs = {0}
    if job_seg.injections:
        trial_idxs = set([inj.get('trial_idx', 0) for inj in job_seg.injections])

    if catalog_file is not None:
        base = os.path.basename(catalog_file)
        stem, _ = os.path.splitext(base)
        wave_file = stem.replace('catalog', 'wave') + '.h5'
    else:
        wave_file = None

    # Outer loop: one iteration per injection trial (or a single pass when there are no injections).
    for trial_idx in trial_idxs:
        # check if all lags for this trial_idx are in skip_lags, if so, skip this trial_idx entirely
        if skip_lags and trial_idx in skip_lags and len(skip_lags[trial_idx]) >= job_seg.n_lag:
            logger.info(f"Skipping trial_idx {trial_idx} entirely: all {job_seg.n_lag} lags already completed")
            continue

        trial_timer = time.perf_counter()  # wall-time for this trial
        # ─────────────────────────────────────────────────────────────────────
        # STEP 1 – DATA LOADING & INJECTION SETUP
        # ─────────────────────────────────────────────────────────────────────
        # Single trial: take ownership of base_data directly (no copy needed).
        # Multiple trials: copy so injections from one trial don't bleed into the next.
        if len(trial_idxs) > 1:
            data = [ts.copy() for ts in base_data] if base_data is not None else None
        else:
            data = base_data
            base_data = None

        if job_seg.injections:
            # use sub_job_seg for each trial_idx to avoid passing the trial_idx to the following functions. 
            sub_job_seg = copy(job_seg)
            sub_job_seg.injections = [injection for injection in job_seg.injections
                                      if injection.get('trial_idx', 0) == trial_idx]
            logger.info(f"Processing trial_idx: {trial_idx} with {len(sub_job_seg.injections)} injections: {sub_job_seg.injections}")
            
            # TODO: rename all MDC and injection into simulation
            # Allocate a zero-filled MDC (Mock data challenge) buffer for each IFO.
            # MDC buffer must span the full padded window [padded_start, padded_end] so that
            # whitening_mdc and get_INJ_waveform see the same time axis as the conditioned strains.
            mdc = [TimeSeries(data=np.zeros(int(sub_job_seg.padded_duration * sub_job_seg.sample_rate)),
                              t0=sub_job_seg.padded_start,
                              dt=1 / sub_job_seg.sample_rate)
                   for i in range(len(sub_job_seg.ifos))]

            injection_envelopes = []
            for injection in sub_job_seg.injections:
                inj = generate_strain_from_injection(injection, config, sub_job_seg.sample_rate, sub_job_seg.ifos)
                # Track signal timing envelope across all IFOs.
                n_ifo = len(sub_job_seg.ifos)
                real_start = min(float(inj[i].t0) for i in range(n_ifo))
                real_end = max(float(inj[i].t0) + len(inj[i].data) * float(inj[i].dt)
                               for i in range(n_ifo))
                injection['real_start'] = real_start
                injection['real_end'] = real_end
                # Both mdc and data are our own buffers — always inject in-place.
                for i in range(n_ifo):
                    mdc[i].inject(inj[i], copy=False)
                for i in range(n_ifo):
                    data[i].inject(inj[i], copy=False)
                # Free the per-injection signal buffer immediately to reduce peak memory.
                del inj
        else:
            logger.info(f"Processing trial_idx: {trial_idx} without injections")
            sub_job_seg = job_seg
            mdc = None
        # add trial_idx to the sub_job_seg
        sub_job_seg.trial_idx = trial_idx

        # ─────────────────────────────────────────────────────────────────────
        # BENCHMARK ONLY – cWB comparison workdir
        # ─────────────────────────────────────────────────────────────────────
        # Dumps raw (injected) data and a matching user_parameters.C so the
        # same segment can be replayed by native cWB for side-by-side checks.
        # Must run BEFORE resampling so the data is still at config.inRate.
        if getattr(config, 'cwb_compare', False):
            _cwb_compare_dir = getattr(config, 'cwb_compare_dir', '') or None
            create_cwb_workdir(working_dir, config, sub_job_seg, data,
                               cwb_compare_dir=_cwb_compare_dir)

        # ─────────────────────────────────────────────────────────────────────
        # STEP 2 – RESAMPLING & DATA CONDITIONING
        # ─────────────────────────────────────────────────────────────────────
        # Resample data (and MDC simultaneously) to config.fSample.  Doing MDC
        # here – rather than after whitening – frees the high-sample-rate buffers
        # before the heavy coherence/supercluster allocations below.
        data = [check_and_resample_py(data[i], config, i) for i in range(len(job_seg.ifos))]
        if mdc is not None:
            mdc = [check_and_resample_py(mdc[i], config, i) for i in range(len(sub_job_seg.ifos))]
        logger.info("Memory usage: %f.2 MB", psutil.Process().memory_info().rss / 1024 / 1024)

        # Whiten and normalise: produces conditioned strains and per-IFO noise RMS.
        stage_timer = time.perf_counter()
        strains, nRMS = data_conditioning(config, data)
        data = None  # raw data no longer needed; drop reference to free memory
        release_memory()
        logger.info("Data conditioning time: %.2f s", time.perf_counter() - stage_timer)
        logger.info("Memory usage: %f.2 MB", psutil.Process().memory_info().rss / 1024 / 1024)

        # Whiten the MDC buffers EARLY (before coherence + supercluster setup)
        # so the resampled high-sample-rate MDC arrays can be freed sooner.
        mdc_maps = None
        HoT_list = None
        if mdc is not None and hasattr(sub_job_seg, 'injections') and sub_job_seg.injections:
            from pycwb.modules.data_conditioning.whitening_mdc import whitening_mdc_py
            mdc_maps, HoT_list = zip(*[whitening_mdc_py(config, m, nrms) for m, nrms in zip(mdc, nRMS)])
            del mdc
            release_memory()

        # ─────────────────────────────────────────────────────────────────────
        # STEP 3 – ONE-TIME LAG-INDEPENDENT SETUP
        # ─────────────────────────────────────────────────────────────────────
        # Everything below is computed once per trail and then reused by every
        # lag iteration.  The ordering matters for memory:  coherence builds
        # WDM TF maps → TD cache is derived from those strains → supercluster
        # and likelihood share the same sky-pattern arrays.

        # 3a. Coherence setup: WDM decomposition + TF maps for all IFOs.
        stage_timer = time.perf_counter()
        coherence_setup = setup_coherence(config, strains, job_seg=sub_job_seg)
        logger.info("Coherence setup time: %.2f s", time.perf_counter() - stage_timer)
        logger.info("Memory usage: %f.2 MB", psutil.Process().memory_info().rss / 1024 / 1024)

        # 3b. TD-input cache: float32 padded planes for the supercluster step.
        #     Stored in float32 (vs. float64) to halve memory usage;
        #     Numba accumulates in float64 internally, so precision is preserved.
        stage_timer = time.perf_counter()
        td_inputs_cache = build_td_inputs_cache(config, strains)
        logger.info("TD inputs cache build time: %.2f s", time.perf_counter() - stage_timer)
        logger.info("Memory usage: %f.2 MB", psutil.Process().memory_info().rss / 1024 / 1024)

        n_lag = sub_job_seg.n_lag
        logger.info("Lag plan: n_lag=%d (from job segment duration=%.2f s)", n_lag, sub_job_seg.duration)

        gps_time = float(strains[0].start_time)

        # 3c. Supercluster setup: sky patterns (FP/FX/ml) + cross-talk catalog.
        stage_timer = time.perf_counter()
        xtalk = XTalk.load(config.MRAcatalog)
        supercluster_setup = setup_supercluster(config, gps_time)
        logger.info("Supercluster setup time: %.2f s", time.perf_counter() - stage_timer)
        logger.info("Memory usage: %f.2 MB", psutil.Process().memory_info().rss / 1024 / 1024)

        # 3d. Likelihood setup: reuse full-resolution sky arrays from supercluster
        #     so the sky scan runs at config.healpix resolution without rebuilding them.
        stage_timer = time.perf_counter()
        likelihood_setup = setup_likelihood(config, strains, config.nIFO,
                                    ml=supercluster_setup.get("ml_likelihood", supercluster_setup["ml"]),
                                    FP=supercluster_setup.get("FP_likelihood", supercluster_setup["FP"]),
                                    FX=supercluster_setup.get("FX_likelihood", supercluster_setup["FX"]))
        logger.info("Likelihood setup time: %.2f s", time.perf_counter() - stage_timer)
        logger.info("Memory usage: %f.2 MB", psutil.Process().memory_info().rss / 1024 / 1024)

        # ─────────────────────────────────────────────────────────────────────
        # STEP 4 – STREAMING PER-LAG LOOP
        # ─────────────────────────────────────────────────────────────────────
        # Lag 0 is the zero-lag (no additional time slide within the segment).
        # Note: the segment itself may already be superlagged, so lag 0 is not
        # necessarily on-source.  Lags 1…n_lag-1 apply additional IFO-specific
        # time shifts for background estimation.
        # Each iteration runs coherence → supercluster → likelihood in sequence
        # and then saves the accepted triggers before releasing lag-local memory.
        likelihood_timer = time.perf_counter()
        for lag in range(n_lag):
            lag_timer = time.perf_counter()  # measures only this lag, not the setup above
            lag_shifts = sub_job_seg.lag_shifts[lag]
            lag_shift_str = ", ".join(
                f"{ifo}={shift:.3f}s"
                for ifo, shift in zip(sub_job_seg.ifos, lag_shifts)
            )
            logger.info("Processing lag %d / %d  [%s]", lag, n_lag - 1, lag_shift_str)
            if skip_lags and lag in skip_lags.get(trial_idx, set()):
                logger.info("Skipping lag %d (trial %d) due to skip_lags", lag, trial_idx)
                continue

            # ── segTHR check ─────────────────────────────────────────────────
            # If CAT2 veto_windows are set, the per-lag livetime may be too
            # short for meaningful analysis.  Skip lags below config.segTHR.
            seg_thr = getattr(config, 'segTHR', 0.0) or 0.0
            if seg_thr > 0 and sub_job_seg.veto_windows:
                lag_livetime = sub_job_seg.livetime(lag)
                if lag_livetime < seg_thr:
                    logger.warning(
                        "Skipping lag %d: post-CAT2 livetime %.2f s < segTHR %.2f s",
                        lag, lag_livetime, seg_thr,
                    )
                    progress_record = dict(
                        job_id=sub_job_seg.index, trial_idx=trial_idx, lag_idx=lag,
                        n_triggers=0, livetime=lag_livetime, status="skipped_segTHR",
                    )
                    if queue is not None:
                        queue.put({"type": "progress", **progress_record})
                    elif catalog_file:
                        from pycwb.modules.catalog.catalog import Catalog
                        Catalog.open(catalog_file).add_lag_progress(**progress_record)
                    continue
                else:
                    logger.info(
                        "Processing lag %d: post-CAT2 livetime %.2f s >= segTHR %.2f s, lost %.2f s",
                        lag, lag_livetime, seg_thr, sub_job_seg.duration - lag_livetime,
                    )


            # Populate veto_windows when injection-only analysis is enabled.
            # If CAT2 veto_windows already exist on the job segment, intersect
            # with the injection windows so both conditions must be satisfied.
            if getattr(config, 'analyze_injection_only', False) and job_seg.injections:
                injection_envelopes = [(inj['real_start'], inj['real_end']) for inj in sub_job_seg.injections]
                inj_windows = build_injection_veto_windows(
                    injection_envelopes,
                    padding=getattr(config, 'injection_padding', 1.0),
                    duration=sub_job_seg.duration,
                )
                if sub_job_seg.veto_windows:
                    sub_job_seg.veto_windows = intersect_intervals(
                        sorted(sub_job_seg.veto_windows), sorted(inj_windows),
                    )
                else:
                    sub_job_seg.veto_windows = inj_windows
                    
            # ── 4a. Coherence ────────────────────────────────────────────────
            # Pixel selection and fragment clustering for this specific time slide.
            timer_coherence = time.perf_counter()
            frag_clusters_this_lag = coherence_single_lag(
                coherence_setup, lag,
                veto_windows=sub_job_seg.veto_windows,
            )
            logger.info("Coherence time for lag %d: %.2f s", lag, time.perf_counter() - timer_coherence)

            # ── 4b. Supercluster ─────────────────────────────────────────────
            # Compute TD amplitudes, apply subnet veto, and merge fragments.
            timer_supercluster = time.perf_counter()
            fragment_cluster = supercluster_single_lag(
                supercluster_setup, config, frag_clusters_this_lag, lag,
                xtalk=xtalk, td_inputs_cache=td_inputs_cache,
            )
            logger.info("Supercluster time for lag %d: %.2f s", lag, time.perf_counter() - timer_supercluster)

            if fragment_cluster is None:
                logger.warning(
                    "No supercluster results for lag %d (job segment %s trial_idx=%s)",
                    lag, job_seg.index, trial_idx,
                )
                # Still record progress for zero-trigger lags
                progress_record = dict(
                    job_id=sub_job_seg.index, trial_idx=trial_idx, lag_idx=lag,
                    n_triggers=0, livetime=sub_job_seg.livetime(lag),
                    status="completed",
                )
                if queue is not None:
                    queue.put({"type": "progress", **progress_record})
                elif catalog_file:
                    from pycwb.modules.catalog.catalog import Catalog
                    Catalog.open(catalog_file).add_lag_progress(**progress_record)
                continue

            # ── 4c. Likelihood ───────────────────────────────────────────────
            # Run sky scan and reconstruction veto on every surviving cluster.
            events_data = []
            for k, selected_cluster in enumerate(fragment_cluster.clusters):
                if selected_cluster.cluster_status > 0:
                    continue
                selected_cluster.cluster_id = k + 1
                result_cluster, sky_stats = likelihood(
                    config.nIFO, selected_cluster, config,
                    cluster_id=k + 1, nRMS=nRMS, setup=likelihood_setup, xtalk=xtalk,
                )
                if result_cluster is None or result_cluster.cluster_status != -1:
                    continue
                logger.info(
                    "likelihood accepted cluster %d in lag %d (%d pixels, from %.2f - %.2f s with freq %.2f - %.2f Hz)",
                    result_cluster.cluster_id, lag, len(result_cluster.pixel_arrays),
                    result_cluster.start_time, result_cluster.stop_time,
                    result_cluster.low_frequency, result_cluster.high_frequency,
                )

                # Construct Event from native cluster
                event = Event()
                event.output_py(sub_job_seg, result_cluster, config)
                event.job_id = sub_job_seg.index
                event.trial_idx = trial_idx
                event.lag_idx = lag
                event.id = event.long_id  # recompute id with job_id, trial_idx, lag_idx

                # Match this event to its injection by GPS overlap.
                # FIXME: overlapping signals share the same GPS window; only the
                #        first matching injection is associated here.
                if sub_job_seg.injections:
                    for injection in sub_job_seg.injections:
                        if event.start[0] - 0.1 < injection['gps_time'] < event.stop[0] + 0.1:
                            event.injection = injection

                events_data.append((event, result_cluster, sky_stats))

            logger.info("Memory usage: %f.2 MB", psutil.Process().memory_info().rss / 1024 / 1024)

            # ── 4d. Save triggers & post-process ────────────────────────────
            # Persist raw cluster data on disk first, then run optional
            # waveform reconstruction, injection comparison, Q-veto, and plots.
            trigger_folders = []
            for trigger in events_data:
                trigger_folder = create_single_trigger_folder(working_dir, config.trigger_dir, sub_job_seg, trigger)
                trigger_folders.append(trigger_folder)
                save_trigger(trigger_folder=trigger_folder, trigger_data=trigger,
                             save_cluster=config.save_cluster, save_sky_map=config.save_sky_map)

            logger.info("Memory usage: %f.2 MB", psutil.Process().memory_info().rss / 1024 / 1024)

            # Post-process each trigger: waveform reconstruction → injection stats
            # → Q-veto → optional plots.  Heavy artefacts are deleted promptly to
            # keep per-lag memory footprint low.
            for trigger_folder, trigger in zip(trigger_folders, events_data):
                event, cluster_out, event_skymap_statistics = trigger

                # Reconstruct and optionally save/plot the detected waveform.
                reconst_data = reconstruct_waveforms_flow(
                    trigger_folder, config, sub_job_seg.ifos,
                    event, cluster_out, epoch=sub_job_seg.padded_start,
                    wave_file=wave_file, save=config.save_waveform, plot=config.plot_waveform,
                    queue=queue,
                )

                # If this event was matched to an injection, whiten the injected
                # waveform and compute hrss / SNR / frequency / bandwidth / duration.
                if event.injection:
                    injected_data = reconstruct_INJwaveforms_flow(
                        trigger_folder, config, sub_job_seg.ifos, event,
                        HoT_list, mdc_maps, config.iwindow / 2, config.segEdge, config.inRate,
                        wave_file=wave_file, save=config.save_injection, plot=config.plot_injection,
                        queue=queue,
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

                # Q-veto: compute the minimum Q-veto and Q-factor across all IFOs
                # and waveform types (DAT = data, REC = reconstructed).  The minimum
                # across IFOs is the most conservative (worst-case) estimate.
                try:
                    min_qveto   = 1e23
                    min_qfactor = 1e23
                    for ifo in sub_job_seg.ifos:
                        for a_type in ['DAT', 'REC']:
                            [qveto, qfactor] = get_qveto(reconst_data[f'{ifo}_wf_{a_type}_whiten'])
                            min_qveto   = min(min_qveto, qveto)
                            min_qfactor = min(min_qfactor, qfactor)
                    event.Qveto   = [min_qveto, min_qfactor]
                    event.qveto   = min_qveto
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

            # ── 4e. Catalog ──────────────────────────────────────────────────
            # Append accepted events to the run-level catalog file.
            for trigger in events_data:
                event, _, _ = trigger
                trigger_obj = Trigger.from_event(event)
                if queue is not None:
                    queue.put({"type": "trigger", "trigger": trigger_obj})
                elif catalog_file:
                    from pycwb.modules.catalog.catalog import Catalog
                    if not os.path.isabs(catalog_file):
                        catalog_file = os.path.join(working_dir, config.catalog_dir, catalog_file)
                    Catalog.open(catalog_file).add_triggers(trigger_obj)

            # ── 4f. Record lag progress ──────────────────────────────────────
            progress_record = dict(
                job_id=sub_job_seg.index, trial_idx=trial_idx, lag_idx=lag,
                n_triggers=len(events_data), livetime=sub_job_seg.livetime(lag),
                status="completed",
            )
            if queue is not None:
                queue.put({"type": "progress", **progress_record})
            elif catalog_file:
                from pycwb.modules.catalog.catalog import Catalog
                Catalog.open(catalog_file).add_lag_progress(**progress_record)

            logger.info("-------------------------------------------")
            logger.info("Lag %d processing time: %.2f s", lag, time.perf_counter() - lag_timer)
            logger.info("Memory usage: %f.2 MB", psutil.Process().memory_info().rss / 1024 / 1024)
            logger.info("-------------------------------------------")

            # ── Lag memory cleanup ───────────────────────────────────────────
            # Explicitly delete all lag-local objects and call gc.collect() so
            # JAX device buffers and Python heap are freed before the next lag.
            # Without this, 100+ background lags can accumulate GBs of stale memory.
            del frag_clusters_this_lag, fragment_cluster, events_data, trigger_folders
            gc.collect()

        logger.info("Native likelihood loop time: %.2f s", time.perf_counter() - likelihood_timer)

        if len(trial_idxs) > 1:
            trial_walltime = time.perf_counter() - trial_timer
            logger.info("--------------------------------------------")
            logger.info("Trial %d / %d processing time: %.2f s",
                        trial_idx + 1, len(trial_idxs), trial_walltime)
            logger.info("--------------------------------------------")

    job_walltime = time.perf_counter() - job_timer
    speed_factor = job_seg.duration / job_walltime if job_walltime > 0 else float('inf')
    logger.info("============================================")
    logger.info("Job segment %s total time: %.2f s", job_seg.index, job_walltime)
    logger.info("Effective data length:     %.2f s  (padded %.2f s - 2 x segEdge %.2f s)",
                job_seg.duration, job_seg.padded_duration, job_seg.seg_edge)
    logger.info("Speed factor:              %.2fx  (data / walltime)", speed_factor)
    logger.info("============================================")
