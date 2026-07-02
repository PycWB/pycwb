"""
Core analysis pipeline for a single job segment (native Python path).

Pipeline overview (one job segment = one GPS time window)
--------------------------------------------------------

  ┌──────────────────────────────────────────────────────────┐
  │                  process_job_segment()                   │
  │  For each trial_idx (injection realisation):             │
  │                                                          │
  │  1. DATA LOADING    read frames / generate noise        │
  │                     inject simulated signals (MDC)       │
  │                          │                               │
  │  2. CONDITIONING    resample → whiten data + MDC         │
  │                     ──────── (one-time setup) ────────   │
  │  3. SETUP           coherence (WDM TF maps)              │
  │                     TD-input cache (float32 planes)      │
  │                     supercluster (sky patterns + XTalk)  │
  │                     likelihood (sky scan arrays)         │
  │                          │                               │
  │  4. PER-LAG LOOP    ┌──────────────────────────┐        │
  │    for lag 0..N-1:  │ coherence → supercluster │        │
  │                     │    → likelihood (per cluster) │    │
  │                     │    → save & post-process  │        │
  │                     └──────────────────────────┘        │
  │                                                          │
  │  Lags are parallelized via ThreadPoolExecutor when       │
  │  parallel_lag_workers > 1 and no injections present.     │
  │  Each lag produces a LagResult; outputs are saved        │
  │  via LagOutputContext (catalog / waveforms / plots).     │
  └──────────────────────────────────────────────────────────┘

Data flow between pipeline stages
----------------------------------

  raw data (TimeSeries[])
    │
    ├─[check_and_resample_py]──► resampled strains
    ├─[data_conditioning]──────► conditioned strains + nRMS
    │
    ├─[setup_coherence]────────► coherence_setup (WDM TF maps)
    ├─[build_td_inputs_cache]──► td_inputs_cache (float32)
    ├─[setup_supercluster]─────► supercluster_setup + xtalk catalog
    ├─[setup_likelihood]───────► likelihood_setup (sky arrays)
    │
    └─[per lag]────────────────► LagResult (events + metadata)
                                  │
                                  └─[_save_lag_outputs]──► catalog / HDF5 / plots

Key abstractions (why they exist)
---------------------------------
- ``LagAnalysisContext`` : frozen snapshot of everything one lag needs.
  Avoids passing 10+ parameters to every per-lag function.
- ``LagOutputContext``   : I/O handles (dirs, catalog, queue) kept
  separate from physics state so analysis and persistence don't couple.
- ``LagResult``          : value object bundling a lag's products;
  makes the analysis→save handoff explicit and testable.

Reference
---------
See ``docs/3.run_pycwb_with_yaml_config.md`` for configuration details.
"""
import logging
import os
import time
import psutil
import numpy as np
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from copy import copy
from dataclasses import dataclass, replace
from pycwb.config import Config
from pycwb.types.time_series import TimeSeries
from pycwb.modules.super_cluster_native.super_cluster import setup_supercluster, supercluster_single_lag
from pycwb.utils.td_vector_batch import build_td_inputs_cache
from pycwb.modules.xtalk.type import XTalk
from pycwb.modules.coherence_native.coherence import setup_coherence, coherence_single_lag
from pycwb.modules.read_data import generate_strain_from_injection, generate_noise_for_job_seg, read_from_job_segment
from pycwb.modules.read_data.data_check import check_and_resample_py
from pycwb.modules.data_conditioning.data_conditioning import data_conditioning
from pycwb.modules.cwb_interop import create_cwb_workdir
from pycwb.modules.likelihoodWP.likelihood import likelihood, setup_likelihood
from pycwb.types.job import WaveSegment
from pycwb.types.network_event import Event
from pycwb.modules.workflow_utils.job_setup import print_job_info, print_node_info
from pycwb.utils.memory import release_memory
from pycwb.workflow.subflow.job_segment_output import (
    _cleanup_lag_output_state,
    _create_and_save_trigger_folders,
    _log_lag_completion,
    _log_lag_output_timing,
    _postprocess_saved_triggers,
    _record_output_progress,
    _write_trigger_records,
)
from pycwb.workflow.subflow.job_segment_progress import (
    _catalog_path as _catalog_path,
    _lag_metadata,
    _lag_progress_record,
    _record_lag_progress as _record_lag_progress,
)
from pycwb.workflow.subflow.job_segment_resources import (
    _free_jax_buffers as _free_jax_buffers,
    _parallel_inner_threads,
    _temporary_numba_threads,
)
from pycwb.workflow.subflow.job_segment_veto import (
    _effective_veto_windows,
    _injection_aware_veto_windows,
    _lag_livetime,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LagAnalysisContext:
    """Read-only state needed to analyse one trial across its lags."""

    config: Config
    job_seg: WaveSegment
    sub_job_seg: WaveSegment
    trial_idx: int
    n_lag: int
    coherence_setup: object
    supercluster_setup: object
    td_inputs_cache: object
    xtalk: object
    likelihood_setup: object
    nRMS: object
    veto_windows: object
    numba_threads: int | None = None


@dataclass(frozen=True)
class LagOutputContext:
    """State used only while saving lag results."""

    working_dir: str
    config: Config
    sub_job_seg: WaveSegment
    catalog_file: str | None
    wave_file: str | None
    queue: object
    HoT_list: object
    mdc_maps: object


@dataclass(frozen=True)
class LagResult:
    """The physics products and bookkeeping produced by one lag."""

    lag: int
    lag_timer: float
    time_lag: list[float]
    segment_lag: list[float]
    events_data: list[tuple[Event, object, object]]
    progress_record: dict


def _run_lag_analysis(context: LagAnalysisContext, lag: int) -> LagResult:
    config = context.config
    sub_job_seg = context.sub_job_seg
    lag_timer = time.perf_counter()
    time_lag, segment_lag, lag_shifts = _lag_metadata(sub_job_seg, lag)
    lag_shift_str = ", ".join(
        f"{ifo}={shift:.3f}s"
        for ifo, shift in zip(sub_job_seg.ifos, lag_shifts)
    )
    logger.info("Processing lag %d / %d  [%s]", lag, sub_job_seg.n_lag - 1, lag_shift_str)

    seg_thr = getattr(config, 'segTHR', 0.0) or 0.0
    if seg_thr > 0 and context.veto_windows is not None:
        lag_livetime = _lag_livetime(context, lag)
        if lag_livetime < seg_thr:
            logger.warning(
                "Skipping lag %d: post-CAT2 livetime %.2f s < segTHR %.2f s",
                lag, lag_livetime, seg_thr,
            )
            return LagResult(
                lag=lag,
                lag_timer=lag_timer,
                time_lag=time_lag,
                segment_lag=segment_lag,
                events_data=[],
                progress_record=_lag_progress_record(
                    context, lag, n_triggers=0,
                    livetime=0.0, status="skipped_segTHR",
                ),
            )
        logger.info(
            "Processing lag %d: post-CAT2 livetime %.2f s >= segTHR %.2f s, lost %.2f s",
            lag, lag_livetime, seg_thr, sub_job_seg.duration - lag_livetime,
        )

    # Build injection-aware veto windows AFTER the segTHR check so that
    # small injection windows cannot cause the analysis to be skipped.
    effective_veto = _injection_aware_veto_windows(
        config, sub_job_seg, context.veto_windows,
    )

    with _temporary_numba_threads(context.numba_threads):
        timer_coherence = time.perf_counter()
        frag_clusters_this_lag = coherence_single_lag(
            context.coherence_setup, lag,
            veto_windows=effective_veto,
        )
        logger.info("Coherence time for lag %d: %.2f s", lag, time.perf_counter() - timer_coherence)

        timer_supercluster = time.perf_counter()
        fragment_cluster = supercluster_single_lag(
            context.supercluster_setup, config, frag_clusters_this_lag, lag,
            xtalk=context.xtalk, td_inputs_cache=context.td_inputs_cache,
        )
        logger.info("Supercluster time for lag %d: %.2f s", lag, time.perf_counter() - timer_supercluster)

        if fragment_cluster is None:
            logger.warning(
                "No supercluster results for lag %d (job segment %s trial_idx=%s)",
                lag, context.job_seg.index, context.trial_idx,
            )
            return LagResult(
                lag=lag,
                lag_timer=lag_timer,
                time_lag=time_lag,
                segment_lag=segment_lag,
                events_data=[],
                progress_record=_lag_progress_record(
                    context, lag, n_triggers=0,
                    livetime=_lag_livetime(context, lag), status="completed",
                ),
            )

        events_data = []
        for k, selected_cluster in enumerate(fragment_cluster.clusters):
            if selected_cluster.cluster_status > 0:
                continue
            selected_cluster.cluster_id = k + 1

            # NOTE: likelihood() errors are intentionally NOT caught here.
            # A failure indicates a systemic issue (e.g. corrupt sky arrays,
            # OOM, NaN propagation) that requires investigation — silently
            # skipping the cluster would mask the root cause.
            result_cluster, sky_stats = likelihood(
                config.nIFO, selected_cluster, config,
                cluster_id=k + 1, nRMS=context.nRMS,
                setup=context.likelihood_setup, xtalk=context.xtalk,
            )

            if result_cluster is None or result_cluster.cluster_status != -1:
                continue
            logger.info(
                "likelihood accepted cluster %d in lag %d (%d pixels, from %.2f - %.2f s with freq %.2f - %.2f Hz)",
                result_cluster.cluster_id, lag, len(result_cluster.pixel_arrays),
                result_cluster.start_time, result_cluster.stop_time,
                result_cluster.low_frequency, result_cluster.high_frequency,
            )

            event = Event()
            event.output_py(sub_job_seg, result_cluster, config)
            event.job_id = sub_job_seg.index
            event.trial_idx = context.trial_idx
            event.lag_idx = lag
            event.id = event.long_id

            if sub_job_seg.injections:
                for injection in sub_job_seg.injections:
                    if event.start[0] - 0.1 < injection['gps_time'] < event.stop[0] + 0.1:
                        event.injection = injection

            events_data.append((event, result_cluster, sky_stats))

    logger.info("Memory usage: %f.2 MB", psutil.Process().memory_info().rss / 1024 / 1024)
    return LagResult(
        lag=lag,
        lag_timer=lag_timer,
        time_lag=time_lag,
        segment_lag=segment_lag,
        events_data=events_data,
        progress_record=_lag_progress_record(
            context, lag, n_triggers=len(events_data),
            livetime=_lag_livetime(context, lag), status="completed",
        ),
    )


def _save_lag_outputs(output_context: LagOutputContext, result: LagResult) -> None:
    # Phase A: create trigger folders and persist raw cluster/skymap data.
    trigger_folders = _create_and_save_trigger_folders(output_context, result)

    # Phase B: per-event post-processing: waveforms, injections, Q-veto, plots.
    reconstruct_elapsed, qveto_elapsed, plot_elapsed = _postprocess_saved_triggers(
        output_context, result, trigger_folders,
    )

    # Phase C: convert events to Trigger objects and persist them.
    trigger_convert_elapsed, trigger_write_elapsed = _write_trigger_records(
        output_context, result,
    )

    progress_elapsed = _record_output_progress(output_context, result)
    _log_lag_output_timing(
        result,
        reconstruct_elapsed,
        qveto_elapsed,
        plot_elapsed,
        trigger_convert_elapsed,
        trigger_write_elapsed,
        progress_elapsed,
    )
    _log_lag_completion(result)

    del trigger_folders
    _cleanup_lag_output_state()


def _iter_pending_lags(
    context: LagAnalysisContext,
    skip_lags: dict[int, set[int]] | None,
):
    skipped = skip_lags.get(context.trial_idx, set()) if skip_lags else set()
    for lag in range(context.n_lag):
        if lag in skipped:
            logger.info("Skipping lag %d (trial %d) due to skip_lags", lag, context.trial_idx)
            continue
        yield lag


def _process_lags(
    context: LagAnalysisContext,
    output_context: LagOutputContext,
    skip_lags: dict[int, set[int]] | None,
) -> None:
    lag_workers = max(1, int(getattr(context.config, 'parallel_lag_workers', 1) or 1))
    has_injections = bool(context.sub_job_seg.injections)
    use_threaded_lags = lag_workers > 1 and not has_injections and context.n_lag > 1
    pending_lags = _iter_pending_lags(context, skip_lags)

    if use_threaded_lags:
        _process_background_lags_threaded(
            context, output_context, pending_lags, lag_workers,
        )
        return

    for lag in pending_lags:
        result = _run_lag_analysis(context, lag)
        _save_lag_outputs(output_context, result)


def _process_background_lags_threaded(
    context: LagAnalysisContext,
    output_context: LagOutputContext,
    pending_lags,
    lag_workers: int,
) -> None:
    inner_threads = _parallel_inner_threads(context.config, lag_workers)
    worker_context = replace(context, numba_threads=inner_threads)
    max_inflight = max(lag_workers, 2 * lag_workers)
    lag_iter = iter(pending_lags)
    futures = {}

    logger.info(
        "Threaded background lag mode: workers=%d max_inflight=%d numba_threads_per_lag=%d",
        lag_workers, max_inflight, inner_threads,
    )

    def submit_until_full(executor) -> None:
        while len(futures) < max_inflight:
            try:
                lag = next(lag_iter)
            except StopIteration:
                return
            future = executor.submit(_run_lag_analysis, worker_context, lag)
            futures[future] = lag

    with ThreadPoolExecutor(max_workers=lag_workers, thread_name_prefix="pycwb-lag") as executor:
        submit_until_full(executor)
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                lag = futures.pop(future)
                try:
                    result = future.result()
                except Exception:
                    logger.exception("Lag %d failed in threaded background mode", lag)
                    raise
                _save_lag_outputs(output_context, result)
            submit_until_full(executor)


def process_job_segment(working_dir: str, config: Config, job_seg: WaveSegment, compress_json: bool = True,
                        catalog_file: str = None, queue=None, production_mode: bool = False,
                        skip_lags: dict[int, set[int]] | None = None):
    """
    The core workflow to process single job segment with trials or lags.

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
    for trial_number, trial_idx in enumerate(sorted(trial_idxs), start=1):
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
            from pycwb.modules.data_conditioning.whitening_mdc import whitening_mdc
            mdc_maps, HoT_list = zip(*[whitening_mdc(config, m, nrms) for m, nrms in zip(mdc, nRMS)])
            del mdc
            release_memory()

        # ─────────────────────────────────────────────────────────────────────
        # STEP 3 – ONE-TIME LAG-INDEPENDENT SETUP
        # ─────────────────────────────────────────────────────────────────────
        # Everything below is computed once per trial and then reused by every
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
        veto_windows = _effective_veto_windows(config, sub_job_seg)
        analysis_context = LagAnalysisContext(
            config=config,
            job_seg=job_seg,
            sub_job_seg=sub_job_seg,
            trial_idx=trial_idx,
            n_lag=n_lag,
            coherence_setup=coherence_setup,
            supercluster_setup=supercluster_setup,
            td_inputs_cache=td_inputs_cache,
            xtalk=xtalk,
            likelihood_setup=likelihood_setup,
            nRMS=nRMS,
            veto_windows=veto_windows,
        )
        output_context = LagOutputContext(
            working_dir=working_dir,
            config=config,
            sub_job_seg=sub_job_seg,
            catalog_file=catalog_file,
            wave_file=wave_file,
            queue=queue,
            HoT_list=HoT_list,
            mdc_maps=mdc_maps,
        )
        _process_lags(
            analysis_context,
            output_context,
            skip_lags=skip_lags,
        )

        logger.info("Native likelihood loop time: %.2f s", time.perf_counter() - likelihood_timer)

        if len(trial_idxs) > 1:
            trial_walltime = time.perf_counter() - trial_timer
            logger.info("--------------------------------------------")
            logger.info("Trial %d / %d processing time: %.2f s",
                        trial_number, len(trial_idxs), trial_walltime)
            logger.info("--------------------------------------------")

    job_walltime = time.perf_counter() - job_timer
    speed_factor = job_seg.duration / job_walltime if job_walltime > 0 else float('inf')
    logger.info("============================================")
    logger.info("Job segment %s total time: %.2f s", job_seg.index, job_walltime)
    logger.info("Effective data length:     %.2f s  (padded %.2f s - 2 x segEdge %.2f s)",
                job_seg.duration, job_seg.padded_duration, job_seg.seg_edge)
    logger.info("Speed factor:              %.2fx  (data / walltime)", speed_factor)
    logger.info("============================================")
