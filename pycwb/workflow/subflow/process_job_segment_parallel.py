"""Parallel offline job-segment processor.

This is the parallel counterpart of
:func:`~pycwb.workflow.subflow.process_job_segment_native.process_job_segment`.

Key parallelisations compared to the sequential baseline
---------------------------------------------------------
1. **Resampling**   — all IFOs (data *and* MDC) in one thread pool.
2. **Conditioning** — per-IFO in parallel threads.
3. **Setup**        — coherence (per-resolution), TD-input cache (per-level),
                      supercluster + xtalk, and MDC whitening are all overlapped
                      in a single thread pool.
4. **Lag loop**     — all time-slide lags dispatched to a ``ThreadPoolExecutor``;
                      results are collected and then post-processed in lag order.
5. **Q-veto**       — per-trigger Q-veto calls run in parallel.

Design philosophy
-----------------
All parallelism lives **here** in the workflow layer.  No analysis modules are
modified.  ``ThreadPoolExecutor`` is preferred because the heavy inner loops
(Numba ``@prange``, JAX ``jit``/``vmap``, BLAS) release the GIL, giving true
multi-core parallelism without pickle overhead for large shared arrays.

Thread-safety assumption
------------------------
``coherence_single_lag``, ``supercluster_single_lag``, and ``likelihood`` are
assumed to only *read* from their setup objects (``coherence_setup``,
``supercluster_setup``, ``likelihood_setup``, ``xtalk``, ``td_inputs_cache``).
The cluster objects they return are freshly created per-lag, so mutation of
``cluster_id`` / ``cluster_status`` is safe across concurrent lag workers.
"""

import gc
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import copy

import numpy as np
import psutil

from pycwb.config import Config
from pycwb.modules.catalog import Catalog  # noqa: F401 – kept for type parity with native
from pycwb.modules.cwb_coherence.coherence import (
    _setup_coherence_single_res,
    coherence_single_lag,
    setup_coherence,
)
from pycwb.modules.cwb_interop import create_cwb_workdir
from pycwb.modules.data_conditioning.data_conditioning_python import (
    data_conditioning_single,
)
from pycwb.modules.likelihoodWP.likelihood import likelihood, setup_likelihood
from pycwb.modules.qveto.qveto import get_qveto
from pycwb.modules.read_data import (
    generate_noise_for_job_seg,
    generate_strain_from_injection,
    read_from_job_segment,
)
from pycwb.modules.read_data.data_check import check_and_resample_py
from pycwb.modules.reconstruction import estimate_snr
from pycwb.modules.super_cluster.super_cluster import (
    setup_supercluster,
    supercluster_single_lag,
)
from pycwb.modules.workflow_utils import (
    add_event_to_catalog,
    create_single_trigger_folder,
    save_trigger,
)
from pycwb.modules.workflow_utils.job_setup import print_job_info
from pycwb.modules.xtalk.type import XTalk
from pycwb.types.job import WaveSegment
from pycwb.types.network_event import Event
from pycwb.types.time_series import TimeSeries
from pycwb.utils.memory import release_memory
from pycwb.utils.td_vector_batch import (
    _build_td_inputs_single_level,
    build_td_inputs_cache,
)
from pycwb.workflow.subflow.postprocess_and_plots import (
    plot_skymap_flow,
    plot_trigger_flow,
    reconstruct_INJwaveforms_flow,
    reconstruct_waveforms_flow,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def process_job_segment_parallel(
    working_dir: str,
    config: Config,
    job_seg: WaveSegment,
    compress_json: bool = True,
    catalog_file: str = None,
    queue=None,
    production_mode: bool = False,
    skip_lags: list = None,
):
    """Parallel variant of :func:`process_job_segment`.

    Drop-in replacement for
    :func:`~pycwb.workflow.subflow.process_job_segment_native.process_job_segment`
    that overlaps independent work stages with ``ThreadPoolExecutor``.

    Parameters
    ----------
    working_dir : str
        The working directory for the run.
    config : Config
        The configuration object.
    job_seg : WaveSegment
        The job segment to process.
    compress_json : bool
        Whether to compress the JSON output files.
    catalog_file : str
        The catalog file to save the triggers to.
    queue : Queue
        Queue to send triggers to the collector (production mode).
    production_mode : bool
        If ``True``, send triggers to *queue* instead of saving locally.
    skip_lags : list
        Lag indices to skip (used when resuming after a crash).
    """
    # ─────────────────────────────────────────────────────────────────────
    # HIGH-LEVEL WORKFLOW
    # ─────────────────────────────────────────────────────────────────────
    # For each trail_idx (injection realisation, or 0 if no injections):
    #
    #   1. DATA LOADING     – frames / noise / injection setup (sequential)
    #   2. RESAMPLING       – all IFOs + MDC in one parallel pool
    #   3. CONDITIONING     – per-IFO in parallel
    #   4. SETUP            – coherence (per-res) + TD cache (per-level)
    #                         + supercluster+xtalk + MDC whiten, overlapped
    #   5. LIKELIHOOD SETUP – sequential (depends on supercluster output)
    #   6. LAG LOOP         – all lags dispatched to a thread pool;
    #                         each lag: coherence → supercluster → likelihood
    #   7. POST-PROCESS     – waveforms, injections, Q-veto, plots, catalog
    #                         (sequential per-lag, parallel Q-veto per trigger)
    # ─────────────────────────────────────────────────────────────────────
    print_job_info(job_seg)
    job_timer = time.perf_counter()

    if not job_seg.frames and not job_seg.noise and not job_seg.injections:
        raise ValueError("No data to process")

    base_data = None
    if job_seg.frames:
        base_data = read_from_job_segment(config, job_seg)
    if job_seg.noise:
        base_data = generate_noise_for_job_seg(
            job_seg, config.inRate, f_low=config.fLow, data=base_data
        )

    trail_idxs = {0}
    if job_seg.injections:
        trail_idxs = set(
            inj.get("trail_idx", 0) for inj in job_seg.injections
        )

    if catalog_file is not None:
        base = os.path.basename(catalog_file)
        stem, _ = os.path.splitext(base)
        wave_file = stem.replace("catalog", "wave") + ".h5"
    else:
        wave_file = None

    # ── Outer loop: one pass per injection trail ──────────────────────────
    for trail_idx in trail_idxs:

        # ─────────────────────────────────────────────────────────────────
        # STEP 1 – DATA LOADING & INJECTION SETUP (sequential)
        # ─────────────────────────────────────────────────────────────────
        data = base_data
        if len(trail_idxs) == 1:
            base_data = None

        if job_seg.injections:
            sub_job_seg = copy(job_seg)
            sub_job_seg.injections = [
                inj for inj in job_seg.injections
                if inj.get("trail_idx", 0) == trail_idx
            ]
            logger.info(
                "Processing trail_idx: %d with %d injections: %s",
                trail_idx, len(sub_job_seg.injections), sub_job_seg.injections,
            )
            # MDC buffer must span the full padded window [padded_start, padded_end] so that
            # whitening_mdc and get_INJ_waveform see the same time axis as the conditioned strains.
            mdc = [
                TimeSeries(
                    data=np.zeros(int(sub_job_seg.padded_duration * sub_job_seg.sample_rate)),
                    t0=sub_job_seg.padded_start,
                    dt=1.0 / sub_job_seg.sample_rate,
                )
                for _ in range(len(sub_job_seg.ifos))
            ]
            for injection in sub_job_seg.injections:
                inj = generate_strain_from_injection(
                    injection, config, sub_job_seg.sample_rate, sub_job_seg.ifos
                )
                mdc  = [mdc[i].inject(inj[i])  for i in range(len(sub_job_seg.ifos))]
                data = [data[i].inject(inj[i]) for i in range(len(sub_job_seg.ifos))]
        else:
            logger.info("Processing trail_idx: %d without injections", trail_idx)
            sub_job_seg = job_seg
            mdc = None

        sub_job_seg.trail_idx = trail_idx

        # ─────────────────────────────────────────────────────────────────
        # BENCHMARK ONLY – cWB comparison workdir (sequential, before resample)
        # ─────────────────────────────────────────────────────────────────
        if getattr(config, "cwb_compare", False):
            _cwb_compare_dir = getattr(config, "cwb_compare_dir", "") or None
            create_cwb_workdir(
                working_dir, config, sub_job_seg, data,
                cwb_compare_dir=_cwb_compare_dir,
            )

        nIFO = len(sub_job_seg.ifos)

        # ─────────────────────────────────────────────────────────────────
        # STEP 2 – PARALLEL RESAMPLING (data + MDC, all IFOs at once)
        # ─────────────────────────────────────────────────────────────────
        stage_t = time.perf_counter()
        resampled = [None] * nIFO
        mdc_resampled = [None] * nIFO if mdc is not None else None

        n_resample_workers = nIFO * (2 if mdc is not None else 1)
        with ThreadPoolExecutor(max_workers=n_resample_workers) as pool:
            futures = {}
            for i in range(nIFO):
                futures[pool.submit(check_and_resample_py, data[i], config, i)] = ("data", i)
            if mdc is not None:
                for i in range(nIFO):
                    futures[pool.submit(check_and_resample_py, mdc[i], config, i)] = ("mdc", i)
            for fut in as_completed(futures):
                kind, i = futures[fut]
                if kind == "data":
                    resampled[i] = fut.result()
                else:
                    mdc_resampled[i] = fut.result()

        data = resampled
        if mdc is not None:
            mdc = mdc_resampled
        logger.info(
            "Parallel resample time: %.2f s (%d IFOs%s)",
            time.perf_counter() - stage_t, nIFO,
            " + MDC" if mdc is not None else "",
        )
        logger.info("Memory usage: %.2f MB", psutil.Process().memory_info().rss / 1024 / 1024)

        # ─────────────────────────────────────────────────────────────────
        # STEP 3 – PARALLEL DATA CONDITIONING (per-IFO)
        # ─────────────────────────────────────────────────────────────────
        stage_t = time.perf_counter()
        with ThreadPoolExecutor(max_workers=nIFO) as pool:
            cond_futures = {
                pool.submit(data_conditioning_single, config, data[i]): i
                for i in range(nIFO)
            }
            cond_results = [None] * nIFO
            for fut in as_completed(cond_futures):
                cond_results[cond_futures[fut]] = fut.result()

        strains = [r[0] for r in cond_results]
        nRMS    = [r[1] for r in cond_results]
        del data, cond_results
        release_memory()
        logger.info("Parallel conditioning time: %.2f s", time.perf_counter() - stage_t)
        logger.info("Memory usage: %.2f MB", psutil.Process().memory_info().rss / 1024 / 1024)

        # ─────────────────────────────────────────────────────────────────
        # STEP 4 – OVERLAPPED SETUP
        #   • coherence (per-resolution threads)
        #   • TD-input cache (per-level threads)
        #   • supercluster + xtalk
        #   • MDC whitening (per-IFO threads)  — only when injections present
        # All four submit groups run concurrently in a single outer pool.
        # ─────────────────────────────────────────────────────────────────
        stage_t = time.perf_counter()
        gps_time = float(strains[0].start_time)

        n_setup_workers = 3 + (1 if mdc is not None else 0)
        with ThreadPoolExecutor(max_workers=n_setup_workers) as pool:
            f_coherence = pool.submit(
                _parallel_coherence_setup, config, strains, sub_job_seg
            )
            f_td_cache = pool.submit(
                _parallel_td_cache_build, config, strains
            )
            f_super_xt = pool.submit(
                _setup_supercluster_and_xtalk, config, gps_time
            )
            if mdc is not None:
                f_mdc = pool.submit(_whiten_mdc_parallel, config, mdc, nRMS)

            coherence_setup             = f_coherence.result()
            td_inputs_cache             = f_td_cache.result()
            supercluster_setup, xtalk   = f_super_xt.result()
            if mdc is not None:
                mdc_maps, HoT_list = f_mdc.result()
                del mdc
                release_memory()
            else:
                mdc_maps  = None
                HoT_list  = None

        logger.info(
            "Parallel setup time (coherence + TD cache + supercluster%s): %.2f s",
            " + MDC" if HoT_list is not None else "",
            time.perf_counter() - stage_t,
        )
        logger.info("Memory usage: %.2f MB", psutil.Process().memory_info().rss / 1024 / 1024)

        # ─────────────────────────────────────────────────────────────────
        # STEP 5 – LIKELIHOOD SETUP (sequential; depends on supercluster)
        # ─────────────────────────────────────────────────────────────────
        stage_t = time.perf_counter()
        likelihood_setup = setup_likelihood(
            config, strains, config.nIFO,
            ml=supercluster_setup.get("ml_likelihood", supercluster_setup["ml"]),
            FP=supercluster_setup.get("FP_likelihood", supercluster_setup["FP"]),
            FX=supercluster_setup.get("FX_likelihood", supercluster_setup["FX"]),
        )
        logger.info("Likelihood setup time: %.2f s", time.perf_counter() - stage_t)
        logger.info("Memory usage: %.2f MB", psutil.Process().memory_info().rss / 1024 / 1024)

        # ─────────────────────────────────────────────────────────────────
        # STEP 6 – PARALLEL LAG LOOP
        # ─────────────────────────────────────────────────────────────────
        n_lag = sub_job_seg.n_lag
        logger.info(
            "Lag plan: n_lag=%d (job segment duration=%.2f s)",
            n_lag, sub_job_seg.duration,
        )

        # ``max_lag_workers`` caps how many lags run at the same time.
        # Default: up to 8 concurrent lags (each spawns further Numba/JAX threads).
        max_lag_workers = getattr(config, "max_lag_workers", min(n_lag, 8))

        likelihood_timer = time.perf_counter()
        with ThreadPoolExecutor(max_workers=max_lag_workers) as pool:
            lag_futures = {
                pool.submit(
                    _process_single_lag,
                    lag,
                    coherence_setup,
                    supercluster_setup,
                    config,
                    xtalk,
                    td_inputs_cache,
                    likelihood_setup,
                    nRMS,
                    sub_job_seg,
                    skip_lags,
                ): lag
                for lag in range(n_lag)
            }
            lag_results: dict[int, tuple] = {}
            for fut in as_completed(lag_futures):
                lag_idx = lag_futures[fut]
                try:
                    lag_results[lag_idx] = fut.result()
                except Exception as exc:
                    logger.error("Lag %d raised an exception: %s", lag_idx, exc, exc_info=True)
                    lag_results[lag_idx] = (lag_idx, [])

        logger.info(
            "Parallel lag loop time: %.2f s (%d lags, max_workers=%d)",
            time.perf_counter() - likelihood_timer, n_lag, max_lag_workers,
        )

        # ─────────────────────────────────────────────────────────────────
        # STEP 7 – POST-PROCESS & CATALOG (sequential in lag order)
        # ─────────────────────────────────────────────────────────────────
        for lag in range(n_lag):
            if lag not in lag_results:
                continue
            _lag_idx, events_data = lag_results[lag]
            if not events_data:
                continue

            lag_shifts = sub_job_seg.lag_shifts[lag]
            lag_shift_str = ", ".join(
                f"{ifo}={shift:.3f}s"
                for ifo, shift in zip(sub_job_seg.ifos, lag_shifts)
            )
            logger.info(
                "Post-processing lag %d / %d  [%s]  (%d events)",
                lag, n_lag - 1, lag_shift_str, len(events_data),
            )

            # ── 7a. Persist raw cluster data ─────────────────────────────
            trigger_folders = []
            for trigger in events_data:
                trigger_folder = create_single_trigger_folder(
                    working_dir, config.trigger_dir, sub_job_seg, trigger
                )
                trigger_folders.append(trigger_folder)
                save_trigger(
                    trigger_folder=trigger_folder,
                    trigger_data=trigger,
                    save_cluster=config.save_cluster,
                    save_sky_map=config.save_sky_map,
                )
            logger.info("Memory usage: %.2f MB", psutil.Process().memory_info().rss / 1024 / 1024)

            # ── 7b. Per-trigger post-processing ─────────────────────────
            for trigger_folder, trigger in zip(trigger_folders, events_data):
                event, cluster_out, event_skymap_statistics = trigger

                # Waveform reconstruction
                reconst_data = reconstruct_waveforms_flow(
                    trigger_folder, config, sub_job_seg.ifos,
                    event, cluster_out, epoch=sub_job_seg.padded_start,
                    wave_file=wave_file,
                    save=config.save_waveform,
                    plot=config.plot_waveform,
                )

                # Injected waveform comparison (only for matched events)
                if event.injection:
                    injected_data = reconstruct_INJwaveforms_flow(
                        trigger_folder, config, sub_job_seg.ifos, event,
                        HoT_list, mdc_maps,
                        config.iwindow / 2, config.segEdge, config.inRate,
                        wave_file=wave_file,
                        save=config.save_injection,
                        plot=config.plot_injection,
                    )
                    event.hrss      += injected_data["hrss"]
                    event.time      += injected_data["central_time"]
                    event.iSNR       = injected_data["snr"]
                    event.frequency += injected_data["central_freq"]
                    event.bandwidth += injected_data["bandwidth"]
                    event.duration  += injected_data["duration"]

                    inj_waveforms = injected_data["whitened_injected_waveform"]
                    rec_waveforms = [
                        reconst_data[f"{ifo}_wf_REC_whiten"]
                        for ifo in sub_job_seg.ifos
                    ]
                    event.oSNR  = [estimate_snr(rec) for rec in rec_waveforms]
                    event.ioSNR = [
                        estimate_snr(inj, rec)
                        if (inj is not None) and (rec is not None) else None
                        for inj, rec in zip(inj_waveforms, rec_waveforms)
                    ]
                    del injected_data, inj_waveforms, rec_waveforms

                # Q-veto: all (ifo, type) pairs in parallel
                try:
                    qveto_inputs = [
                        reconst_data[f"{ifo}_wf_{a_type}_whiten"]
                        for ifo in sub_job_seg.ifos
                        for a_type in ("DAT", "REC")
                    ]
                    with ThreadPoolExecutor(max_workers=len(qveto_inputs)) as pool:
                        qv_futures = [pool.submit(get_qveto, wf) for wf in qveto_inputs]
                        qveto_results = [f.result() for f in qv_futures]
                    min_qveto   = min(r[0] for r in qveto_results)
                    min_qfactor = min(r[1] for r in qveto_results)
                    event.Qveto   = [min_qveto, min_qfactor]
                    event.qveto   = min_qveto
                    event.qfactor = min_qfactor
                    logger.info(
                        "Qveto for event %s: %s, Qfactor: %s",
                        event.hash_id, event.qveto, event.qfactor,
                    )
                except Exception as e:
                    logger.error("Error calculating Qveto for event %s: %s", event.hash_id, e)

                if config.plot_trigger:
                    plot_trigger_flow(trigger_folder, event, cluster_out)

                if config.plot_sky_map:
                    plot_skymap_flow(trigger_folder, event, event_skymap_statistics)

                del reconst_data

            # ── 7c. Catalog ──────────────────────────────────────────────
            for trigger in events_data:
                catalog_file = add_event_to_catalog(
                    working_dir, config.catalog_dir,
                    trigger_data=trigger,
                    catalog_file=catalog_file,
                )
            logger.info("-------------------------------------------")
            logger.info("Lag %d post-processing done", lag)
            logger.info("Memory usage: %.2f MB", psutil.Process().memory_info().rss / 1024 / 1024)
            logger.info("-------------------------------------------")
            del events_data, trigger_folders

        # Clean up trail-level objects before the next injection trail
        del coherence_setup, td_inputs_cache, supercluster_setup, xtalk
        del likelihood_setup, lag_results
        gc.collect()
        logger.info("Trail %d done.", trail_idx)

    job_walltime = time.perf_counter() - job_timer
    speed_factor = job_seg.duration / job_walltime if job_walltime > 0 else float("inf")
    logger.info("============================================")
    logger.info("Job segment %s total time: %.2f s", job_seg.index, job_walltime)
    logger.info(
        "Effective data length:     %.2f s  (padded %.2f s - 2 x segEdge %.2f s)",
        job_seg.duration, job_seg.padded_duration, job_seg.seg_edge,
    )
    logger.info("Speed factor:              %.2fx  (data / walltime)", speed_factor)
    logger.info("============================================")


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers — parallel wrappers around existing per-item functions
# ─────────────────────────────────────────────────────────────────────────────

def _process_single_lag(
    lag,
    coherence_setup,
    supercluster_setup,
    config,
    xtalk,
    td_inputs_cache,
    likelihood_setup,
    nRMS,
    sub_job_seg,
    skip_lags,
):
    """Run coherence → supercluster → likelihood for one time-slide lag.

    Called concurrently from the lag ``ThreadPoolExecutor``.

    Parameters
    ----------
    lag : int
        Lag index (0 = zero-lag).
    coherence_setup, supercluster_setup, likelihood_setup :
        Pre-built setup objects (read-only).
    config : Config
    xtalk : XTalk
    td_inputs_cache : dict
    nRMS : list
        Per-IFO noise RMS.
    sub_job_seg : WaveSegment
        Current trail's job segment (provides ``lag_shifts``, ``n_lag``, …).
    skip_lags : list or None

    Returns
    -------
    tuple[int, list]
        ``(lag, events_data)`` where *events_data* is a list of
        ``(Event, result_cluster, sky_stats)`` tuples.
    """
    if skip_lags and lag in skip_lags:
        logger.info("Skipping lag %d due to skip_lags", lag)
        return lag, []

    lag_timer = time.perf_counter()
    lag_shifts = sub_job_seg.lag_shifts[lag]
    lag_shift_str = ", ".join(
        f"{ifo}={shift:.3f}s"
        for ifo, shift in zip(sub_job_seg.ifos, lag_shifts)
    )
    logger.info("Processing lag %d / %d  [%s]", lag, sub_job_seg.n_lag - 1, lag_shift_str)

    # ── 4a. Coherence ────────────────────────────────────────────────────
    frag_clusters_this_lag = coherence_single_lag(coherence_setup, lag)

    # ── 4b. Supercluster ─────────────────────────────────────────────────
    fragment_cluster = supercluster_single_lag(
        supercluster_setup, config, frag_clusters_this_lag, lag,
        xtalk=xtalk, td_inputs_cache=td_inputs_cache,
    )

    if fragment_cluster is None:
        logger.warning(
            "No supercluster results for lag %d (job segment %s trail_idx=%s)",
            lag, sub_job_seg.index, sub_job_seg.trail_idx,
        )
        return lag, []

    # ── 4c. Likelihood ───────────────────────────────────────────────────
    events_data = []
    for k, selected_cluster in enumerate(fragment_cluster.clusters):
        if selected_cluster.cluster_status > 0:
            continue

        selected_cluster.cluster_id = k + 1

        result_cluster, sky_stats = likelihood(
            config.nIFO,
            selected_cluster,
            config.MRAcatalog,
            cluster_id=k + 1,
            nRMS=nRMS,
            setup=likelihood_setup,
            xtalk=xtalk,
            config=config,
        )

        if result_cluster is None or result_cluster.cluster_status != -1:
            logger.info("Likelihood rejected cluster %d in lag %d", k + 1, lag)
            continue

        logger.info("Likelihood accepted cluster %d in lag %d", k + 1, lag)

        event = Event()
        event.output_py(sub_job_seg, result_cluster, config)
        event.job_id = sub_job_seg.index

        # Match to the first injection whose GPS falls within the event window.
        # FIXME: overlapping signals share the GPS window; only the first match
        #        is associated here.
        if sub_job_seg.injections:
            for injection in sub_job_seg.injections:
                if event.start[0] - 0.1 < injection["gps_time"] < event.stop[0] + 0.1:
                    event.injection = injection
                    break

        events_data.append((event, result_cluster, sky_stats))

    logger.info(
        "Lag %d: %.2f s, %d events",
        lag, time.perf_counter() - lag_timer, len(events_data),
    )
    return lag, events_data


def _parallel_coherence_setup(config, strains, job_seg):
    """Run per-resolution coherence setup in parallel threads.

    Calls :func:`_setup_coherence_single_res` for every resolution level
    concurrently, mirroring the inline approach used by the online processor.

    Returns
    -------
    list[dict]
        One setup dict per resolution (same structure as
        :func:`setup_coherence`).
    """
    up_n = max(1, int(config.rateANA / 1024))
    normalized = [PyCWBTimeSeries.from_input(s) for s in strains]
    nRES = config.nRES
    with ThreadPoolExecutor(max_workers=nRES) as pool:
        futures = {
            pool.submit(
                _setup_coherence_single_res, i, config, normalized, up_n,
                job_seg=job_seg,
            ): i
            for i in range(nRES)
        }
        setups = [None] * nRES
        for fut in as_completed(futures):
            setups[futures[fut]] = fut.result()
    return setups


def _parallel_td_cache_build(config, strains):
    """Build TD-input cache with per-level parallelism.

    Calls :func:`_build_td_inputs_single_level` for every WDM level in
    parallel, then assembles the dual-key alias dict.

    Returns
    -------
    dict[int, list]
        Same structure as :func:`build_td_inputs_cache`.
    """
    from pycwb.types.time_series import TimeSeries as TS

    strains_ts = [TS.from_input(s) for s in strains]
    upTDF = int(getattr(config, "upTDF", 1))
    levels = list(config.WDM_level)

    with ThreadPoolExecutor(max_workers=len(levels)) as pool:
        futures = {
            pool.submit(
                _build_td_inputs_single_level, level, config, strains_ts, upTDF
            ): level
            for level in levels
        }
        level_results: dict[int, object] = {}
        for fut in as_completed(futures):
            level = futures[fut]
            wdm_layers, per_ifo = fut.result()
            level_results[wdm_layers] = per_ifo

    td_inputs_cache: dict = {}
    for wdm_layers, per_ifo in level_results.items():
        td_inputs_cache[int(wdm_layers)]     = per_ifo
        td_inputs_cache[int(wdm_layers) + 1] = per_ifo

    return td_inputs_cache


def _setup_supercluster_and_xtalk(config, gps_time):
    """Load the cross-talk catalog and compute sky patterns.

    Returned as a tuple so it can be submitted as a single future alongside
    the coherence and TD-cache futures.

    Returns
    -------
    tuple[dict, XTalk]
        ``(supercluster_setup, xtalk)``
    """
    xtalk = XTalk.load(config.MRAcatalog)
    sc_setup = setup_supercluster(config, gps_time)
    return sc_setup, xtalk


def _whiten_mdc_parallel(config, mdc, nRMS):
    """Whiten MDC buffers for all IFOs in parallel threads.

    Parameters
    ----------
    config : Config
    mdc : list
        Per-IFO MDC time series (already resampled to ``config.fSample``).
    nRMS : list
        Per-IFO noise RMS from :func:`data_conditioning_single`.

    Returns
    -------
    tuple[tuple, tuple]
        ``(mdc_maps, HoT_list)`` — same convention as the sequential version
        in the native workflow.
    """
    from pycwb.modules.data_conditioning.whitening_mdc import whitening_mdc_py

    nIFO = len(mdc)
    with ThreadPoolExecutor(max_workers=nIFO) as pool:
        futures = [
            pool.submit(whitening_mdc_py, config, m, nrms)
            for m, nrms in zip(mdc, nRMS)
        ]
        results = [f.result() for f in futures]

    mdc_maps, HoT_list = zip(*results)
    return mdc_maps, HoT_list
