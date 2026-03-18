"""
Parallel online segment processor.

This module provides :func:`process_online_segment`, the online counterpart
of :func:`~pycwb.workflow.subflow.process_job_segment_native.process_job_segment`.

Design philosophy
-----------------
All parallelism lives **here** in the workflow layer.  Individual analysis
modules (data-conditioning, coherence, supercluster, likelihood, reconstruction,
Q-veto) are called through their existing per-item functions — none of them are
modified.  This keeps the modules simple and sequentially readable while letting
the orchestrator overlap independent work via ``ThreadPoolExecutor``.

ThreadPoolExecutor is preferred over ProcessPoolExecutor because the heavy
inner loops (Numba ``@prange``, JAX ``jit``/``vmap``, BLAS) release the GIL,
giving true parallelism without pickle overhead for large arrays.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import psutil

from pycwb.config import Config
from pycwb.types.job import WaveSegment
from pycwb.types.online import OnlineSegment, OnlineTrigger
from pycwb.types.network_event import Event

# ── Per-item functions imported from existing modules ────────────────────
from pycwb.modules.read_data.data_check import check_and_resample_py
from pycwb.modules.data_conditioning.data_conditioning_python import (
    data_conditioning_single,
)
from pycwb.modules.cwb_coherence.coherence import (
    _setup_coherence_single_res,
    coherence_single_lag,
)
from pycwb.utils.td_vector_batch import (
    build_td_inputs_cache,
    _build_td_inputs_single_level,
)
from pycwb.modules.super_cluster.super_cluster import (
    setup_supercluster,
    supercluster_single_lag,
)
from pycwb.modules.xtalk.type import XTalk
from pycwb.modules.likelihoodWP.likelihood import likelihood, setup_likelihood
from pycwb.modules.reconstruction import get_network_MRA_wave
from pycwb.modules.qveto.qveto import get_qveto
from pycwb.utils.memory import release_memory
from pycwb.types.time_series import TimeSeries as PyCWBTimeSeries

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────

def process_online_segment(config: Config, online_seg: OnlineSegment):
    """Process a single online segment with intra-segment parallelism.

    This is the online equivalent of ``process_job_segment``.  It is
    designed to run inside a ``ProcessPoolExecutor`` worker spawned by
    :class:`~pycwb.workflow.online.OnlineSearchManager`.

    Parameters
    ----------
    config : Config
        Configuration object (with online extension parameters available
        via ``getattr``).
    online_seg : OnlineSegment
        Pre-read segment data with GPS metadata.

    Returns
    -------
    list[OnlineTrigger]
        Triggers found in this segment (may be empty).
    """
    seg_timer = time.perf_counter()
    nIFO = len(online_seg.ifos)

    # Cap intra-segment thread parallelism to nIFO so that with
    # N workers the total core usage is bounded to N × nIFO.
    max_threads = nIFO

    # data_payload is {channel_name: TimeSeries}; convert to ordered list
    # matching the IFO order in online_seg.ifos / config.ifo.
    payload = online_seg.data_payload
    if isinstance(payload, dict):
        channels = list(payload.keys())
        # match each IFO to its channel (first channel whose prefix == ifo)
        data_list = []
        for ifo in online_seg.ifos:
            matched = next(
                (payload[ch] for ch in channels if ch.startswith(ifo + ":")),
                None,
            )
            if matched is None:
                raise ValueError(
                    f"No channel for IFO {ifo!r} in data_payload keys "
                    f"{list(payload.keys())}"
                )
            data_list.append(matched)
        data = data_list
    else:
        # already a list
        data = list(payload)

    # Normalise each element to PyCWBTimeSeries so downstream modules
    # (which were designed for pycbc/pycwb TimeSeries) see the right types.
    data = [PyCWBTimeSeries.from_input(d) for d in data]

    # Build a minimal WaveSegment so existing modules see what they expect.
    wave_seg = _online_seg_to_wave_seg(online_seg, config)

    # ─────────────────────────────────────────────────────────────────
    # STEP 1 — Parallel resample (per-IFO)
    # ─────────────────────────────────────────────────────────────────
    stage_t = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max_threads) as pool:
        futures = {
            pool.submit(check_and_resample_py, data[i], config, i): i
            for i in range(nIFO)
        }
        resampled = [None] * nIFO
        for fut in as_completed(futures):
            resampled[futures[fut]] = fut.result()
    data = resampled
    logger.info("Parallel resample time: %.2f s (%d IFOs)",
                time.perf_counter() - stage_t, nIFO)

    # ─────────────────────────────────────────────────────────────────
    # STEP 2 — Parallel data conditioning (per-IFO)
    # ─────────────────────────────────────────────────────────────────
    stage_t = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max_threads) as pool:
        futures = {
            pool.submit(data_conditioning_single, config, data[i]): i
            for i in range(nIFO)
        }
        results = [None] * nIFO
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()

    strains = [r[0] for r in results]
    nRMS = [r[1] for r in results]
    del data, results
    release_memory()
    logger.info("Parallel conditioning time: %.2f s", time.perf_counter() - stage_t)
    logger.info("Memory usage: %.2f MB", psutil.Process().memory_info().rss / 1024 / 1024)

    # ─────────────────────────────────────────────────────────────────
    # STEP 3 — Overlap three independent setup stages
    # ─────────────────────────────────────────────────────────────────
    stage_t = time.perf_counter()
    gps_time = float(strains[0].start_time)

    with ThreadPoolExecutor(max_workers=max_threads) as pool:
        f_coherence = pool.submit(
            _parallel_coherence_setup, config, strains, wave_seg,
        )
        f_td_cache = pool.submit(
            _parallel_td_cache_build, config, strains,
        )
        f_super_xt = pool.submit(
            _setup_supercluster_and_xtalk, config, gps_time,
        )

        coherence_setup = f_coherence.result()
        td_inputs_cache = f_td_cache.result()
        supercluster_setup, xtalk = f_super_xt.result()

    logger.info("Parallel setup time (3 stages overlapped): %.2f s",
                time.perf_counter() - stage_t)
    logger.info("Memory usage: %.2f MB", psutil.Process().memory_info().rss / 1024 / 1024)

    # 3d. Likelihood setup — depends on supercluster output (fast, sequential)
    stage_t = time.perf_counter()
    likelihood_setup = setup_likelihood(
        config, strains, config.nIFO,
        ml=supercluster_setup.get("ml_likelihood", supercluster_setup["ml"]),
        FP=supercluster_setup.get("FP_likelihood", supercluster_setup["FP"]),
        FX=supercluster_setup.get("FX_likelihood", supercluster_setup["FX"]),
    )
    logger.info("Likelihood setup time: %.2f s", time.perf_counter() - stage_t)

    # ─────────────────────────────────────────────────────────────────
    # STEP 4 — Single lag (lag=0, online zero-lag only)
    # ─────────────────────────────────────────────────────────────────
    lag = 0
    lag_t = time.perf_counter()

    # 4a. Coherence — pixel selection + fragment clustering
    frag_clusters = coherence_single_lag(coherence_setup, lag)

    # 4b. Supercluster — TD amplitudes + subnet veto
    fragment_cluster = supercluster_single_lag(
        supercluster_setup, config, frag_clusters, lag,
        xtalk=xtalk, td_inputs_cache=td_inputs_cache,
    )

    triggers = []

    if fragment_cluster is None:
        logger.warning("No supercluster results for online segment %d", online_seg.index)
    else:
        # 4c. Likelihood — per-cluster (sequential; inner sky scan is Numba @prange)
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
                logger.info("Likelihood rejected cluster %d", k + 1)
                continue

            logger.info("Likelihood accepted cluster %d", k + 1)
            event = Event()
            event.output_py(wave_seg, result_cluster, config)
            event.job_id = wave_seg.index
            events_data.append((event, result_cluster, sky_stats))

        # 4d. Post-process — parallel reconstruction + Q-veto
        for event, cluster_out, sky_stats in events_data:
            _parallel_postprocess(config, online_seg.ifos, event, cluster_out)
            triggers.append(OnlineTrigger(
                event=event,
                cluster=cluster_out,
                sky_stats=sky_stats,
                segment_index=online_seg.index,
                segment_gps=online_seg.segment_gps_start,
                wall_time_done=time.time(),
            ))

    logger.info("Lag 0 time: %.2f s", time.perf_counter() - lag_t)

    # Capture values needed for the final log before releasing online_seg.
    _seg_index = online_seg.index
    _seg_duration = online_seg.segment_gps_end - online_seg.segment_gps_start

    # Cleanup — free large intermediate objects and return heap to OS
    del coherence_setup, td_inputs_cache, supercluster_setup, xtalk
    del likelihood_setup, frag_clusters, fragment_cluster
    del strains, nRMS, online_seg
    release_memory()

    seg_walltime = time.perf_counter() - seg_timer
    speed_factor = _seg_duration / seg_walltime if seg_walltime > 0 else float("inf")
    logger.info("Online segment %d: %.2f s wall-time, %.2fx speed factor, %d triggers",
                _seg_index, seg_walltime, speed_factor, len(triggers))
    return triggers


# ─────────────────────────────────────────────────────────────────────────
# Internal helpers — parallel wrappers around existing per-item functions
# ─────────────────────────────────────────────────────────────────────────

def _online_seg_to_wave_seg(online_seg: OnlineSegment, config) -> WaveSegment:
    """Convert an OnlineSegment to a WaveSegment for module compatibility."""
    return WaveSegment(
        index=online_seg.index,
        ifos=online_seg.ifos,
        analyze_start=online_seg.segment_gps_start,
        analyze_end=online_seg.segment_gps_end,
        sample_rate=online_seg.sample_rate,
        seg_edge=online_seg.seg_edge,
        lag_size=1,
        lag_step=getattr(config, "lagStep", 1.0),
        lag_off=0,
        lag_max=0,
    )


def _parallel_coherence_setup(config, strains, wave_seg):
    """Run per-resolution coherence setup in parallel threads."""
    up_n = max(1, int(config.rateANA / 1024))
    normalized = [PyCWBTimeSeries.from_input(s) for s in strains]

    nRES = config.nRES
    max_threads = config.nIFO
    with ThreadPoolExecutor(max_workers=max_threads) as pool:
        futures = {
            pool.submit(
                _setup_coherence_single_res, i, config, normalized, up_n,
                job_seg=wave_seg,
            ): i
            for i in range(nRES)
        }
        setups = [None] * nRES
        for fut in as_completed(futures):
            setups[futures[fut]] = fut.result()
    return setups


def _parallel_td_cache_build(config, strains):
    """Build TD inputs cache with per-level parallelism."""
    return _build_td_inputs_single_level_parallel(config, strains)


def _build_td_inputs_single_level_parallel(config, strains):
    """Call _build_td_inputs_single_level per level in parallel, then
    assemble the cache dict with aliasing and deduplication."""
    from pycwb.types.time_series import TimeSeries as TS

    strains_ts = [TS.from_input(s) for s in strains]
    upTDF = int(getattr(config, "upTDF", 1))

    # Dispatch per-level work in parallel
    levels = list(config.WDM_level)
    max_threads = config.nIFO
    with ThreadPoolExecutor(max_workers=max_threads) as pool:
        futures = {
            pool.submit(
                _build_td_inputs_single_level,
                level, config, strains_ts, upTDF,
            ): level
            for level in levels
        }
        level_results = {}
        for fut in as_completed(futures):
            level = futures[fut]
            wdm_layers, per_ifo = fut.result()
            level_results[wdm_layers] = per_ifo

    # Build the dual-key cache (wdm_layers and wdm_layers+1 alias the same data)
    td_inputs_cache = {}
    for wdm_layers, per_ifo in level_results.items():
        td_inputs_cache[int(wdm_layers)] = per_ifo
        td_inputs_cache[int(wdm_layers) + 1] = per_ifo

    return td_inputs_cache


def _setup_supercluster_and_xtalk(config, gps_time):
    """Load cross-talk catalog and compute sky patterns.

    Returned as a tuple ``(supercluster_setup, xtalk)`` so it can be
    submitted as a single future alongside coherence and TD cache.
    """
    xtalk = XTalk.load(config.MRAcatalog)
    sc_setup = setup_supercluster(config, gps_time)
    return sc_setup, xtalk


def _parallel_postprocess(config, ifos, event, cluster_out):
    """Run the 4 reconstruction modes + Q-veto in parallel."""
    max_threads = config.nIFO
    # 4 independent get_network_MRA_wave calls
    recon_args = [
        ("signal", 0, True, False),   # reconstructed signal
        ("signal", 0, True, True),    # whitened reconstructed signal
        ("strain", 0, True, False),   # reconstructed data
        ("strain", 0, True, True),    # whitened reconstructed data
    ]

    with ThreadPoolExecutor(max_workers=max_threads) as pool:
        futures = {
            pool.submit(
                get_network_MRA_wave,
                config, cluster_out, config.rateANA, config.nIFO,
                config.TDRate, a_type, mode, tof, whiten=whiten,
            ): idx
            for idx, (a_type, mode, tof, whiten) in enumerate(recon_args)
        }
        recon_results = [None] * 4
        for fut in as_completed(futures):
            recon_results[futures[fut]] = fut.result()

    rec_signal, rec_signal_w, rec_data, rec_data_w = recon_results

    # Q-veto: collect all (ifo, type) pairs and run in parallel
    qveto_inputs = []
    for i, ifo in enumerate(ifos):
        qveto_inputs.append(("DAT", i, rec_data_w[i]))
        qveto_inputs.append(("REC", i, rec_signal_w[i]))

    with ThreadPoolExecutor(max_workers=max_threads) as pool:
        qveto_futures = [
            pool.submit(get_qveto, wf) for _, _, wf in qveto_inputs
        ]
        qveto_results = [f.result() for f in qveto_futures]

    min_qveto = min(r[0] for r in qveto_results) if qveto_results else 1e23
    min_qfactor = min(r[1] for r in qveto_results) if qveto_results else 1e23
    event.Qveto = [min_qveto, min_qfactor]
    event.qveto = min_qveto
    event.qfactor = min_qfactor
    logger.info("Qveto for event %s: %s, Qfactor: %s",
                event.hash_id, event.qveto, event.qfactor)
