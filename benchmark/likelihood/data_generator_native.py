"""
data_generator_native.py — Native-pipeline workflow processor for the
likelihood benchmark.

Runs the pure-Python (non-ROOT) pycwb pipeline through:
    data loading → conditioning → coherence (lag 0) → supercluster

then immediately tests both CPU (likelihoodWP) and GPU (likelihoodWPGPU)
likelihood implementations on the same cluster for a live correctness
comparison, and saves all inputs to ``test_data_native.pkl`` so the
cluster can be reloaded for repeated timing benchmarks.

Usage
-----
    pycwb run user_parameters_injection_native.yaml

The processor is registered via the YAML key::

    segment_processer: ./data_generator_native.process_job_segment
"""

import copy
import gc
import logging
import os
import pickle
import time

import numpy as np
import psutil

from pycwb.config import Config
from pycwb.modules.cwb_coherence.coherence import setup_coherence, coherence_single_lag
from pycwb.modules.data_conditioning.data_conditioning_python import data_conditioning
from pycwb.modules.likelihoodWP.likelihood import likelihood as likelihood_cpu, setup_likelihood
from pycwb.modules.likelihoodWPGPU.likelihood import likelihood as likelihood_gpu
from pycwb.modules.read_data import (
    generate_noise_for_job_seg,
    read_from_job_segment,
    generate_strain_from_injection,
)
from pycwb.modules.read_data.data_check import check_and_resample_py
from pycwb.modules.super_cluster.super_cluster import setup_supercluster, supercluster_single_lag
from pycwb.modules.xtalk.type import XTalk
from pycwb.types.job import WaveSegment
from pycwb.types.time_series import TimeSeries
from pycwb.utils.memory import release_memory
from pycwb.utils.td_vector_batch import build_td_inputs_cache
from pycwb.modules.workflow_utils.job_setup import print_job_info

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------

def _compare_scalar(name: str, cpu_val, gpu_val, rtol: float = 1e-3, atol: float = 1e-5) -> bool:
    """Log and return whether two scalar values agree within tolerance."""
    cpu_f = float(cpu_val)
    gpu_f = float(gpu_val)
    match = np.isclose(cpu_f, gpu_f, rtol=rtol, atol=atol)
    tag = "OK      " if match else "MISMATCH"
    logger.info("  [%s]  %-28s  CPU=%14.6g   GPU=%14.6g", tag, name, cpu_f, gpu_f)
    return match


def _compare_array(name: str, cpu_arr, gpu_arr, rtol: float = 1e-3, atol: float = 1e-5) -> bool:
    """Log and return whether two arrays agree element-wise within tolerance."""
    a = np.asarray(cpu_arr, dtype=np.float64).ravel()
    b = np.asarray(gpu_arr, dtype=np.float64).ravel()
    if a.shape != b.shape:
        logger.warning("  [MISMATCH]  %-28s  shape mismatch: CPU=%s  GPU=%s",
                       name, a.shape, b.shape)
        return False
    max_diff = float(np.max(np.abs(a - b))) if a.size > 0 else 0.0
    close = np.allclose(a, b, rtol=rtol, atol=atol)
    tag = "OK      " if close else "MISMATCH"
    logger.info("  [%s]  %-28s  max_diff=%.4e  (n=%d)", tag, name, max_diff, a.size)
    return close


def _run_correctness_test(
    target_cluster,
    config: Config,
    likelihood_setup: dict,
    xtalk: XTalk,
    nRMS,
) -> None:
    """Run both CPU and GPU likelihood on the same cluster and compare all key outputs."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("CORRECTNESS TEST: CPU (likelihoodWP) vs GPU (likelihoodWPGPU)")
    logger.info("  cluster: %d pixels  |  n_sky: %d  |  n_ifo: %d",
                len(target_cluster.pixels), likelihood_setup["n_sky"], config.nIFO)
    logger.info("=" * 70)

    # Deep-copy so both runs start from the exact same state
    cluster_cpu = copy.deepcopy(target_cluster)
    cluster_gpu = copy.deepcopy(target_cluster)

    # ---------- CPU ----------
    logger.info("Running CPU likelihood ...")
    t0 = time.perf_counter()
    cpu_cluster, cpu_skymap = likelihood_cpu(
        config.nIFO, cluster_cpu, config,
        cluster_id=1, nRMS=nRMS, setup=likelihood_setup, xtalk=xtalk,
    )
    cpu_time = time.perf_counter() - t0
    logger.info("CPU wall time: %.3f s", cpu_time)

    # ---------- GPU ----------
    logger.info("Running GPU likelihood ...")
    t0 = time.perf_counter()
    gpu_cluster, gpu_skymap = likelihood_gpu(
        config.nIFO, cluster_gpu, config,
        cluster_id=1, nRMS=nRMS, setup=likelihood_setup, xtalk=xtalk,
    )
    gpu_time = time.perf_counter() - t0
    logger.info("GPU wall time: %.3f s", gpu_time)

    # ---------- Acceptance check ----------
    cpu_accepted = cpu_cluster is not None and cpu_cluster.cluster_status == -1
    gpu_accepted = gpu_cluster is not None and gpu_cluster.cluster_status == -1

    logger.info("")
    logger.info("CPU: %s  (%.3f s)", "ACCEPTED" if cpu_accepted else "REJECTED", cpu_time)
    logger.info("GPU: %s  (%.3f s)", "ACCEPTED" if gpu_accepted else "REJECTED", gpu_time)

    if not cpu_accepted and not gpu_accepted:
        logger.warning("Both implementations rejected the cluster — no output comparison possible.")
        return

    if cpu_accepted != gpu_accepted:
        logger.warning("ACCEPTANCE MISMATCH — CPU and GPU disagree on cluster selection.")
        return

    # ---------- SkyMapStatistics comparison ----------
    all_ok = True
    logger.info("")
    logger.info("--- SkyMapStatistics ---")
    all_ok &= _compare_scalar("l_max (best sky idx)",
                               cpu_skymap.l_max, gpu_skymap.l_max, rtol=0, atol=0)
    for arr_name in (
        "nSkyStat", "nLikelihood", "nNullEnergy", "nCorrEnergy",
        "nCorrelation", "nDisbalance", "nNetIndex", "nEllipticity",
        "nPolarisation", "nAntennaPrior", "nAlignment",
    ):
        all_ok &= _compare_array(
            arr_name,
            getattr(cpu_skymap, arr_name),
            getattr(gpu_skymap, arr_name),
        )

    # ---------- Cluster metadata comparison ----------
    logger.info("")
    logger.info("--- Cluster metadata ---")
    meta_cpu = cpu_cluster.cluster_meta
    meta_gpu = gpu_cluster.cluster_meta

    scalar_fields = (
        "net_ecor", "net_rho", "sub_net", "sub_net2",
        "g_net", "a_net", "i_net",
        "norm_cor", "like_sky", "energy_sky",
        "g_noise", "sky_chi2", "ndof", "sky_size",
    )
    for field in scalar_fields:
        cpu_v = getattr(meta_cpu, field, None)
        gpu_v = getattr(meta_gpu, field, None)
        if cpu_v is not None and gpu_v is not None:
            all_ok &= _compare_scalar(field, cpu_v, gpu_v)

    # ---------- Sky coordinates ----------
    logger.info("")
    logger.info("--- Sky coordinates ---")
    all_ok &= _compare_scalar("theta [deg]", meta_cpu.theta, meta_gpu.theta,
                               rtol=1e-2, atol=0.5)
    all_ok &= _compare_scalar("phi [deg]", meta_cpu.phi, meta_gpu.phi,
                               rtol=1e-2, atol=0.5)

    # ---------- Summary ----------
    logger.info("")
    logger.info("=" * 70)
    if all_ok:
        logger.info("RESULT : ALL CORRECTNESS CHECKS PASSED")
    else:
        logger.warning("RESULT : SOME CHECKS FAILED — see MISMATCH lines above")
    logger.info("Speed ratio (CPU / GPU):  %.2fx", cpu_time / gpu_time if gpu_time > 0 else float("inf"))
    logger.info("=" * 70)


# ---------------------------------------------------------------------------
# Workflow processor
# ---------------------------------------------------------------------------

def process_job_segment(
    working_dir: str,
    config: Config,
    job_seg: WaveSegment,
    compress_json: bool = True,
    catalog_file: str = None,
    queue=None,
    production_mode: bool = False,
    skip_lags: list = None,
) -> None:
    """
    Native-pipeline workflow processor: generate benchmark data + correctness test.

    Pipeline stages
    ---------------
    1. Data loading (frames / Gaussian noise / signal injection)
    2. Resampling + pure-Python data conditioning  (regression → whitening)
    3. One-time setup: coherence, TD-input cache, supercluster, likelihood
    4. Coherence + supercluster for lag 0 to obtain a real cluster
    5. Save cluster + all setup objects to ``test_data_native.pkl``
    6. Correctness test: run both CPU and GPU likelihood and compare outputs
    """
    print_job_info(job_seg)

    if not job_seg.frames and not job_seg.noise and not job_seg.injections:
        raise ValueError("No data to process (no frames, noise, or injections specified)")

    # ──────────────────────────────────────────────────────────────────────
    # STEP 1 – DATA LOADING
    # ──────────────────────────────────────────────────────────────────────
    base_data = None
    if job_seg.frames:
        base_data = read_from_job_segment(config, job_seg)
    if job_seg.noise:
        base_data = generate_noise_for_job_seg(
            job_seg, config.inRate, f_low=config.fLow, data=base_data
        )

    data = base_data

    # Inject signals: build per-IFO MDC buffer and add to data
    if job_seg.injections:
        mdc = [
            TimeSeries(
                data=np.zeros(int(job_seg.padded_duration * job_seg.sample_rate)),
                t0=job_seg.padded_start,
                dt=1.0 / job_seg.sample_rate,
            )
            for _ in range(len(job_seg.ifos))
        ]
        for injection in job_seg.injections:
            inj = generate_strain_from_injection(
                injection, config, job_seg.sample_rate, job_seg.ifos
            )
            for i in range(len(job_seg.ifos)):
                mdc[i].inject(inj[i], copy=False)
                data[i].inject(inj[i], copy=False)
            del inj
    else:
        mdc = None

    logger.info("Memory after data loading: %.2f MB",
                psutil.Process().memory_info().rss / 1024 / 1024)

    # ──────────────────────────────────────────────────────────────────────
    # STEP 2 – RESAMPLING + DATA CONDITIONING
    # ──────────────────────────────────────────────────────────────────────
    data = [check_and_resample_py(data[i], config, i) for i in range(len(job_seg.ifos))]
    logger.info("Memory after resampling: %.2f MB",
                psutil.Process().memory_info().rss / 1024 / 1024)

    t0 = time.perf_counter()
    strains, nRMS = data_conditioning(config, data)
    data = None
    release_memory()
    logger.info("Data conditioning: %.2f s  |  memory: %.2f MB",
                time.perf_counter() - t0,
                psutil.Process().memory_info().rss / 1024 / 1024)

    # ──────────────────────────────────────────────────────────────────────
    # STEP 3 – ONE-TIME SETUP (coherence, TD cache, supercluster, likelihood)
    # ──────────────────────────────────────────────────────────────────────
    gps_time = float(strains[0].start_time)

    t0 = time.perf_counter()
    coherence_setup = setup_coherence(config, strains, job_seg=job_seg)
    logger.info("Coherence setup: %.2f s", time.perf_counter() - t0)

    t0 = time.perf_counter()
    td_inputs_cache = build_td_inputs_cache(config, strains)
    logger.info("TD inputs cache build: %.2f s", time.perf_counter() - t0)

    logger.info("Memory after setup: %.2f MB",
                psutil.Process().memory_info().rss / 1024 / 1024)

    t0 = time.perf_counter()
    xtalk = XTalk.load(config.MRAcatalog)
    supercluster_setup = setup_supercluster(config, gps_time)
    logger.info("Supercluster setup: %.2f s", time.perf_counter() - t0)

    t0 = time.perf_counter()
    likelihood_setup = setup_likelihood(
        config, strains, config.nIFO,
        ml=supercluster_setup.get("ml_likelihood", supercluster_setup["ml"]),
        FP=supercluster_setup.get("FP_likelihood", supercluster_setup["FP"]),
        FX=supercluster_setup.get("FX_likelihood", supercluster_setup["FX"]),
    )
    logger.info("Likelihood setup: %.2f s", time.perf_counter() - t0)

    # ──────────────────────────────────────────────────────────────────────
    # STEP 4 – COHERENCE + SUPERCLUSTER FOR LAG 0
    # ──────────────────────────────────────────────────────────────────────
    logger.info("Running coherence for lag 0 ...")
    t0 = time.perf_counter()
    frag_clusters_lag0 = coherence_single_lag(coherence_setup, lag_idx=0)
    logger.info("Coherence lag 0: %.2f s  |  %d resolution fragment clusters",
                time.perf_counter() - t0, len(frag_clusters_lag0))

    logger.info("Running supercluster for lag 0 ...")
    t0 = time.perf_counter()
    fragment_cluster = supercluster_single_lag(
        supercluster_setup, config, frag_clusters_lag0, lag_idx=0,
        xtalk=xtalk, td_inputs_cache=td_inputs_cache,
    )
    logger.info("Supercluster lag 0: %.2f s", time.perf_counter() - t0)

    if fragment_cluster is None or len(fragment_cluster.clusters) == 0:
        logger.error("No clusters survived supercluster — cannot generate test data.")
        return

    # Select the cluster with the most pixels as the benchmark target
    active_clusters = [c for c in fragment_cluster.clusters if c.cluster_status <= 0]
    if not active_clusters:
        logger.error("All clusters were rejected by supercluster — no active clusters.")
        return

    target_cluster = max(active_clusters, key=lambda c: len(c.pixels))
    n_pix = len(target_cluster.pixels)
    logger.info("Selected cluster: %d pixels  (from %d active clusters)",
                n_pix, len(active_clusters))

    # ──────────────────────────────────────────────────────────────────────
    # STEP 5 – SAVE TEST DATA
    # ──────────────────────────────────────────────────────────────────────
    # nRMS comes from data_conditioning as a tuple; convert to list for clean pickling
    nRMS_list = list(nRMS)

    # Extract injection distance for labelled output (first injection, if present)
    _distance = None
    if job_seg.injections:
        try:
            _distance = int(round(job_seg.injections[0].parameters.get("distance", 0)))
        except Exception:
            pass

    # Build a label: "d{distance}Mpc_p{n_pix}pix" if distance available, else just "p{n_pix}pix"
    if _distance is not None:
        _label = f"d{_distance}Mpc_p{n_pix}pix"
    else:
        _label = f"p{n_pix}pix"

    test_data = {
        # Primary benchmark inputs (all that benchmark_cpu_vs_gpu.py needs)
        "cluster":            target_cluster,
        "n_ifo":              config.nIFO,
        "n_sky":              likelihood_setup["n_sky"],
        "config":             config,
        "likelihood_setup":   likelihood_setup,
        "xtalk":              xtalk,
        "nRMS":               nRMS_list,
        # Legacy-compatible keys for performance_test_opt_sky.py
        "FP":                 likelihood_setup["FP"],
        "FX":                 likelihood_setup["FX"],
        "ml":                 likelihood_setup["ml"],
        "pixels":             target_cluster.pixels,
        "delta_regulator":    likelihood_setup["delta_regulator"],
        "gamma_regulator":    likelihood_setup["gamma_regulator"],
        "netEC_threshold":    likelihood_setup["netEC_threshold"],
        "network_energy_threshold": likelihood_setup["network_energy_threshold"],
        "netCC":              likelihood_setup["netCC"],
        # Metadata for benchmark reporting
        "n_pix":              n_pix,
        "distance_Mpc":       _distance,
        "label":              _label,
    }

    # Resolve working_dir to an absolute path (conda run may change CWD)

    os.makedirs(working_dir, exist_ok=True)

    # Save with label so multiple runs don't overwrite each other
    output_path = os.path.join(working_dir, f"test_data_native_{_label}.pkl")
    with open(output_path, "wb") as fh:
        pickle.dump(test_data, fh)
    logger.info("Saved test data → %s  (%d keys)", output_path, len(test_data))

    # Also write a copy directly in the benchmark directory for convenience
    local_path = os.path.join(working_dir, f"test_data_native_{_label}.pkl")
    with open(local_path, "wb") as fh:
        pickle.dump(test_data, fh)
    logger.info("Saved test data → %s (local copy)", local_path)

    # ──────────────────────────────────────────────────────────────────────
    # STEP 6 – CORRECTNESS TEST: CPU vs GPU
    # ──────────────────────────────────────────────────────────────────────
    _run_correctness_test(
        target_cluster=target_cluster,
        config=config,
        likelihood_setup=likelihood_setup,
        xtalk=xtalk,
        nRMS=nRMS_list,
    )

    gc.collect()
