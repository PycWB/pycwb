#!/usr/bin/env python
"""
Benchmark: CPU (likelihoodWP) vs GPU (likelihoodWPGPU) likelihood.

Loads pre-generated cluster data from test-data pickle files (produced by
``data_generator_native.py`` via ``pycwb run``), then runs both the CPU and
GPU likelihood on the same cluster and compares faithfulness (numerical
agreement) and wall-clock performance.

Supports multiple data files in a single run for a multi-scale report.

Usage
-----
    # Single file (backward compat):
    python benchmark_cpu_vs_gpu.py --data test_data_native.pkl

    # Multiple files (multi-scale report):
    python benchmark_cpu_vs_gpu.py \
        --data test_data_native_d1030Mpc_p462pix.pkl \
               test_data_native_d130Mpc_p1736pix.pkl \
               test_data_native_d50Mpc_p5100pix.pkl \
        --n-repeats 3
"""

from __future__ import annotations

import argparse
import copy
import logging
import os
import pickle
import sys
import time
from math import sqrt

import numpy as np
import jax

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _timer(fn, label: str, n_repeats: int = 1):
    """Run *fn* n_repeats times, return (first_result, times_list)."""
    times = []
    result = None
    for i in range(n_repeats):
        t0 = time.perf_counter()
        result = fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
        logger.info("  %s  run %d/%d: %.4f s", label, i + 1, n_repeats, times[-1])
    return result, times


def _compare_scalar(name: str, cpu_val, gpu_val, rtol=1e-4, atol=1e-6):
    """Compare two scalars and return True if they match."""
    cpu_f = float(cpu_val)
    gpu_f = float(gpu_val)
    match = np.isclose(cpu_f, gpu_f, rtol=rtol, atol=atol)
    tag = "OK" if match else "MISMATCH"
    logger.info("  %-25s  CPU=%12.6g  GPU=%12.6g  [%s]", name, cpu_f, gpu_f, tag)
    return match


def _compare_array(name: str, cpu_arr, gpu_arr, rtol=1e-4, atol=1e-6):
    """Compare two arrays element-wise and return True if they match."""
    a = np.asarray(cpu_arr, dtype=np.float64).ravel()
    b = np.asarray(gpu_arr, dtype=np.float64).ravel()
    if a.shape != b.shape:
        logger.warning("  %-25s  shape mismatch: CPU=%s  GPU=%s", name, a.shape, b.shape)
        return False
    close = np.allclose(a, b, rtol=rtol, atol=atol)
    max_diff = float(np.max(np.abs(a - b))) if a.size > 0 else 0.0
    tag = "OK" if close else "MISMATCH"
    logger.info("  %-25s  max_diff=%.6e  [%s]", name, max_diff, tag)
    return close


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def _benchmark_one(data_path, n_repeats, skip_warmup, likelihood_cpu, likelihood_gpu,
                   warmup_done):
    """Benchmark a single test-data file. Returns a result dict."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("DATA: %s", data_path)
    with open(data_path, "rb") as f:
        test_data = pickle.load(f)

    config           = test_data["config"]
    likelihood_setup = test_data["likelihood_setup"]
    xtalk            = test_data["xtalk"]
    cluster_orig     = test_data["cluster"]
    n_ifo            = test_data["n_ifo"]
    nRMS             = test_data.get("nRMS") or test_data.get("nRMS_list")
    n_pix            = len(cluster_orig.pixels)
    n_sky            = test_data["n_sky"]
    label            = test_data.get("label") or f"p{n_pix}pix"
    distance_Mpc     = test_data.get("distance_Mpc")

    logger.info("Cluster: n_pix=%d, n_ifo=%d, n_sky=%d, label=%s",
                n_pix, n_ifo, n_sky, label)

    # Warm-up once across all files (JAX traces for first shapes it sees)
    if not skip_warmup and not warmup_done[0]:
        logger.info("=== Warm-up run (Numba + JAX compilation) ===")
        wc = copy.deepcopy(cluster_orig)
        likelihood_cpu(n_ifo, wc, config, cluster_id=0, nRMS=nRMS,
                       setup=likelihood_setup, xtalk=xtalk)
        wc = copy.deepcopy(cluster_orig)
        likelihood_gpu(n_ifo, wc, config, cluster_id=0, nRMS=nRMS,
                       setup=likelihood_setup, xtalk=xtalk)
        logger.info("Warm-up complete.\n")
        warmup_done[0] = True

    # CPU timed runs — force JAX to CPU device to prevent any GPU usage
    logger.info("=== CPU (likelihoodWP) — %d repeats ===", n_repeats)
    cpu_times = []
    cpu_result = None
    cpu_stage_timings = None
    _cpu_device = jax.devices("cpu")[0]
    with jax.default_device(_cpu_device):
        for i in range(n_repeats):
            c = copy.deepcopy(cluster_orig)
            t0 = time.perf_counter()
            cpu_result = likelihood_cpu(n_ifo, c, config, cluster_id=1, nRMS=nRMS,
                                        setup=likelihood_setup, xtalk=xtalk)
            cpu_times.append(time.perf_counter() - t0)
            logger.info("  CPU  run %d/%d: %.4f s", i + 1, n_repeats, cpu_times[-1])
            # Collect per-stage timings from each CPU run (last one is used for report)
            cpu_cluster_run, cpu_skymap_run = cpu_result
            if cpu_skymap_run is not None and hasattr(cpu_skymap_run, "stage_timings") and cpu_skymap_run.stage_timings:
                cpu_stage_timings = cpu_skymap_run.stage_timings

    # GPU timed runs — collect per-stage timings from last run
    logger.info("=== GPU (likelihoodWPGPU) — %d repeats ===", n_repeats)
    gpu_times = []
    gpu_result = None
    gpu_stage_timings = None
    for i in range(n_repeats):
        c = copy.deepcopy(cluster_orig)
        t0 = time.perf_counter()
        gpu_result = likelihood_gpu(n_ifo, c, config, cluster_id=1, nRMS=nRMS,
                                    setup=likelihood_setup, xtalk=xtalk)
        gpu_times.append(time.perf_counter() - t0)
        logger.info("  GPU  run %d/%d: %.4f s", i + 1, n_repeats, gpu_times[-1])
        # Collect per-stage timings from each GPU run (last one is used for report)
        gpu_cluster, gpu_skymap = gpu_result
        if gpu_skymap is not None and hasattr(gpu_skymap, "stage_timings") and gpu_skymap.stage_timings:
            gpu_stage_timings = gpu_skymap.stage_timings

    cpu_cluster, cpu_skymap = cpu_result
    gpu_cluster, gpu_skymap = gpu_result

    cpu_med = float(np.median(cpu_times))
    gpu_med = float(np.median(gpu_times))
    speedup = cpu_med / gpu_med if gpu_med > 0 else float("inf")

    return {
        "label": label,
        "distance_Mpc": distance_Mpc,
        "n_pix": n_pix,
        "n_sky": n_sky,
        "cpu_med": cpu_med,
        "cpu_min": min(cpu_times),
        "cpu_max": max(cpu_times),
        "gpu_med": gpu_med,
        "gpu_min": min(gpu_times),
        "gpu_max": max(gpu_times),
        "speedup": speedup,
        "cpu_stage_timings": cpu_stage_timings,
        "gpu_stage_timings": gpu_stage_timings,
        "cpu_cluster": cpu_cluster,
        "cpu_skymap": cpu_skymap,
        "gpu_cluster": gpu_cluster,
        "gpu_skymap": gpu_skymap,
    }


def _print_report(results, n_repeats):
    """Print the consolidated multi-scale benchmark report."""
    sep = "=" * 90

    print()
    print(sep)
    print("BENCHMARK REPORT — CPU vs GPU likelihood  (median of %d runs)" % n_repeats)
    print(sep)

    # --- Performance table ---
    hdr = ("%-22s  %7s  %7s  %10s  %10s  %8s" %
           ("Label", "n_pix", "n_sky", "CPU (s)", "GPU (s)", "Speedup"))
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print("%-22s  %7d  %7d  %10.4f  %10.4f  %7.2fx" % (
            r["label"], r["n_pix"], r["n_sky"],
            r["cpu_med"], r["gpu_med"], r["speedup"],
        ))

    def _print_stage_table(label, key):
        all_stages = []
        for r in results:
            st = r.get(key) or {}
            for s in st:
                if s != "total" and s not in all_stages:
                    all_stages.append(s)
        if not all_stages:
            print("  (no stage timings available)")
            return
        stage_hdr = "%-30s" % "Stage"
        for r in results:
            stage_hdr += "  %20s" % r["label"]
        print(stage_hdr)
        print("-" * 90)
        for stage in all_stages:
            row = "%-30s" % stage
            for r in results:
                st = r.get(key) or {}
                val = st.get(stage)
                total = (st.get("total") or 1.0)
                if val is not None:
                    row += "  %8.4fs (%5.1f%%)" % (val, 100.0 * val / total)
                else:
                    row += "  %20s" % "—"
            print(row)
        row = "%-30s" % "TOTAL"
        for r in results:
            st = r.get(key) or {}
            val = st.get("total")
            if val is not None:
                row += "  %8.4fs (%5.1f%%)" % (val, 100.0)
            else:
                row += "  %20s" % "—"
        print(row)

    # --- Per-stage CPU timing table ---
    print()
    print("CPU STAGE BREAKDOWN (last accepted run per dataset)")
    print("-" * 90)
    _print_stage_table("CPU", "cpu_stage_timings")

    # --- Per-stage GPU timing table ---
    print()
    print("GPU STAGE BREAKDOWN (last accepted run per dataset)")
    print("-" * 90)
    _print_stage_table("GPU", "gpu_stage_timings")

    # --- Faithfulness summary ---
    print()
    print("FAITHFULNESS SUMMARY")
    print("-" * 50)
    all_passed = True
    for r in results:
        cpu_c = r["cpu_cluster"]
        gpu_c = r["gpu_cluster"]
        if cpu_c is None and gpu_c is None:
            status = "both rejected"
        elif cpu_c is None or gpu_c is None:
            status = "DETECTION MISMATCH"
            all_passed = False
        else:
            cm = cpu_c.cluster_meta
            gm = gpu_c.cluster_meta
            theta_ok = np.isclose(cm.theta, gm.theta, atol=0.1)
            phi_ok = np.isclose(cm.phi, gm.phi, atol=0.1)
            rho_ok = np.isclose(float(cm.net_rho), float(gm.net_rho), rtol=1e-3)
            if theta_ok and phi_ok and rho_ok:
                status = "OK (theta/phi/rho match)"
            else:
                status = "MISMATCH (theta=%s phi=%s rho=%s)" % (
                    "OK" if theta_ok else "FAIL",
                    "OK" if phi_ok else "FAIL",
                    "OK" if rho_ok else "FAIL",
                )
                all_passed = False
        print("  %-22s  %s" % (r["label"], status))

    print()
    print("OVERALL:", "ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED")
    print(sep)
    return all_passed


def main():
    parser = argparse.ArgumentParser(description="CPU vs GPU likelihood benchmark")
    parser.add_argument("--n-repeats", type=int, default=3,
                        help="Number of timed repeats per dataset (default: 3)")
    parser.add_argument("--skip-warmup", action="store_true",
                        help="Skip JAX/Numba warm-up run")
    parser.add_argument("--data", type=str, nargs="+", default=["test_data.pkl"],
                        help="Path(s) to pre-generated test data pickle(s). "
                             "Multiple files produce a multi-scale report.")
    args = parser.parse_args()

    from pycwb.modules.likelihoodWP.likelihood import likelihood as likelihood_cpu
    from pycwb.modules.likelihoodWPGPU.likelihood import likelihood as likelihood_gpu

    # Resolve data paths: if relative, interpret them relative to this script's directory
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    resolved_paths = []
    for p in args.data:
        if not os.path.isabs(p) and not os.path.exists(p):
            candidate = os.path.join(_script_dir, p)
            if os.path.exists(candidate):
                p = candidate
        resolved_paths.append(p)

    warmup_done = [False]  # mutable flag shared across _benchmark_one calls
    results = []
    for data_path in resolved_paths:
        r = _benchmark_one(data_path, args.n_repeats, args.skip_warmup,
                           likelihood_cpu, likelihood_gpu, warmup_done)
        results.append(r)

    all_ok = _print_report(results, args.n_repeats)
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
