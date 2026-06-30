#!/usr/bin/env python
"""Warm persistent Numba cache entries for the installed pycWB wheel.

This script intentionally uses tiny synthetic arrays. It is meant for Docker
image builds, where running the full integration pipeline is too expensive and
test fixtures may not be present in the repository.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from collections.abc import Callable

import numpy as np


def _run(name: str, func: Callable[[], None], keep_going: bool) -> bool:
    print(f"warming {name}...", flush=True)
    try:
        func()
    except Exception:
        print(f"failed while warming {name}", file=sys.stderr, flush=True)
        traceback.print_exc()
        if not keep_going:
            raise
        return False
    return True


def _warm_types() -> None:
    from numba.core.registry import CPUDispatcher

    from pycwb.types.job import _generate_extended_lag_ids_core
    from pycwb.types import time_series

    lag_site = np.array([0, 1], dtype=np.int64)
    _generate_extended_lag_ids_core(2, 2, 0, 3, lag_site, False, 256)

    resample = getattr(time_series, "_cwb_resample", None)
    if isinstance(resample, CPUDispatcher):
        data = np.linspace(0.0, 1.0, 16, dtype=np.float64)
        resample(data, 8)


def _warm_td_vector_batch() -> None:
    from pycwb.utils import td_vector_batch as td

    m = 4
    n_coeffs = 2
    j = 4
    k = 2
    padded00 = np.arange(16 * (m + 1), dtype=np.float32).reshape(16, m + 1) / 100.0
    padded90 = padded00 + np.float32(0.125)
    t0 = np.ones((2 * j + 1, 2 * n_coeffs + 1), dtype=np.float64)
    tx = np.full_like(t0, 0.25)
    pixel_indices = np.array([2 * (m + 1) + 1, 3 * (m + 1) + 2], dtype=np.int32)

    td._get_pixel_amplitude_nb(2, 1, 0, padded00, t0, tx, m, n_coeffs, j, False)
    td.batch_get_td_vecs(pixel_indices, padded00, padded90, t0, tx, m, n_coeffs, k, j)


def _warm_likelihood_dpf() -> None:
    from pycwb.modules.likelihoodWP import dpf

    fp0 = np.array([0.35, 0.75], dtype=np.float32)
    fx0 = np.array([0.25, -0.45], dtype=np.float32)
    rms = np.array(
        [[1.0, 0.8], [0.9, 1.1], [1.2, 0.7]],
        dtype=np.float32,
    )
    fp = np.vstack((fp0, fp0 * np.float32(0.8))).astype(np.float32)
    fx = np.vstack((fx0, fx0 * np.float32(1.1))).astype(np.float32)
    valid = np.array([0, 1], dtype=np.int64)

    dpf.avx_dpf_ps(fp0, fx0, rms)
    dpf.dpf_np_loops_local(fp0, fx0, rms)
    dpf.dpf_np_loops(fp0, fx0, rms)
    dpf.dpf_np(fp0, fx0, rms)
    dpf.dpf_np_loops_vec(fp0, fx0, rms)
    dpf.calculate_dpf(fp, fx, rms, 2, 2, 0.1, 1.0, valid)


def _warm_likelihood_sky() -> None:
    from pycwb.modules.likelihoodWP import dpf, sky_scan, sky_stat

    v00 = np.array([[0.3, 0.4, 0.2], [0.2, 0.5, 0.1]], dtype=np.float32)
    v90 = np.array([[0.1, 0.2, 0.3], [0.4, 0.2, 0.1]], dtype=np.float32)
    _, _, energy_total, mask = sky_stat.load_data_from_td(v00, v90, 0.0)

    fp0 = np.array([0.35, 0.75], dtype=np.float32)
    fx0 = np.array([0.25, -0.45], dtype=np.float32)
    rms = np.array(
        [[1.0, 0.8], [0.9, 1.1], [1.2, 0.7]],
        dtype=np.float32,
    )
    _, f_arr, f_cross, fp_norm, fx_norm, si, co, ni = dpf.dpf_np_loops_vec(fp0, fx0, rms)
    reg = np.array([1.0, 1.0, 0.0], dtype=np.float32)

    sky_stat._avx_loadata_ps(v00, v90, 0.0)
    _, sig00, sig90, mask2, *_ = sky_stat.avx_GW_ps(
        v00, v90, f_arr, f_cross, fp_norm, fx_norm, ni, energy_total, mask, reg
    )
    _, si2, co2, _, _ = sky_stat.avx_ort_ps(sig00, sig90, mask2)
    sky_stat.avx_stat_ps(v00, v90, sig00, sig90, si2, co2, mask2)

    n_ifo = 2
    n_pix = 3
    n_sky = 2
    fp = np.vstack((fp0, fp0 * np.float32(0.8))).astype(np.float32)
    fx = np.vstack((fx0, fx0 * np.float32(1.1))).astype(np.float32)
    td00 = np.repeat(v00[np.newaxis, :, :], 3, axis=0).astype(np.float32)
    td90 = np.repeat(v90[np.newaxis, :, :], 3, axis=0).astype(np.float32)
    ml = np.zeros((n_ifo, n_sky), dtype=np.int64)
    valid = np.array([0, 1], dtype=np.int64)
    sky_scan.scan_sky_for_best_fit(
        n_ifo, n_pix, n_sky, fp, fx, rms, td00, td90, ml,
        reg, -1.0, -1.0, 0.0, valid,
    )


def _warm_likelihood_packets() -> None:
    from pycwb.modules.likelihoodWP import packet_ops

    p = np.array([[0.3, 0.4, 0.2], [0.2, 0.5, 0.1]], dtype=np.float64)
    q = np.array([[0.1, 0.2, 0.3], [0.4, 0.2, 0.1]], dtype=np.float64)
    mask = np.ones(3, dtype=np.float32)
    xtalks = np.array(
        [
            [0, 0, 0, 0, 0.5, 0.0, 0.0, 0.5],
            [1, 0, 0, 0, 0.5, 0.0, 0.0, 0.5],
            [2, 0, 0, 0, 0.5, 0.0, 0.0, 0.5],
        ],
        dtype=np.float32,
    )
    xtalks_lookup = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int64)
    q_energy = np.ones(2, dtype=np.float64)
    p_energy = np.full(2, 0.8, dtype=np.float64)
    ec = np.ones(3, dtype=np.float64)

    packet_ops.avx_packet_ps(p.astype(np.float32), q.astype(np.float32), mask)
    _, _, _, q_norm = packet_ops.packet_norm_numpy(
        p, q, xtalks, xtalks_lookup, mask.astype(np.float64), q_energy
    )
    packet_ops.gw_norm_numpy(q_norm, q_energy, p_energy, ec)


def _warm_detection_statistics() -> None:
    from pycwb.modules.likelihoodWP import detection_statistics as ds

    x = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    y = np.array([1.0, 0.9, 0.8], dtype=np.float64)
    xerr = np.full(3, 0.02, dtype=np.float64)
    yerr = np.full(3, 0.03, dtype=np.float64)
    wgt = np.ones(3, dtype=np.float64)
    m_vals = np.array([-2.0, -1.0, 1.0], dtype=np.float64)
    candidates = np.array([0, 1], dtype=np.int64)

    ds._count_chirp_track_overlaps_numba(x, y, xerr, yerr, 0.01, m_vals)
    ds._fit_chirp_track_candidates_numba(
        x, y, xerr, yerr, wgt, 0.01, m_vals, candidates, 2, 100.0
    )


def _warm_xtalk() -> None:
    from pycwb.modules.xtalk import monster

    layers = np.array([1], dtype=np.int32)
    xtalk_coeff = np.array(
        [[0.0, 0.5, 0.1, 0.2, 0.4], [1.0, 0.4, 0.1, 0.2, 0.3]],
        dtype=np.float32,
    )
    lookup = np.zeros((1, 1, 2, 2, 2), dtype=np.int32)
    lookup[0, 0, 0, 0] = np.array([0, 1], dtype=np.int32)
    lookup[0, 0, 1, 0] = np.array([1, 2], dtype=np.int32)
    pixels = np.array([[2, 0], [2, 1]], dtype=np.int64)

    monster.getXTalk(2, 0, 2, 0, layers, xtalk_coeff, lookup)
    monster.getXTalk_pixels_numba(pixels, True, layers, xtalk_coeff, lookup)
    monster.getXTalk_pixels_fast(pixels, True, layers, xtalk_coeff, lookup)

    n_ifo = 2
    n_pix = 3
    null_k = np.array([0, 1, 2], dtype=np.int64)
    like_k = np.array([0, 1, 2], dtype=np.int64)
    pn = np.ones((n_ifo, n_pix), dtype=np.float64)
    pN = pn * 0.5
    ps = pn * 0.75
    pS = pn * 0.25
    gn = np.ones(n_pix, dtype=np.float64)
    ec = np.ones(n_pix, dtype=np.float64)
    xtalks = np.array(
        [
            [0, 0, 0, 0, 0.5, 0.0, 0.0, 0.5],
            [1, 0, 0, 0, 0.5, 0.0, 0.0, 0.5],
            [2, 0, 0, 0, 0.5, 0.0, 0.0, 0.5],
        ],
        dtype=np.float32,
    )
    xtalks_lookup = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int64)
    mask = np.ones(n_pix, dtype=np.bool_)
    null_out = np.zeros(n_pix, dtype=np.float64)
    like_out = np.zeros(n_pix, dtype=np.float64)
    monster._compute_null_likelihood_numba(
        null_k, like_k, pn, pN, ps, pS, gn, ec,
        xtalks_lookup, xtalks, mask, mask, null_out, like_out,
    )


def _warm_supercluster() -> None:
    from pycwb.modules.super_cluster_native import utils

    links = np.array([[0, 1], [2, 3]], dtype=np.int32)
    utils.aggregate_clusters(links)
    utils.remove_duplicates_sorted(np.array([[0, 1], [0, 1], [1, 2]], dtype=np.int32))

    link_pixels = np.array(
        [
            [0.000, 16.0, 0.01, 50.0, 0.0, 0.0, 0.0],
            [0.005, 16.2, 0.01, 50.0, 1.0, 0.0, 0.0],
            [0.100, 64.0, 0.02, 25.0, 2.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    utils.get_cluster_links(link_pixels, 2.0, 2)
    utils.get_defragment_link(link_pixels, 0.02, 4.0, 2)

    stat_pixels = np.array(
        [
            [1.0, 0.0, 16.0, 64.0, 4.0, 1.2, 0.8, 0.7],
            [1.0, 1.0, 32.0, 64.0, 4.0, 1.1, 0.6, 0.9],
            [1.0, 2.0, 48.0, 32.0, 8.0, 0.9, 0.5, 0.6],
        ],
        dtype=np.float64,
    )
    utils.calculate_statistics_arrays(
        stat_pixels[:, 0].astype(np.bool_),
        stat_pixels[:, 1],
        stat_pixels[:, 2],
        stat_pixels[:, 3],
        stat_pixels[:, 4],
        stat_pixels[:, 5],
        stat_pixels[:, 6:].T,
        "L",
        True,
        False,
        1,
        0.1,
        0.0,
    )
    utils.calculate_statistics(stat_pixels, "L", True, False, 1, 0.1, 0.0)


def _warm_coherence_helpers() -> None:
    import importlib

    coherence = importlib.import_module("pycwb.modules.coherence_native.coherence")

    f_arr = np.array([1, 1, 2], dtype=np.int64)
    t_arr = np.array([1, 2, 2], dtype=np.int64)
    coherence._label_components_grid(f_arr, t_arr, 4, 4, 1, 1)

    combined = np.ones((5, 6), dtype=np.float64)
    coherence._candidate_passes_support_numba(combined, 2, 2, 3, 0.1, 10.0, 0.1)

    asnr = np.array([[0.8, 0.7], [0.6, 0.5], [0.9, 0.4]], dtype=np.float64)
    noise = np.ones_like(asnr)
    offsets = np.array([0, 2, 3], dtype=np.int64)
    coherence._subnet_subrho_numba(asnr, noise, 1.0)
    coherence._subnet_subrho_batch_numba(asnr, noise, offsets, 1.0)

    arrays_stack = np.ones((2, 5, 8), dtype=np.float64)
    shift_bins = np.array([0, 1], dtype=np.int64)
    veto = np.ones(8, dtype=np.int16)
    coherence._align_threshold_map_numba(
        arrays_stack, shift_bins, 1, 5, veto, True, 1, 1, 4, 0.1, 2.0
    )


def _count_cached_signatures() -> int:
    import gc
    from numba.core.registry import CPUDispatcher

    total = 0
    for obj in gc.get_objects():
        try:
            if isinstance(obj, CPUDispatcher):
                total += len(obj.signatures)
        except ReferenceError:
            continue
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="continue after a warmup group fails",
    )
    args = parser.parse_args()

    print("NUMBA_CACHE_DIR=" + os.environ.get("NUMBA_CACHE_DIR", ""), flush=True)
    tasks: list[tuple[str, Callable[[], None]]] = [
        ("types", _warm_types),
        ("td-vector-batch", _warm_td_vector_batch),
        ("likelihood-dpf", _warm_likelihood_dpf),
        ("likelihood-sky", _warm_likelihood_sky),
        ("likelihood-packets", _warm_likelihood_packets),
        ("detection-statistics", _warm_detection_statistics),
        ("xtalk", _warm_xtalk),
        ("supercluster", _warm_supercluster),
        ("coherence-helpers", _warm_coherence_helpers),
    ]

    failures = 0
    for name, func in tasks:
        if not _run(name, func, args.keep_going):
            failures += 1

    print(f"warmed numba signatures: {_count_cached_signatures()}", flush=True)
    if failures:
        print(f"numba warmup completed with {failures} failure(s)", file=sys.stderr)
        return 1
    print("numba warmup complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
