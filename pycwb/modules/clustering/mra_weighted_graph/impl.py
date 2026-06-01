"""Multi-resolution weighted-graph clustering over raw pixel candidates.

Unlike the non-MRA ``weighted_graph`` backend, this method pools selected
pixels from all WDM resolutions before primary clustering.  Cross-resolution
edges let coarse pixels bridge fine-resolution islands when their physical
time-frequency cells and detector-energy patterns are compatible.
"""

from __future__ import annotations

import logging

import numpy as np

from pycwb.modules.clustering.common import build_cluster_from_mask
from pycwb.modules.clustering.pixel_utils import pool_mra_pixel_candidates
from pycwb.types.network_cluster import FragmentCluster

try:
    from scipy.sparse import csr_array
    from scipy.sparse.csgraph import connected_components
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

logger = logging.getLogger(__name__)


_DEFAULTS = {
    "time_radius_bins": 2.0,
    "freq_radius_bins": 3.0,
    "time_gap_seconds": None,
    "freq_gap_hz": None,
    "min_edge_weight": 0.1,
    "detector_similarity_weight": 0.5,
    "energy_similarity_weight": 0.25,
    "energy_balance_wt": 0.5,
    "resolution_penalty": 0.1,
    "max_level_gap": 99,
    "enable_cross_resolution_edges": True,
    "min_pixels": 1,
    "final_defrag": False,
}


def _get_params(config, **overrides) -> dict:
    """Merge defaults, ``config.clustering.mra_weighted_graph``, and overrides."""
    params = dict(_DEFAULTS)
    cfg_block = None
    if config is not None:
        cfg_clustering = getattr(config, "clustering", None)
        if isinstance(cfg_clustering, dict):
            cfg_block = cfg_clustering.get("mra_weighted_graph")
    if cfg_block:
        for key in params:
            if key in cfg_block:
                params[key] = cfg_block[key]
    params.update(overrides)
    return params


def _interval_gap(start_i: float, stop_i: float, start_j: float, stop_j: float) -> float:
    """Return the positive gap between two intervals, or zero if they overlap."""
    return max(0.0, max(start_i, start_j) - min(stop_i, stop_j))


def _axis_score(gap: float, allowed_gap: float) -> float:
    """Convert a gap and allowed gap into a score in [0, 1]."""
    if allowed_gap <= 0.0:
        return 1.0 if gap <= 0.0 else 0.0
    if gap > allowed_gap:
        return 0.0
    return 1.0 - gap / allowed_gap


def _detector_similarity(detector_energy: np.ndarray, i: int, j: int) -> float:
    """Cosine similarity between per-detector energy vectors."""
    vi = detector_energy[i]
    vj = detector_energy[j]
    denom = float(np.linalg.norm(vi) * np.linalg.norm(vj))
    if denom <= 0.0:
        return 1.0
    return max(0.0, min(1.0, float(np.dot(vi, vj) / denom)))


def _edge_weight(pool, i: int, j: int, params: dict) -> float:
    """Compute an MRA graph edge weight for two pooled pixels."""
    pa = pool.pixel_arrays
    same_resolution = pool.resolution_index[i] == pool.resolution_index[j]
    if not same_resolution and not bool(params["enable_cross_resolution_edges"]):
        return 0.0

    level_gap = abs(int(pool.level[i]) - int(pool.level[j]))
    if level_gap > int(params["max_level_gap"]):
        return 0.0

    time_gap = _interval_gap(
        float(pool.time_start[i]), float(pool.time_stop[i]),
        float(pool.time_start[j]), float(pool.time_stop[j]),
    )
    freq_gap = _interval_gap(
        float(pool.frequency_low[i]), float(pool.frequency_high[i]),
        float(pool.frequency_low[j]), float(pool.frequency_high[j]),
    )

    if params["time_gap_seconds"] is None:
        ri = float(pa.rate[i])
        rj = float(pa.rate[j])
        time_allow = float(params["time_radius_bins"]) / max(min(ri, rj), 1e-30)
    else:
        time_allow = float(params["time_gap_seconds"])

    if params["freq_gap_hz"] is None:
        ri = float(pa.rate[i])
        rj = float(pa.rate[j])
        freq_allow = float(params["freq_radius_bins"]) * max(ri, rj) / 2.0
    else:
        freq_allow = float(params["freq_gap_hz"])

    time_score = _axis_score(time_gap, time_allow)
    freq_score = _axis_score(freq_gap, freq_allow)
    if time_score <= 0.0 or freq_score <= 0.0:
        return 0.0
    tf_score = 0.5 * (time_score + freq_score)

    det_sim = _detector_similarity(pool.detector_energy, i, j)
    det_weight = float(params["detector_similarity_weight"])
    det_factor = (1.0 - det_weight) + det_weight * det_sim

    li = max(float(pa.likelihood[i]), 0.0)
    lj = max(float(pa.likelihood[j]), 0.0)
    energy_delta = abs(np.log1p(li) - np.log1p(lj))
    energy_factor = np.exp(-float(params["energy_balance_wt"]) * energy_delta)
    energy_weight = float(params["energy_similarity_weight"])
    energy_factor = (1.0 - energy_weight) + energy_weight * energy_factor

    resolution_factor = np.exp(-float(params["resolution_penalty"]) * level_gap)
    weight = tf_score * det_factor * energy_factor * resolution_factor
    return float(weight)


def _build_edges(pool, params: dict) -> tuple[list[int], list[int]]:
    """Build upper-triangle COO edges for the pooled MRA graph."""
    n_pix = len(pool.pixel_arrays)
    rows: list[int] = []
    cols: list[int] = []
    min_weight = float(params["min_edge_weight"])

    for i in range(n_pix):
        for j in range(i + 1, n_pix):
            if _edge_weight(pool, i, j, params) >= min_weight:
                rows.append(i)
                cols.append(j)
    return rows, cols


def _fragment_cluster_from_pool(
    pool,
    clusters: list,
    pixel_candidates_by_resolution: list[dict],
) -> FragmentCluster:
    """Wrap MRA clusters in one mixed-resolution FragmentCluster."""
    pa = pool.pixel_arrays
    n_pix = sum(len(c.pixel_arrays) for c in clusters)
    if len(pa) == 0:
        rate = 0.0
        f_low = 0.0
        f_high = 0.0
    else:
        rate = float(np.max(pa.rate))
        f_low = float(np.min(pool.frequency_low))
        f_high = float(np.max(pool.frequency_high))

    starts = [float(c.get("start", 0.0)) for c in pixel_candidates_by_resolution]
    stops = [float(c.get("stop", 0.0)) for c in pixel_candidates_by_resolution]

    fc = FragmentCluster(
        rate=rate,
        start=min(starts) if starts else 0.0,
        stop=max(stops) if stops else 0.0,
        bpp=1.0,
        shift=0.0,
        f_low=f_low,
        f_high=f_high,
        n_pix=n_pix,
        run=0,
        pair=False,
        subnet_threshold=0.0,
    )
    fc.clusters = list(clusters)
    return fc


def cluster_candidates(
    pixel_candidates_by_resolution: list[dict],
    config=None,
    lag_idx: int | None = None,
    setup: dict | None = None,
    xtalk=None,
    td_inputs_cache: dict | None = None,
    **kwargs,
):
    """Cluster all selected pixels together using a multi-resolution graph."""
    if not HAS_SCIPY:
        raise ImportError(
            "scipy is required for the mra_weighted_graph clustering backend. "
            "Install it with: pip install scipy"
        )

    from pycwb.modules.clustering.pipeline import finalize_mra_clusters_for_likelihood

    params = _get_params(config, **kwargs)
    lag_str = f"lag {lag_idx}" if lag_idx is not None else "lag ?"
    pool = pool_mra_pixel_candidates(pixel_candidates_by_resolution)
    n_pix = len(pool.pixel_arrays)
    if n_pix == 0:
        logger.debug("[mra_wg] %s: no selected pixels -> None", lag_str)
        return None

    rows, cols = _build_edges(pool, params)
    if rows:
        row_arr = np.asarray(rows, dtype=np.int32)
        col_arr = np.asarray(cols, dtype=np.int32)
        data = np.ones(len(row_arr), dtype=np.float32)
        adj = csr_array(
            (
                np.concatenate([data, data]),
                (np.concatenate([row_arr, col_arr]), np.concatenate([col_arr, row_arr])),
            ),
            shape=(n_pix, n_pix),
        )
    else:
        adj = csr_array((n_pix, n_pix), dtype=np.float32)

    n_components, labels = connected_components(adj, directed=False, return_labels=True)
    min_pixels = int(params["min_pixels"])
    clusters = []
    for comp_id in range(n_components):
        mask = labels == comp_id
        if int(mask.sum()) < min_pixels:
            continue
        clusters.append(build_cluster_from_mask(pool.pixel_arrays, mask))

    logger.debug(
        "[mra_wg] %s pixels=%d edges=%d clusters=%d",
        lag_str, n_pix, len(rows), len(clusters),
    )
    if not clusters:
        return None

    fragment_cluster = _fragment_cluster_from_pool(
        pool, clusters, pixel_candidates_by_resolution
    )
    return finalize_mra_clusters_for_likelihood(
        fragment_cluster,
        config=config,
        setup=setup,
        xtalk=xtalk,
        lag_idx=lag_idx if lag_idx is not None else -1,
        td_inputs_cache=td_inputs_cache,
        final_defrag=bool(params["final_defrag"]),
    )