"""Multi-resolution HDBSCAN clustering over raw pixel candidates.

This backend pools selected pixels from all WDM resolutions, builds one scaled
feature matrix in physical time/frequency/resolution space, and runs HDBSCAN
once over the pooled pixels.  It is additive to the existing per-resolution
``hdbscan`` backend.
"""

from __future__ import annotations

import logging

import numpy as np

from pycwb.modules.clustering.common import build_cluster_from_mask, labels_to_clusters
from pycwb.modules.clustering.pixel_utils import pool_mra_pixel_candidates
from pycwb.types.network_cluster import FragmentCluster

try:
    from sklearn.cluster import HDBSCAN as _HDBSCAN
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False

logger = logging.getLogger(__name__)


_DEFAULTS = {
    "time_scale_seconds": 0.01,
    "freq_scale_hz": 64.0,
    "level_weight": 0.5,
    "log_energy_weight": 0.25,
    "detector_balance_weight": 0.5,
    "min_cluster_size": 2,
    "min_samples": None,
    "cluster_selection_epsilon": 0.0,
    "cluster_selection_method": "eom",
    "noise_as_singletons": True,
    "min_pixels": 1,
    "final_defrag": False,
}


def _get_params(config, **overrides) -> dict:
    """Merge defaults, ``config.clustering.mra_hdbscan``, and overrides."""
    params = dict(_DEFAULTS)
    cfg_block = None
    if config is not None:
        cfg_clustering = getattr(config, "clustering", None)
        if isinstance(cfg_clustering, dict):
            cfg_block = cfg_clustering.get("mra_hdbscan")
    if cfg_block:
        for key in params:
            if key in cfg_block:
                params[key] = cfg_block[key]
    params.update(overrides)
    return params


def _safe_scale(value: float) -> float:
    """Return a positive scale value for feature normalization."""
    value = float(value)
    return value if value > 0.0 else 1.0


def _feature_matrix(pool, params: dict) -> np.ndarray:
    """Build a scaled feature matrix for pooled MRA pixels."""
    pa = pool.pixel_arrays
    time_center = 0.5 * (pool.time_start + pool.time_stop)
    freq_center = 0.5 * (pool.frequency_low + pool.frequency_high)

    cols = [
        time_center.astype(np.float64) / _safe_scale(params["time_scale_seconds"]),
        freq_center.astype(np.float64) / _safe_scale(params["freq_scale_hz"]),
    ]

    level_weight = float(params["level_weight"])
    if level_weight != 0.0:
        level = pool.level.astype(np.float64)
        level = level - np.min(level) if len(level) else level
        cols.append(level_weight * level)

    log_energy_weight = float(params["log_energy_weight"])
    if log_energy_weight != 0.0:
        log_energy = np.log1p(np.maximum(pa.likelihood.astype(np.float64), 0.0))
        median = np.median(log_energy) + 1e-30
        cols.append(log_energy_weight * log_energy / median)

    balance_weight = float(params["detector_balance_weight"])
    if balance_weight != 0.0:
        det_energy = pool.detector_energy.astype(np.float64)
        total = np.sum(det_energy, axis=1) + 1e-30
        balance = det_energy[:, 0] / total if det_energy.shape[1] > 0 else np.zeros(len(pa))
        cols.append(balance_weight * balance)

    return np.column_stack(cols)


def _fragment_cluster_from_pool(
    pool,
    clusters: list,
    pixel_candidates_by_resolution: list[dict],
) -> FragmentCluster:
    """Wrap MRA HDBSCAN clusters in one mixed-resolution FragmentCluster."""
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


def _labels_to_mra_clusters(pool, labels: np.ndarray, params: dict) -> list:
    """Convert HDBSCAN labels into mixed-resolution Cluster objects."""
    return labels_to_clusters(
        pool.pixel_arrays,
        labels,
        noise_as_singletons=bool(params["noise_as_singletons"]),
        min_pixels=int(params["min_pixels"]),
    )


def cluster_candidates(
    pixel_candidates_by_resolution: list[dict],
    config=None,
    lag_idx: int | None = None,
    setup: dict | None = None,
    xtalk=None,
    td_inputs_cache: dict | None = None,
    **kwargs,
):
    """Cluster all selected pixels together using MRA HDBSCAN."""
    if not HAS_HDBSCAN:
        raise ImportError(
            "scikit-learn >= 1.3 is required for the mra_hdbscan backend. "
            "Install it with: pip install 'scikit-learn>=1.3'"
        )

    from pycwb.modules.clustering.pipeline import finalize_mra_clusters_for_likelihood

    params = _get_params(config, **kwargs)
    lag_str = f"lag {lag_idx}" if lag_idx is not None else "lag ?"
    pool = pool_mra_pixel_candidates(pixel_candidates_by_resolution)
    n_pix = len(pool.pixel_arrays)
    if n_pix == 0:
        logger.debug("[mra_hdbscan] %s: no selected pixels -> None", lag_str)
        return None

    min_cluster_size = int(params["min_cluster_size"])
    if n_pix < min_cluster_size:
        if n_pix < int(params["min_pixels"]):
            return None
        clusters = [build_cluster_from_mask(pool.pixel_arrays, np.ones(n_pix, dtype=bool))]
    else:
        hdb_kwargs: dict = dict(
            min_cluster_size=min_cluster_size,
            cluster_selection_epsilon=float(params["cluster_selection_epsilon"]),
            cluster_selection_method=str(params["cluster_selection_method"]),
            n_jobs=1,
        )
        if params["min_samples"] is not None:
            hdb_kwargs["min_samples"] = int(params["min_samples"])

        labels = _HDBSCAN(**hdb_kwargs).fit_predict(_feature_matrix(pool, params))
        clusters = _labels_to_mra_clusters(pool, labels, params)

    logger.debug(
        "[mra_hdbscan] %s pixels=%d clusters=%d (min_cluster_size=%d)",
        lag_str, n_pix, len(clusters), min_cluster_size,
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