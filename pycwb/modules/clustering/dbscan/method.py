"""
DBSCAN pixel clustering backend.

Algorithm
---------
For each resolution-specific :class:`~pycwb.types.network_cluster.FragmentCluster`:

1. Pool all accepted cluster pixels into one flat table using
   :func:`~pycwb.modules.clustering.common.pool_accepted_pixels`.
2. Build a 2-D (or higher) feature matrix where time and frequency bin
   indices are normalised by configurable scale factors so that the default
   WDM TF neighbourhood maps to roughly ±1 in feature space.  Optional
   log-energy and energy-balance columns can be included.
3. Run :class:`sklearn.cluster.DBSCAN` with the configured ``eps`` and
   ``min_samples``.  Pixels labelled ``-1`` (noise) are handled according
   to ``noise_as_singletons``.
4. Reconstruct new :class:`~pycwb.types.network_cluster.Cluster` objects
   from the resulting label assignments.

Compared with weighted-graph clustering, DBSCAN

* requires no explicit adjacency graph construction,
* can identify clusters of arbitrary shape,
* is sensitive to the choice of ``eps`` and feature scaling, and
* may produce noise pixels when ``min_samples > 1``.

Configuration
-------------
Parameters are read from ``config.clustering.dbscan`` when a config object
is provided; individual values can be overridden via keyword arguments.

Default parameter values
~~~~~~~~~~~~~~~~~~~~~~~~
======================== ======= ================================================
Parameter                Default  Description
======================== ======= ================================================
eps_time_bins            2.0     Normalisation divisor for time bin coordinate
eps_freq_bins            3.0     Normalisation divisor for frequency bin coordinate
eps                      1.2     DBSCAN neighbourhood radius in feature space
min_samples              1       Min neighbours for a point to be a core point
log_energy_weight        0.0     Weight of log-energy feature (0 = disabled)
energy_bal_weight        0.0     Weight of energy-balance feature (0 = disabled)
noise_as_singletons      True    Keep noise pixels as single-pixel clusters
min_pixels               1       Min pixels per output cluster
======================== ======= ================================================

Interface
---------
cluster(fragment_clusters, config=None, lag_idx=None, **kwargs)
    Returns a new ``list[FragmentCluster]`` with DBSCAN-clustered pixels.
"""

from __future__ import annotations

import logging

import numpy as np

from pycwb.modules.clustering.common import (
    pool_accepted_pixels,
    rebuild_fragment_cluster,
    build_feature_matrix,
    labels_to_clusters,
)

try:
    from sklearn.cluster import DBSCAN as _DBSCAN
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------

_DEFAULTS: dict = {
    "eps_time_bins":       2.0,   # time bin normalisation scale
    "eps_freq_bins":       3.0,   # frequency bin normalisation scale
    "eps":                 1.2,   # DBSCAN eps in normalised feature space
    "min_samples":         1,     # core-point neighbour threshold
    "log_energy_weight":   0.0,   # 0 = exclude log-energy feature
    "energy_bal_weight":   0.0,   # 0 = exclude energy-balance feature
    "noise_as_singletons": True,  # keep noise pixels as single-pixel clusters
    "min_pixels":          1,     # drop output clusters with fewer pixels
}


def _get_params(config, **overrides) -> dict:
    params = dict(_DEFAULTS)
    if config is not None:
        cfg_clustering = getattr(config, "clustering", None)
        if isinstance(cfg_clustering, dict):
            cfg_block = cfg_clustering.get("dbscan", None)
            if cfg_block:
                for key in params:
                    if key in cfg_block:
                        params[key] = cfg_block[key]
    params.update(overrides)
    return params


# ---------------------------------------------------------------------------
# Per-fragment-cluster clustering
# ---------------------------------------------------------------------------

def _cluster_one_fragment(fc, params: dict):
    pooled, _origin = pool_accepted_pixels(fc)
    n_pix = len(pooled)

    if n_pix == 0:
        return rebuild_fragment_cluster(fc, [])

    X = build_feature_matrix(
        pooled,
        time_scale=float(params["eps_time_bins"]),
        freq_scale=float(params["eps_freq_bins"]),
        log_energy_weight=float(params["log_energy_weight"]),
        energy_bal_weight=float(params["energy_bal_weight"]),
    )

    model = _DBSCAN(
        eps=float(params["eps"]),
        min_samples=int(params["min_samples"]),
        algorithm="ball_tree",
        n_jobs=1,
    )
    labels = model.fit_predict(X)

    rejected = [c for c in fc.clusters if c.cluster_status > 0]
    new_clusters = labels_to_clusters(
        pooled, labels,
        noise_as_singletons=bool(params["noise_as_singletons"]),
        min_pixels=int(params["min_pixels"]),
        rejected_clusters=rejected,
    )
    return rebuild_fragment_cluster(fc, new_clusters)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def cluster(
    fragment_clusters: list,
    config=None,
    lag_idx: int | None = None,
    **kwargs,
) -> list:
    """Re-cluster pixels in *fragment_clusters* using DBSCAN.

    Parameters
    ----------
    fragment_clusters : list[FragmentCluster]
        Per-resolution cluster list from :func:`coherence_single_lag`.
    config : Config | None
        Configuration object; may contain a ``clustering.dbscan`` sub-block.
    lag_idx : int | None
        Lag index used only for logging.
    **kwargs
        Per-call parameter overrides (e.g. ``eps=0.8``, ``min_samples=2``).

    Returns
    -------
    list[FragmentCluster]
        New list with the same length as *fragment_clusters*.

    Raises
    ------
    ImportError
        If scikit-learn is not installed.
    """
    if not HAS_SKLEARN:
        raise ImportError(
            "scikit-learn is required for the DBSCAN clustering backend. "
            "Install it with: pip install scikit-learn"
        )

    params = _get_params(config, **kwargs)
    lag_str = f"lag {lag_idx}" if lag_idx is not None else "lag ?"

    result = []
    for res_idx, fc in enumerate(fragment_clusters):
        orig_n = len(fc.clusters)
        new_fc = _cluster_one_fragment(fc, params)
        new_n = len(new_fc.clusters)
        logger.debug(
            "[dbscan] %s res=%d  %d→%d clusters  (eps=%.3f min_samples=%d)",
            lag_str, res_idx, orig_n, new_n,
            params["eps"], params["min_samples"],
        )
        result.append(new_fc)

    return result
