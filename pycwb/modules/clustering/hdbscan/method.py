"""
HDBSCAN pixel clustering backend.

Algorithm
---------
For each resolution-specific :class:`~pycwb.types.network_cluster.FragmentCluster`:

1. Pool all accepted cluster pixels into one flat table using
   :func:`~pycwb.modules.clustering.common.pool_accepted_pixels`.
2. Build a normalised feature matrix from time/frequency bin indices (and
   optionally log-energy and energy-balance columns).
3. Run :class:`sklearn.cluster.HDBSCAN` with the configured parameters.
   Pixels labelled ``-1`` (noise / outlier) are handled according to
   ``noise_as_singletons``.
4. Reconstruct new :class:`~pycwb.types.network_cluster.Cluster` objects.

Advantages over DBSCAN
-----------------------
* Automatically selects cluster density — no single ``eps`` to tune.
* Produces a cluster hierarchy; extraction via ``min_cluster_size`` is robust
  across varying signal densities.
* Handles clusters of varying density (useful for glitch vs CBC morphology).

Requires scikit-learn ≥ 1.3 (HDBSCAN was added in that release).

Configuration
-------------
Parameters are read from ``config.clustering.hdbscan`` when a config object
is provided; individual values can be overridden via keyword arguments.

Default parameter values
~~~~~~~~~~~~~~~~~~~~~~~~
========================= ======= ================================================
Parameter                 Default  Description
========================= ======= ================================================
eps_time_bins             2.0     Normalisation divisor for time bin coordinate
eps_freq_bins             3.0     Normalisation divisor for frequency bin coordinate
min_cluster_size          2       Minimum pixels to form a dense cluster
min_samples               None    Core-point requirement (None → min_cluster_size)
cluster_selection_epsilon 0.0     Distance threshold for flat cluster extraction
cluster_selection_method  "eom"   ``"eom"`` (excess-of-mass) or ``"leaf"``
log_energy_weight         0.0     Weight of log-energy feature (0 = disabled)
energy_bal_weight         0.0     Weight of energy-balance feature (0 = disabled)
noise_as_singletons       True    Keep noise pixels as single-pixel clusters
min_pixels                1       Min pixels per output cluster
========================= ======= ================================================

Interface
---------
cluster(fragment_clusters, config=None, lag_idx=None, **kwargs)
    Returns a new ``list[FragmentCluster]`` with HDBSCAN-clustered pixels.
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
    from sklearn.cluster import HDBSCAN as _HDBSCAN
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------

_DEFAULTS: dict = {
    "eps_time_bins":              2.0,
    "eps_freq_bins":              3.0,
    "min_cluster_size":           2,
    "min_samples":                None,    # None → sklearn default (= min_cluster_size)
    "cluster_selection_epsilon":  0.0,
    "cluster_selection_method":   "eom",   # "eom" or "leaf"
    "log_energy_weight":          0.0,
    "energy_bal_weight":          0.0,
    "noise_as_singletons":        True,
    "min_pixels":                 1,
}


def _get_params(config, **overrides) -> dict:
    params = dict(_DEFAULTS)
    if config is not None:
        cfg_clustering = getattr(config, "clustering", None)
        if isinstance(cfg_clustering, dict):
            cfg_block = cfg_clustering.get("hdbscan", None)
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

    # HDBSCAN requires at least min_cluster_size samples
    min_cs = int(params["min_cluster_size"])
    if n_pix < min_cs:
        # Too few pixels to form any cluster; keep as one cluster
        from pycwb.modules.clustering.common import build_cluster_from_mask
        mask = np.ones(n_pix, dtype=bool)
        rejected = [c for c in fc.clusters if c.cluster_status > 0]
        return rebuild_fragment_cluster(
            fc, [build_cluster_from_mask(pooled, mask)] + rejected
        )

    X = build_feature_matrix(
        pooled,
        time_scale=float(params["eps_time_bins"]),
        freq_scale=float(params["eps_freq_bins"]),
        log_energy_weight=float(params["log_energy_weight"]),
        energy_bal_weight=float(params["energy_bal_weight"]),
    )

    kwargs: dict = dict(
        min_cluster_size=min_cs,
        cluster_selection_epsilon=float(params["cluster_selection_epsilon"]),
        cluster_selection_method=str(params["cluster_selection_method"]),
        n_jobs=1,
    )
    if params["min_samples"] is not None:
        kwargs["min_samples"] = int(params["min_samples"])

    model = _HDBSCAN(**kwargs)
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
    """Re-cluster pixels in *fragment_clusters* using HDBSCAN.

    Parameters
    ----------
    fragment_clusters : list[FragmentCluster]
        Per-resolution cluster list from :func:`coherence_single_lag`.
    config : Config | None
        Configuration object; may contain a ``clustering.hdbscan`` sub-block.
    lag_idx : int | None
        Lag index used only for logging.
    **kwargs
        Per-call parameter overrides (e.g. ``min_cluster_size=3``).

    Returns
    -------
    list[FragmentCluster]
        New list with the same length as *fragment_clusters*.

    Raises
    ------
    ImportError
        If scikit-learn ≥ 1.3 is not installed.
    """
    if not HAS_HDBSCAN:
        raise ImportError(
            "scikit-learn ≥ 1.3 is required for the HDBSCAN clustering backend "
            "(HDBSCAN was added in sklearn 1.3). "
            "Install or upgrade with: pip install -U scikit-learn"
        )

    params = _get_params(config, **kwargs)
    lag_str = f"lag {lag_idx}" if lag_idx is not None else "lag ?"

    result = []
    for res_idx, fc in enumerate(fragment_clusters):
        orig_n = len(fc.clusters)
        new_fc = _cluster_one_fragment(fc, params)
        new_n = len(new_fc.clusters)
        logger.debug(
            "[hdbscan] %s res=%d  %d→%d clusters  (min_cluster_size=%d)",
            lag_str, res_idx, orig_n, new_n,
            params["min_cluster_size"],
        )
        result.append(new_fc)

    return result
