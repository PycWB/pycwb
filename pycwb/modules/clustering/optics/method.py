"""
OPTICS pixel clustering backend.

Algorithm
---------
For each resolution-specific :class:`~pycwb.types.network_cluster.FragmentCluster`:

1. Pool all accepted cluster pixels into one flat table using
   :func:`~pycwb.modules.clustering.common.pool_accepted_pixels`.
2. Build a normalised feature matrix from time/frequency bin indices (and
   optionally log-energy and energy-balance columns).
3. Run :class:`sklearn.cluster.OPTICS` to order points by reachability, then
   extract flat clusters using either the ``xi`` (steep-descent) or
   ``dbscan``-style method.
4. Reconstruct new :class:`~pycwb.types.network_cluster.Cluster` objects.

OPTICS vs DBSCAN
----------------
OPTICS is parameter-light: it does not require a single global ``eps``.
Instead it builds a full reachability ordering and extracts clusters at all
density scales simultaneously.  The ``xi`` parameter controls how steep a
reachability drop is considered a cluster boundary.  OPTICS is slower than
DBSCAN for very large pixel pools but more robust when signal and background
pixels have different densities.

Configuration
-------------
Parameters are read from ``config.clustering.optics`` when a config object
is provided; individual values can be overridden via keyword arguments.

Default parameter values
~~~~~~~~~~~~~~~~~~~~~~~~
======================== ======= ================================================
Parameter                Default  Description
======================== ======= ================================================
eps_time_bins            2.0     Normalisation divisor for time bin coordinate
eps_freq_bins            3.0     Normalisation divisor for frequency bin coordinate
min_samples              2       Min neighbours to compute reachability distance
max_eps                  2.0     Upper bound on neighbourhood radius
xi                       0.05    Steepness threshold for cluster boundary detection
cluster_method           "xi"    ``"xi"`` or ``"dbscan"``
dbscan_eps               None    Used only when cluster_method="dbscan"
log_energy_weight        0.0     Weight of log-energy feature (0 = disabled)
energy_bal_weight        0.0     Weight of energy-balance feature (0 = disabled)
noise_as_singletons      True    Keep noise pixels as single-pixel clusters
min_pixels               1       Min pixels per output cluster
======================== ======= ================================================

Interface
---------
cluster(fragment_clusters, config=None, lag_idx=None, **kwargs)
    Returns a new ``list[FragmentCluster]`` with OPTICS-clustered pixels.
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
    from sklearn.cluster import OPTICS as _OPTICS
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------

_DEFAULTS: dict = {
    "eps_time_bins":      2.0,
    "eps_freq_bins":      3.0,
    "min_samples":        2,
    "max_eps":            2.0,    # in normalised feature space
    "xi":                 0.05,
    "cluster_method":     "xi",   # "xi" or "dbscan"
    "dbscan_eps":         None,   # used only with cluster_method="dbscan"
    "log_energy_weight":  0.0,
    "energy_bal_weight":  0.0,
    "noise_as_singletons": True,
    "min_pixels":         1,
}


def _get_params(config, **overrides) -> dict:
    params = dict(_DEFAULTS)
    if config is not None:
        cfg_clustering = getattr(config, "clustering", None)
        if isinstance(cfg_clustering, dict):
            cfg_block = cfg_clustering.get("optics", None)
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

    # OPTICS requires at least min_samples points
    min_s = int(params["min_samples"])
    if n_pix < min_s:
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

    optics_kwargs: dict = dict(
        min_samples=min_s,
        max_eps=float(params["max_eps"]),
        cluster_method=str(params["cluster_method"]),
        n_jobs=1,
    )
    if params["cluster_method"] == "xi":
        optics_kwargs["xi"] = float(params["xi"])
    elif params["cluster_method"] == "dbscan" and params["dbscan_eps"] is not None:
        optics_kwargs["eps"] = float(params["dbscan_eps"])

    model = _OPTICS(**optics_kwargs)
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
    """Re-cluster pixels in *fragment_clusters* using OPTICS.

    Parameters
    ----------
    fragment_clusters : list[FragmentCluster]
        Per-resolution cluster list from :func:`coherence_single_lag`.
    config : Config | None
        Configuration object; may contain a ``clustering.optics`` sub-block.
    lag_idx : int | None
        Lag index used only for logging.
    **kwargs
        Per-call parameter overrides (e.g. ``xi=0.1``, ``min_samples=3``).

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
            "scikit-learn is required for the OPTICS clustering backend. "
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
            "[optics] %s res=%d  %d→%d clusters  (min_samples=%d xi=%.3f)",
            lag_str, res_idx, orig_n, new_n,
            params["min_samples"], params["xi"] if params["cluster_method"] == "xi" else 0,
        )
        result.append(new_fc)

    return result
