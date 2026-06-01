"""
Entry point for the Phase 3 replaceable clustering stage.

Usage
-----
Insert between :func:`~pycwb.modules.cwb_coherence.coherence.coherence_single_lag`
and :func:`~pycwb.modules.super_cluster.super_cluster.supercluster_single_lag`:

.. code-block:: python

    frag_clusters = coherence_single_lag(setup, lag, ...)
    frag_clusters = cluster_fragment_clusters(
        frag_clusters,
        method=getattr(config, "clustering_method", "connected_components"),
        config=config,
        lag_idx=lag,
    )
    fragment_cluster = supercluster_single_lag(setup, config, frag_clusters, lag, ...)

API
---
cluster_fragment_clusters(fragment_clusters, method, config, lag_idx, **kwargs)
    Dispatch to the named backend and return a replacement list of
    :class:`~pycwb.types.network_cluster.FragmentCluster` objects.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Registry of available backends
# ─────────────────────────────────────────────────────────────────────────────

_BACKENDS: dict[str, str] = {
    "connected_components": "pycwb.modules.clustering.connected_components",
    "weighted_graph":       "pycwb.modules.clustering.weighted_graph",
    "dbscan":               "pycwb.modules.clustering.dbscan",
    "hdbscan":              "pycwb.modules.clustering.hdbscan",
    "optics":               "pycwb.modules.clustering.optics",
}


def _load_backend(name: str):
    """Import and return the ``cluster`` function from the named backend package.

    Parameters
    ----------
    name : str
        Backend name (key in :data:`_BACKENDS`).

    Returns
    -------
    callable
        The ``cluster(fragment_clusters, config, lag_idx, **kwargs)`` function.

    Raises
    ------
    ValueError
        If *name* is not a registered backend.
    ImportError
        If the backend module cannot be imported.
    """
    if name not in _BACKENDS:
        available = ", ".join(f'"{k}"' for k in sorted(_BACKENDS))
        raise ValueError(
            f"Unknown clustering method {name!r}.  "
            f"Available methods: {available}"
        )
    import importlib
    mod = importlib.import_module(_BACKENDS[name])
    return mod.cluster


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def cluster_fragment_clusters(
    fragment_clusters: list,
    method: str = "connected_components",
    config=None,
    lag_idx: int | None = None,
    **kwargs,
) -> list:
    """Apply the named clustering backend to one lag's fragment clusters.

    This function accepts and returns the same ``list[FragmentCluster]``
    shape produced by :func:`~pycwb.modules.cwb_coherence.coherence.coherence_single_lag`,
    so it can be inserted into the per-lag loop as a drop-in reclustering step.

    Parameters
    ----------
    fragment_clusters : list[FragmentCluster]
        Per-resolution cluster list for one lag (output of ``coherence_single_lag``).
        An empty list is returned unchanged.
    method : str
        Name of the clustering backend to use.  Supported values:

        ``"connected_components"``
            Identity pass — returns the input unchanged.  Use this to verify
            the insertion point without changing science output.

        ``"weighted_graph"``
            Physics-informed re-clustering using a weighted adjacency graph
            with TF-proximity and energy-balance edge weights.

        ``"dbscan"``
            Density-based clustering via :class:`sklearn.cluster.DBSCAN`.
            Groups pixels within a configurable neighbourhood radius ``eps``
            in normalised TF feature space.  Requires scikit-learn.

        ``"hdbscan"``
            Hierarchical density-based clustering via
            :class:`sklearn.cluster.HDBSCAN` (scikit-learn ≥ 1.3).
            No global ``eps`` required; uses ``min_cluster_size`` instead.

        ``"optics"``
            Ordering-based density clustering via :class:`sklearn.cluster.OPTICS`.
            Extracts clusters across all density scales; robust when signal
            and background pixels have different densities.  Requires
            scikit-learn.

    config : Config | None
        Configuration object.  Backend-specific parameters are read from
        ``config.clustering.<method>`` when present.  If *None*, defaults
        are used.
    lag_idx : int | None
        Lag index passed through to the backend for logging.
    **kwargs
        Per-call parameter overrides forwarded to the backend.

    Returns
    -------
    list[FragmentCluster]
        A new (or the same) list of fragment clusters, one per resolution,
        ready for :func:`~pycwb.modules.super_cluster.super_cluster.supercluster_single_lag`.

    Raises
    ------
    ValueError
        If *method* is not recognised.
    """
    if not fragment_clusters:
        return fragment_clusters

    backend_fn = _load_backend(method)
    logger.debug(
        "cluster_fragment_clusters: method=%r, %d resolution(s), lag=%s",
        method, len(fragment_clusters), lag_idx,
    )
    return backend_fn(
        fragment_clusters,
        config=config,
        lag_idx=lag_idx,
        **kwargs,
    )
