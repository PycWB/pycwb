"""
Unified entry point for the replacement clustering pipeline.

This module provides :func:`cluster_lag_candidates`, the single function that
replaces the three-call sequence::

    frag_clusters = coherence_single_lag(...)
    frag_clusters = cluster_fragment_clusters(frag_clusters, ...)
    fragment_cluster = supercluster_single_lag(...)

with two calls::

    candidates = select_pixels_single_lag(coherence_setup, lag, ...)
    fragment_cluster = cluster_lag_candidates(candidates, method=..., ...)

Each backend (``connected_components``, ``weighted_graph``, ``dbscan``,
``hdbscan``, ``optics``, and additive ``mra_*`` methods) exposes a
``cluster_candidates`` function that:

1. Runs the backend's clustering algorithm on raw pixel-candidate dicts
   (one dict per WDM resolution level).
2. Merges all per-resolution results into one
   :class:`~pycwb.types.network_cluster.FragmentCluster`.
3. Attaches time-delay amplitudes (when *td_inputs_cache* is provided).
4. Finalises for likelihood computation (when *setup* and *xtalk* are provided).

When *setup*, *xtalk*, or *td_inputs_cache* is ``None`` the corresponding
step is skipped, making it easy to call this function in unit tests without
a full pipeline setup.

Registry
--------
_LAG_BACKENDS maps method names to the module path that contains
``cluster_candidates``.  This registry is independent of the older
``_BACKENDS`` registry in :mod:`pycwb.modules.clustering.cluster`, which
points to the legacy ``cluster`` functions operating on
:class:`~pycwb.types.network_cluster.FragmentCluster` lists.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Backend registry (replacement architecture)
# ─────────────────────────────────────────────────────────────────────────────

_LAG_BACKENDS: dict[str, str] = {
    "connected_components": "pycwb.modules.clustering.connected_components.native",
    "weighted_graph":       "pycwb.modules.clustering.weighted_graph.impl",
    "dbscan":               "pycwb.modules.clustering.dbscan.impl",
    "hdbscan":              "pycwb.modules.clustering.hdbscan.impl",
    "optics":               "pycwb.modules.clustering.optics.impl",
    "mra_weighted_graph":   "pycwb.modules.clustering.mra_weighted_graph.impl",
    "mra_hdbscan":          "pycwb.modules.clustering.mra_hdbscan.impl",
}


def _load_lag_backend(name: str):
    """Import and return ``cluster_candidates`` from the named backend.

    Parameters
    ----------
    name : str
        Backend name (key in :data:`_LAG_BACKENDS`).

    Returns
    -------
    callable
        The ``cluster_candidates(pixel_candidates_by_resolution, ...)``
        function from the backend module.

    Raises
    ------
    ValueError
        If *name* is not a registered backend.
    ImportError
        If the backend module cannot be imported.
    """
    if name not in _LAG_BACKENDS:
        available = ", ".join(f'"{k}"' for k in sorted(_LAG_BACKENDS))
        raise ValueError(
            f"Unknown clustering method {name!r}.  "
            f"Available methods: {available}"
        )
    import importlib
    mod = importlib.import_module(_LAG_BACKENDS[name])
    return mod.cluster_candidates


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def cluster_lag_candidates(
    pixel_candidates_by_resolution: list[dict],
    method: str = "connected_components",
    config=None,
    lag_idx: int | None = None,
    setup: dict | None = None,
    xtalk=None,
    td_inputs_cache: dict | None = None,
    **kwargs,
):
    """Cluster raw pixel candidates and finalise for likelihood computation.

    This is the main entry point for the replacement clustering architecture.
    It replaces the three-call sequence used in
    :mod:`~pycwb.workflow.subflow.process_job_segment_clustering`:

    .. code-block:: python

        # Old (three calls, two intermediate types)
        frag_clusters = coherence_single_lag(setup, lag, veto_windows=...)
        frag_clusters = cluster_fragment_clusters(frag_clusters, method=..., ...)
        fragment_cluster = supercluster_single_lag(setup, config, frag_clusters, ...)

        # New (two calls, one unified pipeline)
        candidates = select_pixels_single_lag(setup, lag, veto_windows=...)
        fragment_cluster = cluster_lag_candidates(candidates, method=..., ...)

    Parameters
    ----------
    pixel_candidates_by_resolution : list[dict]
        Raw candidate dicts produced by
        :func:`~pycwb.modules.cwb_coherence.coherence.select_pixels_single_lag`,
        one dict per WDM resolution level.  An empty list returns ``None``
        immediately.
    method : str
        Clustering backend name.  Supported values:

        ``"connected_components"``
            Re-runs the same WDM connected-component algorithm used inside
            :func:`~pycwb.modules.cwb_coherence.coherence.coherence_single_lag`.
            Produces identical results to the native pipeline.

        ``"weighted_graph"``
            Physics-informed re-clustering using a weighted adjacency graph
            with TF-proximity and energy-balance edge weights.

        ``"dbscan"``
            Density-based clustering via :class:`sklearn.cluster.DBSCAN`.

        ``"hdbscan"``
            Hierarchical density-based clustering via
            :class:`sklearn.cluster.HDBSCAN` (scikit-learn ≥ 1.3).

        ``"optics"``
            Ordering-based density clustering via :class:`sklearn.cluster.OPTICS`.

        ``"mra_weighted_graph"``
            Multi-resolution weighted graph over selected pixels pooled from
            all WDM levels before primary clustering.  Existing non-MRA
            methods remain unchanged baselines.

        ``"mra_hdbscan"``
            Multi-resolution HDBSCAN over a scaled pooled feature matrix.
            Useful for adaptive-density morphology experiments after MRA
            feature scaling is tuned.

    config
        Configuration object.  Backend-specific parameters are read from
        ``config.clustering.<method>`` when present.  If ``None``, defaults
        are used.
    lag_idx : int or None
        Lag index.  Passed to the backend for logging and finalisation.
    setup : dict or None
        Supercluster setup dict from
        :func:`~pycwb.modules.super_cluster.super_cluster.setup_supercluster`.
        When ``None`` the TD-attachment and finalisation steps are skipped
        (useful in unit tests).
    xtalk
        Cross-talk catalog for the subnet cut inside finalisation.  When
        ``None`` finalisation is skipped.
    td_inputs_cache : dict or None
        Pre-built time-delay input cache.  When ``None`` TD-amplitude
        attachment is skipped.
    **kwargs
        Per-call parameter overrides forwarded to the backend
        (e.g. ``eps=0.8`` for the DBSCAN backend).

    Returns
    -------
    FragmentCluster or None
        Likelihood-ready cluster after all cuts and core marking, or
        ``None`` if no pixels survive the clustering and selection stages.

    Raises
    ------
    ValueError
        If *method* is not a recognised backend name.
    """
    if not pixel_candidates_by_resolution:
        logger.debug(
            "cluster_lag_candidates: empty candidate list (lag=%s) → None", lag_idx
        )
        return None

    backend_fn = _load_lag_backend(method)
    logger.debug(
        "cluster_lag_candidates: method=%r, %d resolution(s), lag=%s",
        method, len(pixel_candidates_by_resolution), lag_idx,
    )
    return backend_fn(
        pixel_candidates_by_resolution,
        config        = config,
        lag_idx       = lag_idx,
        setup         = setup,
        xtalk         = xtalk,
        td_inputs_cache = td_inputs_cache,
        **kwargs,
    )
