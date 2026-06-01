"""
Native connected-components replacement backend.

This module is the *replacement* counterpart to the legacy
``connected_components/method.py`` identity pass.  Instead of returning the
input unchanged it re-runs the same pixel-clustering logic that
:func:`~pycwb.modules.cwb_coherence.coherence.coherence_single_lag` uses
internally (``cluster_pixels`` + coherence-level selections), then merges all
per-resolution :class:`~pycwb.types.network_cluster.FragmentCluster` objects,
attaches time-delay amplitudes, and finalises for likelihood computation.

Entry point
-----------
cluster_candidates(pixel_candidates_by_resolution, config, lag_idx,
                   setup, xtalk, td_inputs_cache, **kwargs)
    Replacement for the
    ``coherence_single_lag → cluster_fragment_clusters → supercluster_single_lag``
    triple.  Returns a single likelihood-ready
    :class:`~pycwb.types.network_cluster.FragmentCluster` or *None*.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def cluster_candidates(
    pixel_candidates_by_resolution: list[dict],
    config=None,
    lag_idx: int | None = None,
    setup: dict | None = None,
    xtalk=None,
    td_inputs_cache: dict | None = None,
    **kwargs,
):
    """Run native WDM connected-component clustering on raw pixel candidates.

    For each resolution in *pixel_candidates_by_resolution*:

    1. Call :func:`~pycwb.modules.cwb_coherence.coherence.cluster_pixels`
       with the same ``kt``/``kf`` values used by
       :func:`~pycwb.modules.cwb_coherence.coherence.coherence_single_lag`
       (``kt=2, kf=3`` for ``pattern != 0``; ``kt=1, kf=1`` otherwise).
    2. Apply coherence-level statistical cuts (``subrho``, ``subnet``).
    3. Remove rejected clusters.

    All per-resolution results are then merged, TD-amplitudes are attached,
    and the merged cluster is finalised for likelihood computation.

    Parameters
    ----------
    pixel_candidates_by_resolution : list[dict]
        Raw candidate dicts from
        :func:`~pycwb.modules.cwb_coherence.coherence.select_pixels_single_lag`.
    config
        Configuration object.  Not used for the clustering step itself (the
        WDM algorithm has no tunable parameters here), but forwarded to the
        finalization helpers.
    lag_idx : int or None
        Lag index used for logging and finalisation.
    setup : dict or None
        Supercluster setup dict from
        :func:`~pycwb.modules.super_cluster.super_cluster.setup_supercluster`.
        When ``None`` the finalisation step is skipped (test / offline mode).
    xtalk
        Cross-talk catalog forwarded to :func:`finalize_clusters_for_likelihood`.
    td_inputs_cache : dict or None
        Pre-built TD input cache.  When ``None`` TD attachment is skipped.
    **kwargs
        Accepted but ignored (keeps the entry-point signature uniform).

    Returns
    -------
    FragmentCluster or None
        Likelihood-ready cluster, or ``None`` if no pixels survive selection.
    """
    from pycwb.modules.cwb_coherence.coherence import cluster_pixels
    from pycwb.modules.clustering.pipeline import (
        merge_fragment_clusters,
        attach_td_amplitudes,
        finalize_clusters_for_likelihood,
    )

    lag_str = f"lag {lag_idx}" if lag_idx is not None else "lag ?"

    if not pixel_candidates_by_resolution:
        logger.debug("[native_cc] %s: no resolutions → None", lag_str)
        return None

    fragment_clusters = []
    for res_idx, candidates in enumerate(pixel_candidates_by_resolution):
        pattern      = int(candidates.get("pattern", 1))
        select_subrho = float(candidates.get("select_subrho", 0.0))
        select_subnet = float(candidates.get("select_subnet", 0.0))
        n_candidates  = int(len(candidates.get("frequency", [])))

        if pattern != 0:
            fc = cluster_pixels(candidates, kt=2, kf=3)
            fc.select("subrho", select_subrho)
            fc.select("subnet", select_subnet)
        else:
            fc = cluster_pixels(candidates, kt=1, kf=1)

        fc.remove_rejected()

        logger.debug(
            "[native_cc] %s res=%d candidates=%d clusters=%d pixels=%d",
            lag_str, res_idx, n_candidates, fc.event_count(), fc.pixel_count(),
        )
        fragment_clusters.append(fc)

    merged = merge_fragment_clusters(fragment_clusters)
    if merged is None:
        logger.debug("[native_cc] %s: no clusters after merge → None", lag_str)
        return None

    attach_td_amplitudes(merged, config, setup, td_inputs_cache)
    return finalize_clusters_for_likelihood(merged, config, setup, xtalk, lag_idx)
