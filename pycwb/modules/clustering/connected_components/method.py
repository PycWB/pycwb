"""
Connected-components compatibility backend.

This backend is an *identity pass*: it returns the input
``list[FragmentCluster]`` unchanged.  Its purpose is to prove that the
clustering entry point can be inserted into the workflow without altering
the science output before a real alternative method is enabled.

Once the plumbing is verified, this backend can be extended to rebuild
connected components from scratch using the raw pixel coordinates stored in
``PixelArrays``, making it a fully standalone alternative to the coherence
stage's own clustering.

Interface
---------
cluster(fragment_clusters, config=None, lag_idx=None, **kwargs)
    Returns *fragment_clusters* unchanged.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def cluster(
    fragment_clusters: list,
    config=None,
    lag_idx: int | None = None,
    **kwargs,
) -> list:
    """Identity-pass backend: return input fragment clusters unchanged.

    Parameters
    ----------
    fragment_clusters : list[FragmentCluster]
        Per-resolution cluster list from :func:`coherence_single_lag`.
    config : Config | None
        Unused by the identity pass; reserved for future extension.
    lag_idx : int | None
        Optional lag index for logging.
    **kwargs
        Ignored.

    Returns
    -------
    list[FragmentCluster]
        The same list that was passed in (no copy).
    """
    lag_str = f"lag {lag_idx}" if lag_idx is not None else "lag ?"
    n_clusters = sum(len(fc.clusters) for fc in fragment_clusters)
    logger.debug(
        "[connected_components] %s — identity pass, %d resolution(s), %d cluster(s) total",
        lag_str, len(fragment_clusters), n_clusters,
    )
    return fragment_clusters
