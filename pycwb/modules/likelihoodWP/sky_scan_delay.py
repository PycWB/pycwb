"""Sky scan grouped by identical detector delays; per-direction math is unchanged.

The group order is prepared once for a sky grid. Only data loads, pixel energy
and the initial mask are shared within a group. DPF and every sky statistic
are recomputed for each valid direction. Final tie-breaking uses the original
sky_valid_indices order, independent of group traversal order.
"""

import numpy as np
from .sky_scan_scratch import scan_sky_scratch as scan_sky_grouped_delays

__all__ = ["make_delay_groups", "delay_groups_for_grid", "scan_sky_grouped_delays"]


def make_delay_groups(ml, reuse_delays=True):
    """Return all sky indices grouped by equal integer detector-delay tuples.

    Parameters
    ----------
    ml : numpy.ndarray
        Integer detector delays, shape (n_ifo, n_sky).

    reuse_delays : bool, optional
        Group equal delays by default; false prepares one group per direction.

    Returns
    -------
    order : numpy.ndarray
        Int64 permutation grouping identical detector-delay columns; ties are stable.
    offsets : numpy.ndarray
        Int64 group boundaries, including zero and the final permutation length.
    """
    if not reuse_delays:
        return np.arange(ml.shape[1], dtype=np.int64), np.arange(ml.shape[1] + 1, dtype=np.int64)
    _, inverse = np.unique(ml.T, axis=0, return_inverse=True)
    order = np.argsort(inverse, kind="stable").astype(np.int64)
    counts = np.bincount(inverse)
    offsets = np.empty(len(counts) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    return order, offsets


def delay_groups_for_grid(setup, ml, big_cluster=False, reuse_delays=True):
    """Cache immutable setup geometry separately for main and coarse grids.

    Per-cluster sky masks are deliberately not cached here. As with the other
    prepared sky geometry, callers must replace/rebuild ml rather than mutate
    its contents in place after preparing a setup.

    Parameters
    ----------
    setup : dict
        Job-owned setup mutated to retain main and coarse delay-group caches.
    ml : numpy.ndarray
        Immutable-by-contract integer delay grid, shape (n_ifo, n_sky).
    big_cluster : bool, optional
        Select the separate coarse-grid cache when true.

    reuse_delays : bool, optional
        Group equal delays by default; false prepares one group per direction.

    Returns
    -------
    tuple of numpy.ndarray
        Cached (order, offsets), reused while the grid object is unchanged.

    Notes
    -----
    Cache invalidation tests object identity, not contents. Do not mutate the grid
    or cached arrays in place. Replace the grid to rebuild its groups. Sky masks
    remain per-cluster inputs and are never stored in this geometry cache.
    """
    cache = setup.setdefault("_delay_group_cache", {})
    key = ("big" if big_cluster else "main", bool(reuse_delays))
    cached = cache.get(key)
    if cached is None or cached[0] is not ml:
        cached = (ml, *make_delay_groups(ml, reuse_delays))
        cache[key] = cached
    return cached[1:]
