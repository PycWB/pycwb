"""
Weighted-graph replacement backend operating on raw pixel candidates.

Algorithm
---------
For each WDM resolution in the candidate list:

1. Convert the raw candidate dict to a flat
   :class:`~pycwb.types.pixel_arrays.PixelArrays` using
   :func:`~pycwb.modules.clustering.pixel_utils.build_pixel_arrays_from_candidates`.
2. Build a sparse adjacency graph over pixels.  Two pixels share an edge
   when their TF bin separation is within ``(time_radius_bins, freq_radius_bins)``
   and the edge weight exceeds ``min_edge_weight``.  The weight combines a
   linear TF-proximity score and an exponential energy-balance penalty.
3. Run :func:`scipy.sparse.csgraph.connected_components` on the pruned graph.
4. Collect per-resolution
   :class:`~pycwb.types.network_cluster.FragmentCluster` objects.

All resolutions are then merged, TD-amplitudes attached, and the result
finalised for likelihood computation.

Configuration
-------------
Parameters are read from ``config.clustering.weighted_graph`` when present;
individual values can be overridden via keyword arguments.

Default parameter values
~~~~~~~~~~~~~~~~~~~~~~~~
================== ====== =================================================
Parameter          Default Description
================== ====== =================================================
time_radius_bins   2      Half-width in time bins for adjacency
freq_radius_bins   3      Half-width in frequency bins for adjacency
min_edge_weight    0.1    Edge-weight threshold below which to prune
energy_balance_wt  0.5    Exponential decay rate for energy-balance penalty
min_pixels         1      Minimum pixels per output cluster
================== ====== =================================================

Entry point
-----------
cluster_candidates(pixel_candidates_by_resolution, config, lag_idx,
                   setup, xtalk, td_inputs_cache, **kwargs)
"""

from __future__ import annotations

import logging

import numpy as np

from pycwb.modules.clustering.pixel_utils import (
    build_pixel_arrays_from_candidates,
    build_fragment_cluster_from_candidates,
)
from pycwb.modules.clustering.common import build_cluster_from_mask

# Re-use the Numba-accelerated adjacency builder and helpers from the
# existing weighted_graph backend so we don't duplicate the heavy code.
from pycwb.modules.clustering.weighted_graph.method import (
    _get_params,
    _energy_ratio,
    _compute_adj,
)

try:
    from scipy.sparse import csr_array
    from scipy.sparse.csgraph import connected_components
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

logger = logging.getLogger(__name__)


def _cluster_one_candidates(candidates: dict, params: dict):
    """Apply weighted-graph clustering to one resolution's candidate dict."""
    pooled = build_pixel_arrays_from_candidates(candidates)
    n_pix = len(pooled)

    if n_pix == 0:
        return build_fragment_cluster_from_candidates(candidates, [])

    energy_ratio = _energy_ratio(pooled)
    rows, cols = _compute_adj(pooled.time, pooled.frequency, energy_ratio, params)

    if rows:
        row_arr  = np.array(rows, dtype=np.int32)
        col_arr  = np.array(cols, dtype=np.int32)
        data     = np.ones(len(rows), dtype=np.float32)
        row_sym  = np.concatenate([row_arr, col_arr])
        col_sym  = np.concatenate([col_arr, row_arr])
        data_sym = np.concatenate([data,    data   ])
        adj = csr_array((data_sym, (row_sym, col_sym)), shape=(n_pix, n_pix))
    else:
        adj = csr_array((n_pix, n_pix), dtype=np.float32)

    n_components, comp_labels = connected_components(adj, directed=False, return_labels=True)

    min_pix = int(params["min_pixels"])
    new_clusters = []
    for comp_id in range(n_components):
        mask = comp_labels == comp_id
        if int(mask.sum()) < min_pix:
            continue
        new_clusters.append(build_cluster_from_mask(pooled, mask))

    return build_fragment_cluster_from_candidates(candidates, new_clusters)


def cluster_candidates(
    pixel_candidates_by_resolution: list[dict],
    config=None,
    lag_idx: int | None = None,
    setup: dict | None = None,
    xtalk=None,
    td_inputs_cache: dict | None = None,
    **kwargs,
):
    """Re-cluster raw pixel candidates using a weighted adjacency graph.

    Parameters
    ----------
    pixel_candidates_by_resolution : list[dict]
        Raw candidate dicts from
        :func:`~pycwb.modules.cwb_coherence.coherence.select_pixels_single_lag`.
    config
        Config object; optional ``clustering.weighted_graph`` sub-block
        is read for parameter values.
    lag_idx : int or None
        Lag index for logging.
    setup, xtalk, td_inputs_cache
        Forwarded to :mod:`pycwb.modules.clustering.pipeline` helpers.
    **kwargs
        Per-call parameter overrides (e.g. ``time_radius_bins=3``).

    Returns
    -------
    FragmentCluster or None

    Raises
    ------
    ImportError
        If ``scipy`` is not installed.
    """
    if not HAS_SCIPY:
        raise ImportError(
            "scipy is required for the weighted_graph clustering backend. "
            "Install it with: pip install scipy"
        )

    from pycwb.modules.clustering.pipeline import (
        merge_fragment_clusters,
        attach_td_amplitudes,
        finalize_clusters_for_likelihood,
    )

    params  = _get_params(config, **kwargs)
    lag_str = f"lag {lag_idx}" if lag_idx is not None else "lag ?"

    if not pixel_candidates_by_resolution:
        logger.debug("[wg_impl] %s: no resolutions → None", lag_str)
        return None

    fragment_clusters = []
    for res_idx, candidates in enumerate(pixel_candidates_by_resolution):
        n_cand = int(len(candidates.get("frequency", [])))
        new_fc = _cluster_one_candidates(candidates, params)
        logger.debug(
            "[wg_impl] %s res=%d candidates=%d clusters=%d",
            lag_str, res_idx, n_cand, len(new_fc.clusters),
        )
        fragment_clusters.append(new_fc)

    merged = merge_fragment_clusters(fragment_clusters)
    if merged is None:
        return None

    attach_td_amplitudes(merged, config, setup, td_inputs_cache)
    return finalize_clusters_for_likelihood(merged, config, setup, xtalk, lag_idx)
