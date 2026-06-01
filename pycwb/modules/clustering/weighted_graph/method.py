"""
Weighted-graph pixel clustering backend.

Algorithm
---------
For each resolution-specific :class:`~pycwb.types.network_cluster.FragmentCluster`:

1. Pool all pixels from every accepted cluster into one flat table using
   :func:`~pycwb.modules.clustering.common.pool_accepted_pixels`.
2. Build a sparse adjacency graph whose nodes are pixels.  Two pixels are
   candidates for an edge when their time and frequency bins are within the
   configured neighbourhood radii (``time_radius_bins``, ``freq_radius_bins``).
3. Compute a scalar edge weight that combines:

   * **TF proximity score** — 1 at zero separation, decreasing linearly to 0
     at the neighbourhood boundary.
   * **Energy-balance penalty** — attenuates edges whose end pixels have very
     different detector-energy ratios (proxy for non-astrophysical glitches
     from a single detector).

   .. math::

       w_{ij} = s_{TF}(i,j) \\cdot
                \\exp\\!\\left(-\\lambda_B \\,|r_i - r_j|\\right)

   where :math:`r_k = E_{k,0} / (\\sum_d E_{k,d} + \\varepsilon)` is the
   energy-balance ratio of pixel *k* across detectors and
   :math:`\\lambda_B` = ``energy_balance_weight``.

4. Prune edges with :math:`w_{ij} < \\texttt{min\\_edge\\_weight}`.
5. Run :func:`scipy.sparse.csgraph.connected_components` on the pruned graph.
6. Each component with at least ``min_pixels`` pixels becomes a new
   :class:`~pycwb.types.network_cluster.Cluster`.

Configuration
-------------
Parameters are read from ``config.clustering.weighted_graph`` when a config
object is provided; individual values can be overridden via keyword arguments.
Safe defaults apply if the config block is absent.

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

Interface
---------
cluster(fragment_clusters, config=None, lag_idx=None, **kwargs)
    Returns a new ``list[FragmentCluster]`` with re-clustered pixels.
"""

from __future__ import annotations

import logging

import numpy as np

from pycwb.modules.clustering.common import (
    pool_accepted_pixels,
    build_cluster_from_mask,
    rebuild_fragment_cluster,
)

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(*args, **kwargs):          # type: ignore[misc]
        return lambda f: f
    def prange(n):                      # type: ignore[misc]
        return range(n)

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────────
# Default parameters
# ────────────────────────────────────────────────────────────────────────────

_DEFAULTS = {
    "time_radius_bins":   2,
    "freq_radius_bins":   3,
    "min_edge_weight":    0.1,
    "energy_balance_wt":  0.5,
    "min_pixels":         1,
}


def _get_params(config, **overrides) -> dict:
    """Merge config block, defaults, and caller overrides into a param dict."""
    params = dict(_DEFAULTS)
    # Read from config.clustering.weighted_graph if present
    cfg_block = None
    if config is not None:
        cfg_clustering = getattr(config, "clustering", None)
        if cfg_clustering is not None:
            cfg_block = cfg_clustering.get("weighted_graph", None) if isinstance(cfg_clustering, dict) else None
    if cfg_block:
        for key in params:
            if key in cfg_block:
                params[key] = cfg_block[key]
    params.update(overrides)
    return params


# ────────────────────────────────────────────────────────────────────────────
# Numba-accelerated COO edge builder
# ────────────────────────────────────────────────────────────────────────────

@njit(cache=True)
def _build_adj_coo_weighted(
    time: np.ndarray,
    frequency: np.ndarray,
    energy_ratio: np.ndarray,
    time_radius: int,
    freq_radius: int,
    energy_balance_wt: float,
    min_edge_weight: float,
):
    """Build weighted COO edges for a pool of pixels.

    Parameters
    ----------
    time, frequency : int32 (n_pix,)
        Pixel TF bin coordinates.
    energy_ratio : float32 (n_pix,)
        Energy-balance ratio per pixel: E_ifo0 / (sum_ifo E_ifo + eps).
    time_radius, freq_radius : int
        Half-widths of the TF neighbourhood in bins.
    energy_balance_wt : float
        Exponential decay rate for energy-balance dissimilarity.
    min_edge_weight : float
        Prune edges with weight below this value.

    Returns
    -------
    rows, cols : list[int]
        COO-format edge indices (upper-triangle only; i < j).
    """
    n_pix = len(time)
    rows = []
    cols = []

    for i in prange(n_pix):
        ti = time[i]
        fi = frequency[i]
        ri = energy_ratio[i]
        for j in range(i + 1, n_pix):
            dt = ti - time[j]
            if dt < 0:
                dt = -dt
            if dt > time_radius:
                continue
            df = fi - frequency[j]
            if df < 0:
                df = -df
            if df > freq_radius:
                continue

            # TF proximity score (linear, 1 at origin)
            tf_score = 1.0 - 0.5 * (
                float(dt) / float(time_radius + 1) +
                float(df) / float(freq_radius + 1)
            )

            # Energy-balance penalty
            dr = ri - energy_ratio[j]
            if dr < 0.0:
                dr = -dr
            bal_factor = np.exp(-energy_balance_wt * dr)

            weight = tf_score * bal_factor
            if weight >= min_edge_weight:
                rows.append(i)
                cols.append(j)

    return rows, cols


# Pure-Python fallback (used when Numba is absent)
def _build_adj_coo_weighted_py(
    time: np.ndarray,
    frequency: np.ndarray,
    energy_ratio: np.ndarray,
    time_radius: int,
    freq_radius: int,
    energy_balance_wt: float,
    min_edge_weight: float,
):
    n_pix = len(time)
    rows: list[int] = []
    cols: list[int] = []
    for i in range(n_pix):
        for j in range(i + 1, n_pix):
            dt = abs(int(time[i]) - int(time[j]))
            if dt > time_radius:
                continue
            df = abs(int(frequency[i]) - int(frequency[j]))
            if df > freq_radius:
                continue
            tf_score = 1.0 - 0.5 * (
                dt / (time_radius + 1) + df / (freq_radius + 1)
            )
            dr = abs(float(energy_ratio[i]) - float(energy_ratio[j]))
            bal_factor = float(np.exp(-energy_balance_wt * dr))
            weight = tf_score * bal_factor
            if weight >= min_edge_weight:
                rows.append(i)
                cols.append(j)
    return rows, cols


def _compute_adj(time, frequency, energy_ratio, params):
    """Dispatch to Numba or Python adjacency builder."""
    tr = int(params["time_radius_bins"])
    fr = int(params["freq_radius_bins"])
    bw = float(params["energy_balance_wt"])
    mw = float(params["min_edge_weight"])

    if HAS_NUMBA:
        return _build_adj_coo_weighted(
            time.astype(np.int32), frequency.astype(np.int32),
            energy_ratio.astype(np.float32),
            tr, fr, bw, mw,
        )
    return _build_adj_coo_weighted_py(time, frequency, energy_ratio, tr, fr, bw, mw)


# ────────────────────────────────────────────────────────────────────────────
# Energy-balance ratio helper
# ────────────────────────────────────────────────────────────────────────────

def _energy_ratio(pa) -> np.ndarray:
    """Per-pixel energy balance ratio.

    Returns the fraction of total energy carried by the first detector::

        ratio = asnr[0]**2 / (sum_j asnr[j]**2 + eps)

    For a single-detector network the ratio is always 1.
    """
    asnr = pa.asnr          # shape (n_ifo, n_pix)
    e_per_ifo = np.square(asnr.astype(np.float64))  # (n_ifo, n_pix)
    e_total = e_per_ifo.sum(axis=0) + 1e-30           # (n_pix,)
    ratio = e_per_ifo[0] / e_total
    return ratio.astype(np.float32)


# ────────────────────────────────────────────────────────────────────────────
# Main per-fragment-cluster clustering
# ────────────────────────────────────────────────────────────────────────────

def _cluster_one_fragment(fc, params):
    """Re-cluster a single :class:`FragmentCluster` with weighted-graph CC.

    Parameters
    ----------
    fc : FragmentCluster
    params : dict

    Returns
    -------
    FragmentCluster
        New fragment cluster with re-derived clusters.
    """
    from scipy.sparse import csr_array
    from scipy.sparse.csgraph import connected_components

    pooled, _origin = pool_accepted_pixels(fc)
    n_pix = len(pooled)

    if n_pix == 0:
        return rebuild_fragment_cluster(fc, [])

    # Build weighted adjacency graph
    energy_ratio = _energy_ratio(pooled)
    rows, cols = _compute_adj(
        pooled.time, pooled.frequency, energy_ratio, params,
    )

    if rows:
        row_arr = np.array(rows, dtype=np.int32)
        col_arr = np.array(cols, dtype=np.int32)
        data = np.ones(len(rows), dtype=np.float32)
        # Symmetric sparse matrix (upper + lower triangle)
        row_sym = np.concatenate([row_arr, col_arr])
        col_sym = np.concatenate([col_arr, row_arr])
        data_sym = np.concatenate([data, data])
        adj = csr_array((data_sym, (row_sym, col_sym)), shape=(n_pix, n_pix))
    else:
        adj = csr_array((n_pix, n_pix), dtype=np.float32)

    n_components, comp_labels = connected_components(
        adj, directed=False, return_labels=True
    )

    min_pix = int(params["min_pixels"])
    new_clusters = []
    for comp_id in range(n_components):
        mask = comp_labels == comp_id
        if mask.sum() < min_pix:
            continue
        new_clusters.append(build_cluster_from_mask(pooled, mask))

    # Preserve rejected clusters unchanged (they won't be re-processed)
    rejected = [c for c in fc.clusters if c.cluster_status > 0]
    return rebuild_fragment_cluster(fc, new_clusters + rejected)


# ────────────────────────────────────────────────────────────────────────────
# Public interface
# ────────────────────────────────────────────────────────────────────────────

def cluster(
    fragment_clusters: list,
    config=None,
    lag_idx: int | None = None,
    **kwargs,
) -> list:
    """Re-cluster pixels in *fragment_clusters* using a weighted adjacency graph.

    Parameters
    ----------
    fragment_clusters : list[FragmentCluster]
        Per-resolution cluster list from :func:`coherence_single_lag`.
    config : Config | None
        Configuration object; may contain a ``clustering.weighted_graph``
        sub-block with parameter overrides.
    lag_idx : int | None
        Lag index used only for logging.
    **kwargs
        Per-call parameter overrides (e.g. ``time_radius_bins=3``).

    Returns
    -------
    list[FragmentCluster]
        New list with the same length as *fragment_clusters*.  Each element
        has the same scalar metadata as the input but a new cluster list
        derived from the weighted connected-components algorithm.
    """
    params = _get_params(config, **kwargs)
    lag_str = f"lag {lag_idx}" if lag_idx is not None else "lag ?"

    result = []
    for res_idx, fc in enumerate(fragment_clusters):
        orig_n = len(fc.clusters)
        new_fc = _cluster_one_fragment(fc, params)
        new_n = len(new_fc.clusters)
        logger.debug(
            "[weighted_graph] %s res=%d  %d→%d clusters",
            lag_str, res_idx, orig_n, new_n,
        )
        result.append(new_fc)

    return result
