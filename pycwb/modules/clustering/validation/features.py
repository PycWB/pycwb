"""
Feature extraction for Phase 1 (trigger-level) and Phase 2 (cluster-level)
clustering validation studies.

All functions are deterministic, dependency-free (NumPy / optional pandas),
and sized for use in downstream tests once representative catalogs exist.

Public API
----------
trigger_to_feature_dict(trigger) -> dict[str, float]
cluster_to_feature_dict(cluster) -> dict[str, float]
pixel_arrays_to_table(pixel_arrays, n_ifo) -> dict[str, np.ndarray]
cluster_to_tf_maps(cluster, keys) -> dict[str, object]
"""

from __future__ import annotations

import numpy as np


# ────────────────────────────────────────────────────────────────────────────
# Trigger-level features (Phase 1)
# ────────────────────────────────────────────────────────────────────────────

def trigger_to_feature_dict(trigger) -> dict[str, float]:
    """Extract a flat dict of scalar features from a :class:`~pycwb.types.trigger.Trigger`.

    Parameters
    ----------
    trigger : Trigger
        A fully-reconstructed trigger object from the cWB likelihood stage.

    Returns
    -------
    dict[str, float]
        Keys are feature names; values are Python floats.  Missing fields are
        silently set to ``float('nan')``.

    Notes
    -----
    This function is intentionally conservative: it reads only the fields that
    are always present after likelihood reconstruction (``netRHO``, ``netCC``,
    ``duration``, ``bandwidth``, ``frequency``, ``hrss``, etc.).  Add more
    fields as the catalog schema stabilises.
    """
    def _get(attr, default=float("nan")):
        v = getattr(trigger, attr, default)
        if v is None:
            return default
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    return {
        "netRHO":    _get("netRHO"),
        "netCC":     _get("netCC"),
        "duration":  _get("duration"),
        "bandwidth": _get("bandwidth"),
        "frequency": _get("frequency"),
        "hrss":      _get("hrss"),
        "like_net":  _get("like_net"),
        "sub_net":   _get("sub_net"),
        "net_ecor":  _get("net_ecor"),
        "sky_cc":    _get("sky_cc"),
    }


# ────────────────────────────────────────────────────────────────────────────
# Cluster-level features (Phase 2)
# ────────────────────────────────────────────────────────────────────────────

def cluster_to_feature_dict(cluster) -> dict[str, float]:
    """Extract a flat dict of scalar features from a :class:`~pycwb.types.network_cluster.Cluster`.

    Parameters
    ----------
    cluster : Cluster
        A cluster object (post-coherence or post-likelihood).

    Returns
    -------
    dict[str, float]
    """
    meta = getattr(cluster, "cluster_meta", None)

    def _m(attr):
        if meta is None:
            return float("nan")
        return float(getattr(meta, attr, float("nan")))

    pa = getattr(cluster, "pixel_arrays", None)
    n_pix = len(pa) if pa is not None else 0
    energy = float(np.sum(pa.likelihood)) if pa is not None and n_pix > 0 else 0.0

    return {
        "n_pix":     float(n_pix),
        "energy":    energy,
        "like_net":  _m("like_net"),
        "sub_net":   _m("sub_net"),
        "net_rho":   _m("net_rho"),
        "c_time":    _m("c_time"),
        "c_freq":    _m("c_freq"),
        "net_ecor":  _m("net_ecor"),
        "norm_cor":  _m("norm_cor"),
        "net_null":  _m("net_null"),
    }


# ────────────────────────────────────────────────────────────────────────────
# Pixel-array table (Phase 2 tensor extraction)
# ────────────────────────────────────────────────────────────────────────────

def pixel_arrays_to_table(pixel_arrays, n_ifo: int | None = None) -> dict[str, np.ndarray]:
    """Flatten a :class:`~pycwb.types.pixel_arrays.PixelArrays` to a column dict.

    Parameters
    ----------
    pixel_arrays : PixelArrays
        Pixel data for one cluster.
    n_ifo : int | None
        Number of IFOs.  Inferred from ``pixel_arrays._n_ifo`` if *None*.

    Returns
    -------
    dict[str, np.ndarray]
        ``"time"``, ``"frequency"``, ``"layers"``, ``"rate"``,
        ``"likelihood"``, ``"null"`` — shape ``(n_pix,)``.
        ``"asnr_<i>"``, ``"a_90_<i>"``, ``"pixel_index_<i>"`` — shape
        ``(n_pix,)`` for each IFO index *i*.
    """
    pa = pixel_arrays
    n_ifo = n_ifo if n_ifo is not None else pa._n_ifo

    table: dict[str, np.ndarray] = {
        "time":       np.asarray(pa.time),
        "frequency":  np.asarray(pa.frequency),
        "layers":     np.asarray(pa.layers),
        "rate":       np.asarray(pa.rate),
        "likelihood": np.asarray(pa.likelihood),
        "null":       np.asarray(pa.null),
    }
    for j in range(n_ifo):
        table[f"asnr_{j}"]        = np.asarray(pa.asnr[j])
        table[f"a_90_{j}"]        = np.asarray(pa.a_90[j])
        table[f"pixel_index_{j}"] = np.asarray(pa.pixel_index[j])

    return table


# ────────────────────────────────────────────────────────────────────────────
# TF map extraction (Phase 2)
# ────────────────────────────────────────────────────────────────────────────

def cluster_to_tf_maps(cluster, keys: tuple = ("likelihood", "null")) -> dict[str, object]:
    """Return sparse TF maps keyed by *keys* for a cluster.

    Parameters
    ----------
    cluster : Cluster
        Cluster with a populated ``pixel_arrays``.
    keys : tuple[str]
        Field names to extract as sparse maps.  Each must be a per-pixel
        scalar field of :class:`~pycwb.types.pixel_arrays.PixelArrays`.

    Returns
    -------
    dict[str, scipy.sparse.coo_array | None]
        One entry per key.  Returns ``None`` for a key if the cluster is
        empty or the field is absent.
    """
    try:
        from scipy.sparse import coo_array
    except ImportError:
        raise ImportError("scipy is required for cluster_to_tf_maps")

    pa = getattr(cluster, "pixel_arrays", None)
    if pa is None or len(pa) == 0:
        return {k: None for k in keys}

    result = {}
    for key in keys:
        values = getattr(pa, key, None)
        if values is None:
            result[key] = None
            continue
        n_time = int(pa.time.max()) + 1 if len(pa.time) > 0 else 1
        n_freq = int(pa.frequency.max()) + 1 if len(pa.frequency) > 0 else 1
        result[key] = coo_array(
            (np.asarray(values, dtype=np.float64), (pa.frequency, pa.time)),
            shape=(n_freq, n_time),
        )
    return result
