"""
Evaluation metrics for clustering quality (Phase 1/2 studies).

Functions here do not depend on large catalogs and can be unit-tested with
synthetic label arrays.  Large-catalog evaluation is deferred.

Public API
----------
cluster_size_stats(labels) -> dict
adjusted_rand(true_labels, pred_labels) -> float
normalized_mutual_info(true_labels, pred_labels) -> float
injection_purity(cluster_labels, injection_labels) -> float
"""

from __future__ import annotations

import numpy as np


def cluster_size_stats(labels: np.ndarray) -> dict:
    """Return count and size distribution statistics for a label array.

    Parameters
    ----------
    labels : np.ndarray, shape (n_samples,) int
        Cluster labels.  ``-1`` is treated as noise (excluded from stats).

    Returns
    -------
    dict with keys:
        ``n_clusters``, ``n_noise``, ``sizes`` (array of cluster sizes),
        ``mean_size``, ``median_size``, ``max_size``.
    """
    labels = np.asarray(labels, dtype=np.int64)
    noise_mask = labels < 0
    n_noise = int(noise_mask.sum())
    valid = labels[~noise_mask]

    if len(valid) == 0:
        return {
            "n_clusters": 0,
            "n_noise": n_noise,
            "sizes": np.zeros(0, dtype=np.int64),
            "mean_size": 0.0,
            "median_size": 0.0,
            "max_size": 0,
        }

    unique, counts = np.unique(valid, return_counts=True)
    return {
        "n_clusters":   len(unique),
        "n_noise":      n_noise,
        "sizes":        counts,
        "mean_size":    float(counts.mean()),
        "median_size":  float(np.median(counts)),
        "max_size":     int(counts.max()),
    }


def adjusted_rand(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    """Adjusted Rand Index between two label arrays.

    Parameters
    ----------
    true_labels, pred_labels : array-like of int

    Returns
    -------
    float
        ARI in [-1, 1].  Requires ``scikit-learn``.

    Raises
    ------
    ImportError
        If scikit-learn is unavailable.
    """
    try:
        from sklearn.metrics import adjusted_rand_score
    except ImportError:
        raise ImportError("scikit-learn is required for adjusted_rand")

    return float(adjusted_rand_score(true_labels, pred_labels))


def normalized_mutual_info(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    """Normalised Mutual Information between two label arrays.

    Parameters
    ----------
    true_labels, pred_labels : array-like of int

    Returns
    -------
    float
        NMI in [0, 1].  Requires ``scikit-learn``.

    Raises
    ------
    ImportError
        If scikit-learn is unavailable.
    """
    try:
        from sklearn.metrics import normalized_mutual_info_score
    except ImportError:
        raise ImportError("scikit-learn is required for normalized_mutual_info")

    return float(normalized_mutual_info_score(true_labels, pred_labels))


def injection_purity(
    cluster_labels: np.ndarray,
    injection_labels: np.ndarray,
) -> float:
    """Compute injection-family purity averaged over all clusters.

    For each cluster, purity is the fraction of its members belonging to the
    most common injection family (label value).  The overall purity is the
    pixel-weighted mean.

    Parameters
    ----------
    cluster_labels : np.ndarray, shape (n_pix,) int
        Which cluster each pixel belongs to.
    injection_labels : np.ndarray, shape (n_pix,) int
        Ground-truth family label for each pixel (e.g., injection index or
        ``-1`` for background).

    Returns
    -------
    float
        Weighted mean purity in [0, 1].
    """
    cluster_labels = np.asarray(cluster_labels, dtype=np.int64)
    injection_labels = np.asarray(injection_labels, dtype=np.int64)

    unique_clusters = np.unique(cluster_labels[cluster_labels >= 0])
    if len(unique_clusters) == 0:
        return float("nan")

    total_pix = 0
    weighted_purity = 0.0
    for cl in unique_clusters:
        mask = cluster_labels == cl
        n = int(mask.sum())
        _, counts = np.unique(injection_labels[mask], return_counts=True)
        purity = float(counts.max()) / n
        weighted_purity += purity * n
        total_pix += n

    return weighted_purity / total_pix if total_pix > 0 else float("nan")
