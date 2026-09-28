"""
pycwb.modules.likelihoodWP — Numba-accelerated coherent likelihood (CPU).

Computes sky localization (full HEALPix scan), coherent SNR, null energy,
correlation, and per-cluster detection statistics using per-pixel
time-delay data and antenna patterns. Uses ``@njit`` + ``prange``
for CPU-bound inner loops.
"""

from .likelihood import (
    prepare_likelihood_inputs,
    evaluate_cluster_likelihood,
    evaluate_fragment_clusters,
)

__all__ = [
    "prepare_likelihood_inputs",
    "evaluate_cluster_likelihood",
    "evaluate_fragment_clusters",
]
