"""Shared coherent likelihood with selectable Numba and JAX backends.

Both backends use one orchestration for sky masks, cuts, reconstruction,
uncertainty features, metadata, and acceptance. Select numerical kernels with
the flat ``likelihood_backend`` configuration parameter.
"""

from .likelihood import (
    setup_likelihood,
    likelihood,
    likelihood_wrapper,
    prepare_likelihood_inputs,
    evaluate_cluster_likelihood,
    evaluate_fragment_clusters,
)

__all__ = [
    "setup_likelihood",
    "likelihood",
    "likelihood_wrapper",
    "prepare_likelihood_inputs",
    "evaluate_cluster_likelihood",
    "evaluate_fragment_clusters",
]
