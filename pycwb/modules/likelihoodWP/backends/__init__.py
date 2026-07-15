"""Numerical backends for the shared likelihoodWP orchestration.

Only backend-independent types and the lazy resolver are imported here.  In
particular, importing this package does not load the JAX likelihood kernels;
they are imported only when the JAX backend is selected.
"""

from .base import (
    LikelihoodKernels,
    SkyGrid,
    get_likelihood_backend,
    normalize_likelihood_backend,
)

__all__ = [
    "LikelihoodKernels",
    "SkyGrid",
    "get_likelihood_backend",
    "normalize_likelihood_backend",
]
