"""Direct numerical-kernel selection for likelihoodWP."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class SkyGrid:
    """The active full- or coarse-resolution sky grid for one cluster."""

    delays: np.ndarray
    plus_patterns: np.ndarray
    cross_patterns: np.ndarray
    valid_indices: np.ndarray
    phi_geo: np.ndarray
    latitude: np.ndarray
    healpix_order: int | None

    @property
    def size(self) -> int:
        return int(self.delays.shape[1])


@dataclass(frozen=True)
class LikelihoodKernels:
    """Selected array-level functions; calling a field invokes the real kernel."""

    name: str
    calculate_dpf_regulator: Callable
    scan_sky: Callable
    statistics_at_best_fit: Callable


_ALIASES = {
    "cpu": "numba",
    "numba": "numba",
    "gpu": "jax",
    "jax": "jax",
}


def normalize_likelihood_backend(backend: str | None) -> str:
    """Validate a user-facing likelihood backend name."""

    name = str(backend or "numba").strip().lower()
    try:
        return _ALIASES[name]
    except KeyError as exc:
        allowed = "'numba' (or 'cpu') and 'jax' (or 'gpu')"
        raise ValueError(
            f"likelihood_backend must be one of {allowed}; got {backend!r}"
        ) from exc


def get_likelihood_backend(
    backend: str | LikelihoodKernels | None,
) -> LikelihoodKernels:
    """Resolve direct kernel functions without loading unselected modules."""

    if backend is not None and not isinstance(backend, str):
        if not isinstance(backend, LikelihoodKernels):
            raise TypeError(
                "backend must be a backend name or LikelihoodKernels instance"
            )
        return backend

    name = normalize_likelihood_backend(backend)
    module = import_module(f"{__package__}.{name}")
    return module.KERNELS
