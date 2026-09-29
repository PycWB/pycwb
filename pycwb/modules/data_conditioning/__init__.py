"""Native segment-level regression, whitening, and noise-map preparation.

The workflow applies resampling.py before regression and whitening. Condition each detector segment before lag
processing; use signal-only injection whitening with an existing noise map.
The MESA entry point is loaded on demand to keep its dependencies optional.
"""

from .data_conditioning import condition_strain, condition_strains
from .regression import apply_regression, apply_regression_jax
from .whitening import whiten_wavelet
from .injection_whitening import whiten_injection_strain
from .psd_correction import apply_psd_correction


def __getattr__(name):
    if name == "whiten_mesa":
        from .whitening_mesa import whiten_mesa

        return whiten_mesa
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "condition_strains",
    "condition_strain",
    "apply_regression",
    "apply_regression_jax",
    "whiten_wavelet",
    "whiten_mesa",
    "whiten_injection_strain",
    "apply_psd_correction",
]
