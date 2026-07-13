"""Deprecated compatibility import for ROOT injection-strain whitening."""

import warnings

warnings.warn(
    "pycwb.modules.data_conditioning_root.whitening_mdc is deprecated and will "
    "be removed after one release; use "
    "pycwb.modules.data_conditioning_root.injection_whitening instead",
    DeprecationWarning,
    stacklevel=2,
)

from .injection_whitening import whiten_injection_strain

whitening_mdc = whiten_injection_strain

__all__ = ["whitening_mdc"]
