"""
Compatibility module for pure-Python data conditioning.

Implementations have been split into:
- regression_py.py
- whitening_py.py

Existing imports from data_conditioning_python are preserved.
"""

from .regression_py import (
    regression_python,
    _cwb_percentile_mean,
    _cwb_rotated_products,
    _build_correlation_matrix,
    _build_crosscorr_vector,
)
from .whitening_py import (
    whitening_python,
    _estimate_noise_rms,
    _estimate_noise_rms_cwb,
    _bandpass_rms,
    _whiten_coefficients,
    _apply_wiener_filter,
    _average_phases,
)

__all__ = [
    "regression_python",
    "whitening_python",
    "_cwb_percentile_mean",
    "_cwb_rotated_products",
    "_build_correlation_matrix",
    "_build_crosscorr_vector",
    "_estimate_noise_rms",
    "_estimate_noise_rms_cwb",
    "_bandpass_rms",
    "_whiten_coefficients",
    "_apply_wiener_filter",
    "_average_phases",
]
