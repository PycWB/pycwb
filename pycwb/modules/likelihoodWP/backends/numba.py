"""Declarative selection of the existing Numba likelihood kernels."""

from .base import LikelihoodKernels
from ..dpf import calculate_dpf
from ..sky_scan import scan_sky_for_best_fit
from ..sky_statistics import compute_statistics_at_sky_position


KERNELS = LikelihoodKernels(
    name="numba",
    calculate_dpf_regulator=calculate_dpf,
    scan_sky=scan_sky_for_best_fit,
    statistics_at_best_fit=compute_statistics_at_sky_position,
)

__all__ = ["KERNELS"]
