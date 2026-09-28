"""Segment-level regression and whitening for one or several detectors."""

import logging
import time

from .regression import apply_regression
from .whitening import whiten_wavelet

logger = logging.getLogger(__name__)


def _select_whitener(config):
    """Resolve a whitening method without importing optional MESA dependencies early."""
    method = getattr(config, "whiteMethod", "wavelet")
    if method in {"wavelet", "python"}:
        return whiten_wavelet
    if method == "mesa":
        from .whitening_mesa import whiten_mesa

        return whiten_mesa
    raise ValueError(f"Method {method} is not a valid native whitening method")


def condition_strains(config, strains):
    """Regress and whiten a segment's detector strains.

    Returns two tuples: conditioned TimeSeries objects and their NoiseRMSMap
    anchor maps, in detector order. Regressions finish for all detectors before
    whitening starts, preserving the established execution order. Detector
    parallelism belongs to the caller; regression may parallelize its TF layers.
    """
    start = time.perf_counter()
    whiten = _select_whitener(config)
    regressed = [apply_regression(config, strain) for strain in strains]
    results = [whiten(config, strain) for strain in regressed]
    conditioned, noise_maps = zip(*results)
    logger.info("Native data conditioning time: %.2f seconds", time.perf_counter() - start)
    return conditioned, noise_maps


def condition_strain(config, strain):
    """Return (conditioned TimeSeries, NoiseRMSMap) for one detector segment."""
    whiten = _select_whitener(config)
    return whiten(config, apply_regression(config, strain))


__all__ = ["condition_strains", "condition_strain"]
