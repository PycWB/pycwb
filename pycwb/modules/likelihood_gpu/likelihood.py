"""Compose the release likelihood API with selected CUDA implementations."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import TYPE_CHECKING
from pycwb.constants.gpu_options import gpu_options
from pycwb.modules.likelihoodWP.likelihood import prepare_likelihood_inputs
from functools import partial

# Import the module explicitly: package exports may share its name.
native = importlib.import_module("pycwb.modules.likelihoodWP.likelihood")
__all__ = ["build_likelihood", "prepare_likelihood_inputs"]


if TYPE_CHECKING:
    from pycwb.types.network_cluster import Cluster
    from pycwb.modules.likelihoodWP.results import SkyMapStatistics


def build_likelihood(config: object | None = None) -> Callable[..., tuple[Cluster | None, SkyMapStatistics | None]]:
    """Build process-owned callbacks; native orchestration owns scientific policy."""
    from pycwb.config.validation import validate_runtime_settings

    validate_runtime_settings(config)
    options = gpu_options(config)
    callbacks = {}
    if options.dpf:
        from .dpf_regulator import DPFRegulator

        callbacks["scalar_regulator"] = DPFRegulator(options)
    if options.likelihood:
        from .likelihood_scan import LikelihoodScan

        callbacks["sky_scan"] = LikelihoodScan(options).scan_sky
    if options.chirp:
        from .chirp_bootstrap import make_chirp_update

        callbacks["chirp_update"] = make_chirp_update()
    return (
        partial(native.evaluate_cluster_likelihood, **callbacks)
        if callbacks
        else native.evaluate_cluster_likelihood
    )
