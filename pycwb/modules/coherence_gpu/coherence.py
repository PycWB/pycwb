"""Build process-owned GPU coherence using native payloads and YAML options."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import TYPE_CHECKING, Any
from pycwb.constants.gpu_options import gpu_options
from pycwb.modules.coherence_native import selection
from functools import partial
from .selection import GPUSelector
from .setup_parallel import setup_coherence

coherence = importlib.import_module("pycwb.modules.coherence_native.coherence")
__all__ = ["build_coherence", "setup_coherence", "GPUSelector"]


if TYPE_CHECKING:
    from pycwb.types.network_cluster import FragmentCluster


def build_coherence(config: object | None = None) -> tuple[Callable[..., list[FragmentCluster]], GPUSelector]:
    """Return a single-lag callable and its resident-map owner; clear sessions after use."""
    options = gpu_options(config)
    selector = GPUSelector(options)
    select: Callable[..., Any] = selector
    if options.validate_stages:
        from pycwb.modules.stage_validation import paired

        select = paired(selector, selection.select_network_pixels, "selection", options=options)
    return partial(coherence.coherence_single_lag, select_pixels=select), selector
