"""
pycwb.modules.coherence_native — Native coherence engine.

Production coherence pipeline using JAX-accelerated WDM time→frequency
transforms, max-energy computation, threshold-based pixel selection,
veto application, and single-resolution pixel clustering. Builds lag
plans from configuration.
"""

from .coherence import (
    coherence,
    setup_coherence,
    coherence_single_lag,
    max_energy,
    compute_threshold,
    apply_veto,
    select_network_pixels,
    cluster_pixels,
)
from .lag_plan import LagPlan, build_lag_plan_from_config

__all__ = [
    "coherence",
    "setup_coherence",
    "coherence_single_lag",
    "max_energy",
    "compute_threshold",
    "apply_veto",
    "select_network_pixels",
    "cluster_pixels",
    "LagPlan",
    "build_lag_plan_from_config",
]
