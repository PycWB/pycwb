"""
pycwb.modules.coherence_native — Native coherence engine.

Production coherence pipeline using JAX-accelerated WDM time→frequency
transforms, max-energy computation, threshold-based pixel selection,
veto application, and single-resolution pixel clustering. Uses lag shifts
provided by the job segment.
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

__all__ = [
    "coherence",
    "setup_coherence",
    "coherence_single_lag",
    "max_energy",
    "compute_threshold",
    "apply_veto",
    "select_network_pixels",
    "cluster_pixels",
]
