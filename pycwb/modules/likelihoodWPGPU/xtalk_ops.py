"""
Packet norm and signal norm — re-exported from CPU module.

These functions involve sparse xtalk neighbor lookups that are inherently
sequential.  They run once at the best sky direction only (not in the sky scan)
so they are not performance-critical.  We reuse the Numba implementations.
"""

from pycwb.modules.likelihoodWP.packet_ops import (
    compute_packet_norms,
    compute_signal_norms,
)

__all__ = ["compute_packet_norms", "compute_signal_norms"]
