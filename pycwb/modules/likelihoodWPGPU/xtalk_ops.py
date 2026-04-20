"""
Packet norm and signal norm — re-exported from CPU module.

These functions involve sparse xtalk neighbor lookups that are inherently
sequential.  They run once at the best sky direction only (not in the sky scan)
so they are not performance-critical.  We reuse the Numba implementations.
"""

from pycwb.modules.likelihoodWP.utils import (
    packet_norm_numpy,
    gw_norm_numpy,
)

__all__ = ["packet_norm_numpy", "gw_norm_numpy"]
