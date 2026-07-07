"""
pycwb.modules.super_cluster_native — Native multi-resolution super-clustering.

Merges pixel clusters across resolution levels, applies sub-net cuts,
and defragments using Numba-accelerated link-matrix computation.
"""

from .super_cluster import (
    supercluster_wrapper,
    setup_supercluster,
    supercluster_single_lag,
    defragment,
)

__all__ = [
    "supercluster_wrapper",
    "setup_supercluster",
    "supercluster_single_lag",
    "defragment",
]
