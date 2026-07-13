"""
pycwb.modules.super_cluster_root — ROOT-backed super-clustering.

Wraps cWB's ``netcluster::supercluster`` via ROOT bindings. Handles
sparse table creation, sky-map resolution, sub-net cuts, and
defragmentation.

.. note::
   This module depends on ROOT and is part of the legacy layer being phased out.
"""

from .super_cluster import supercluster

__all__ = ["supercluster"]
