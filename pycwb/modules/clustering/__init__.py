"""
Replaceable pixel-clustering backends for pycWB.

Entry points
------------
:func:`cluster_lag_candidates` (recommended)
    Replacement architecture: accepts raw pixel-candidate dicts from
    :func:`~pycwb.modules.cwb_coherence.coherence.select_pixels_single_lag`,
    runs the named clustering backend, and returns a single
    likelihood-ready :class:`~pycwb.types.network_cluster.FragmentCluster`.
    Replaces the
    ``coherence_single_lag → cluster_fragment_clusters → supercluster_single_lag``
    triple.

:func:`cluster_fragment_clusters` (legacy)
    Insert-between architecture: accepts and returns the same
    ``list[FragmentCluster]`` produced by :func:`coherence_single_lag`, so
    it can be inserted between the coherence and supercluster stages.

Available methods
-----------------
connected_components
    Re-runs the same WDM connected-component algorithm used inside
    :func:`coherence_single_lag`.  Produces identical results to the
    native pipeline; use to verify new plumbing without changing science.
weighted_graph
    Physics-informed re-clustering using a weighted adjacency graph with
    TF-proximity and energy-balance edge weights.
dbscan
    Density-based clustering via :class:`sklearn.cluster.DBSCAN`.
hdbscan
    Hierarchical density-based clustering via
    :class:`sklearn.cluster.HDBSCAN` (scikit-learn ≥ 1.3).
optics
    Ordering-based density clustering via :class:`sklearn.cluster.OPTICS`.
mra_weighted_graph
    Additive multi-resolution weighted graph.  Pools selected pixels from all
    WDM resolutions before clustering; existing non-MRA methods stay unchanged.
mra_hdbscan
    Additive multi-resolution HDBSCAN.  Pools selected pixels from all WDM
    resolutions and clusters them in a scaled physical feature space.
"""

from pycwb.modules.clustering.cluster import cluster_fragment_clusters
from pycwb.modules.clustering.entry_point import cluster_lag_candidates

__all__ = ["cluster_fragment_clusters", "cluster_lag_candidates"]
