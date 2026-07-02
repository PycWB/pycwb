"""Stable facade for native coherence.

The implementation is split across focused sibling modules.  Keep importing
public entry points from this module for workflow and notebook compatibility.
"""

from __future__ import annotations

from .clustering import cluster_pixels as cluster_pixels
from .pipeline import coherence as coherence
from .pipeline import coherence_single_lag as coherence_single_lag
from .projection import (
    _auto_max_energy_backend_for_layers as _auto_max_energy_backend_for_layers,
    _max_energy_backend as _max_energy_backend,
    _max_energy_backend_label as _max_energy_backend_label,
    _normalize_max_energy_backend as _normalize_max_energy_backend,
    max_energy as max_energy,
)
from .selection import (
    _build_selection_cache as _build_selection_cache,
    _shift_bins_from_lag_shifts as _shift_bins_from_lag_shifts,
    select_network_pixels as select_network_pixels,
)
from .setup import (
    _coherence_timing_enabled as _coherence_timing_enabled,
    _setup_coherence_single_res as _setup_coherence_single_res,
    setup_coherence as setup_coherence,
)
from .veto_threshold import (
    _get_tf_energy_array as _get_tf_energy_array,
    _igamma_inv_upper as _igamma_inv_upper,
    apply_veto as apply_veto,
    build_veto_mask as build_veto_mask,
    compute_threshold as compute_threshold,
)

__all__ = [
    "coherence",
    "setup_coherence",
    "coherence_single_lag",
    "max_energy",
    "compute_threshold",
    "apply_veto",
    "build_veto_mask",
    "select_network_pixels",
    "cluster_pixels",
]
