"""Stable facade for WDM time-delay max-energy backends.

The JAX and Numba implementations live in sibling modules.  Keep importing the
public functions from this module for workflow and notebook compatibility.
"""

from __future__ import annotations

from .time_delay_jax import (
    _HAS_JAX as _HAS_JAX,
    _JAX_IMPORT_ERROR as _JAX_IMPORT_ERROR,
    _time_delay_max_energy_complex_jit as _time_delay_max_energy_complex_jit,
    _time_delay_max_energy_pattern_jit as _time_delay_max_energy_pattern_jit,
    _time_delay_max_energy_phase_jit as _time_delay_max_energy_phase_jit,
    _w2t_data_jax as _w2t_data_jax,
    _wdm_packet_energy_jax as _wdm_packet_energy_jax,
    time_delay_max_energy as time_delay_max_energy,
)
from .time_delay_numba import (
    _HAS_NUMBA as _HAS_NUMBA,
    _NUMBA_IMPORT_ERROR as _NUMBA_IMPORT_ERROR,
    _normalize_numba_max_energy_mode as _normalize_numba_max_energy_mode,
    _time_delay_max_energy_pattern_nb as _time_delay_max_energy_pattern_nb,
    _wdm_packet_energy_nb as _wdm_packet_energy_nb,
    time_delay_max_energy_numba as time_delay_max_energy_numba,
)
from .time_delay_packet import (
    _compute_packet_energy_params as _compute_packet_energy_params,
)

__all__ = ["time_delay_max_energy", "time_delay_max_energy_numba"]
