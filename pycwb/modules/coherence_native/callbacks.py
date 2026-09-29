"""Callable contracts for optional replacements within this scientific module.

These describe individual operations, not a required workflow or stage order.
Prepared inputs are shared read-only; cluster mutations stay job-owned.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import numpy as np
    from pycwb.types.time_frequency_map import TimeFrequencyMap


class PixelSelector(Protocol):
    """Return the native candidate dictionary without mutating prepared maps."""

    def __call__(
        self,
        tf_maps: list[TimeFrequencyMap],
        lag_index: int,
        energy_threshold: float,
        lag_shifts: np.ndarray | list | None = None,
        veto: np.ndarray | None = None,
        edge: float = 0.0,
        selection_cache: dict | None = None,
        preindex_shifts: bool = False,
    ) -> dict: ...
