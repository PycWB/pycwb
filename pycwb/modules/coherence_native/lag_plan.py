"""Small lag-plan adapter for native coherence examples and tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LagPlan:
    """Per-lag detector time shifts in seconds."""

    lag_shifts: np.ndarray

    @property
    def n_lag(self) -> int:
        return int(self.lag_shifts.shape[0])


def build_lag_plan_from_config(config, tf_maps=None) -> LagPlan:
    """Build a lag plan from explicit config shifts, with a zero-lag fallback.

    This helper exists for direct native-coherence examples. Production workflow
    code should prefer ``WaveSegment.lag_shifts`` because it already resolves
    superlags, lag files, and per-job segment timing.
    """
    n_ifo = _infer_n_ifo(config, tf_maps)
    configured = getattr(config, "lag_shifts", None)

    if configured is not None:
        lag_shifts = np.asarray(configured, dtype=float)
    else:
        n_lag = int(getattr(config, "n_lag", getattr(config, "lagSize", 1)) or 1)
        lag_shifts = np.zeros((max(1, n_lag), n_ifo), dtype=float)

    if lag_shifts.ndim == 1:
        lag_shifts = lag_shifts.reshape(1, -1)
    if lag_shifts.ndim != 2:
        raise ValueError("lag_shifts must be a 1D or 2D array")
    if lag_shifts.shape[1] != n_ifo:
        raise ValueError(
            f"lag_shifts detector dimension {lag_shifts.shape[1]} does not match n_ifo={n_ifo}"
        )

    return LagPlan(lag_shifts=np.ascontiguousarray(lag_shifts, dtype=float))


def _infer_n_ifo(config, tf_maps) -> int:
    if tf_maps is not None:
        try:
            return max(1, len(tf_maps))
        except TypeError:
            pass
    if hasattr(config, "nIFO"):
        return max(1, int(config.nIFO))
    if hasattr(config, "ifo"):
        return max(1, len(config.ifo))
    return 1
