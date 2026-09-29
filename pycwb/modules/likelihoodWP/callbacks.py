"""Callable contracts for optional replacements within this scientific module.

These describe individual operations, not a required workflow or stage order.
Prepared inputs are shared read-only; cluster mutations stay job-owned.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import numpy as np
    from pycwb.types.network_cluster import Cluster
    from pycwb.config import Config


class ScalarRegulator(Protocol):
    """Compute the host scalar regulator from FP32 detector/sky arrays."""

    def __call__(
        self,
        FP: np.ndarray,
        FX: np.ndarray,
        rms: np.ndarray,
        n_sky: int,
        n_ifo: int,
        gamma_regulator: float,
        network_energy_threshold: float,
        sky_valid_indices: np.ndarray,
    ) -> float: ...


class SkyScanner(Protocol):
    """Score valid skies with native ordering and last-maximum tie breaking."""

    def __call__(
        self,
        geometry: tuple[np.ndarray, np.ndarray, np.ndarray],
        cluster: tuple[np.ndarray, np.ndarray, np.ndarray],
        settings: tuple[np.ndarray, float, float, float, np.ndarray],
        *,
        reuse_delays: bool = True,
        setup: dict | None = None,
        big_cluster: bool = False,
    ) -> tuple: ...


class ChirpUpdater(Protocol):
    """Mutate only cluster chirp metadata, retaining native resets and fallbacks."""

    def __call__(
        self,
        cluster: Cluster,
        config: Config,
        *,
        xgb_rho_mode: bool,
        chirp_seed: int,
        use_native_chirp: bool,
    ) -> None: ...
