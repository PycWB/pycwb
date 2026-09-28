"""Scientific stage protocols; bundles are owned by the process that executes them.

Prepared dictionaries retain the native payload schema. Backends may cache
immutable geometry but must not mutate shared inputs. Clustering/likelihood
may mutate the job-owned clusters passed to them. Device resources are never
serialized as part of a stage bundle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol
from collections.abc import Sequence

if TYPE_CHECKING:
    import numpy as np
    from pycwb.types.time_frequency_map import TimeFrequencyMap
    from pycwb.config import Config
    from pycwb.types.job import WaveSegment
    from pycwb.types.time_series import TimeSeries
    from pycwb.types.noise_rms import NoiseRMSMap
    from pycwb.types.network_cluster import Cluster, FragmentCluster
    from pycwb.modules.xtalk.type import XTalk
    from pycwb.modules.likelihoodWP.results import SkyMapStatistics


class Reader(Protocol):
    """Read per-detector strain for a physical segment."""

    def __call__(self, config: Config, job_seg: WaveSegment) -> list[TimeSeries]: ...


class Conditioner(Protocol):
    """Return detector-ordered whitened strains and noise maps."""

    def __call__(
        self, config: Config, strains: list[TimeSeries]
    ) -> tuple[Sequence[TimeSeries], Sequence[NoiseRMSMap]]: ...


class CoherenceSetup(Protocol):
    """Prepare lag-independent maps and geometry for each resolution."""

    def __call__(
        self,
        config: Config,
        strains: list[TimeSeries],
        job_seg: WaveSegment | None = None,
        nRMS: list | None = None,
    ) -> list[dict]: ...


class TimeDelaySetup(Protocol):
    """Build detector delay inputs indexed by wavelet layer tag."""

    def __call__(self, config: Config, strains: list[TimeSeries]) -> dict[int, list]: ...


class CoherenceStage(Protocol):
    """Select and cluster one lag from prepared resolution maps."""

    def __call__(
        self,
        coherence_setups: list[dict],
        lag_idx: int,
        return_rejected: bool = False,
        veto_windows: list[tuple[float, float]] | None = None,
    ) -> list[FragmentCluster]: ...


class SuperclusterStage(Protocol):
    """Merge one lag's fragments and populate accepted cluster delay vectors."""

    def __call__(
        self,
        setup: dict,
        config: Config,
        fragment_clusters_single_lag: list[FragmentCluster],
        lag_idx: int,
        xtalk: XTalk,
        td_inputs_cache: dict,
    ) -> FragmentCluster | None: ...


class LikelihoodStage(Protocol):
    """Evaluate a cluster in place, returning None for rejected candidates."""

    def __call__(
        self,
        nIFO: int,
        cluster: Cluster,
        config: Config,
        MRAcatalog: str | None = None,
        strains: list[TimeSeries] | None = None,
        cluster_id: int | None = None,
        nRMS: list | None = None,
        setup: dict | None = None,
        xtalk: XTalk | None = None,
        supercluster_setup: dict | None = None,
        chirp_seed: int = 1,
    ) -> tuple[Cluster | None, SkyMapStatistics | None]: ...


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


class TimeDelayPopulator(Protocol):
    """Populate job-owned clusters from immutable prepared delay inputs."""

    def __call__(
        self,
        all_clusters: list[Cluster],
        n_ifo: int,
        K: int,
        td_inputs_cache: dict,
        delay_stride: int = 1,
    ) -> None: ...
