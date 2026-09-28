"""Explicit stage contracts shared by native and accelerated job workflows.

These objects hold callables, not analysis data or device resources. GPU
factories create their resources in the parent or spawned worker that uses
them; stage bundles are never shipped to another process.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING
from pycwb.types.stages import (
    Reader,
    Conditioner,
    CoherenceSetup,
    TimeDelaySetup,
    CoherenceStage,
    SuperclusterStage,
    LikelihoodStage,
)

if TYPE_CHECKING:
    from pycwb.types.network_event import Event


@dataclass(frozen=True)
class PreparationStages:
    """Native-signature data loading, conditioning and per-trial setup."""

    read_from_job_segment: Reader
    condition_strains: Conditioner
    setup_coherence: CoherenceSetup
    build_td_inputs_cache: TimeDelaySetup


@dataclass(frozen=True)
class LagStages:
    """Native-signature scientific stages and event factory for one lag."""

    coherence_single_lag: CoherenceStage
    supercluster_single_lag: SuperclusterStage
    evaluate_cluster_likelihood: LikelihoodStage
    event_factory: Callable[[], "Event"]
