"""Explicit stage contracts shared by native and accelerated job workflows.

These objects hold callables, not analysis data or device resources. GPU
factories create their resources in the parent or spawned worker that uses
them; stage bundles are never shipped to another process.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PreparationStages:
    """Native-signature data loading, conditioning and per-trial setup."""

    read_from_job_segment: Callable[..., Any]
    condition_strains: Callable[..., Any]
    setup_coherence: Callable[..., Any]
    build_td_inputs_cache: Callable[..., Any]


@dataclass(frozen=True)
class LagStages:
    """Native-signature scientific stages and event factory for one lag."""

    coherence_single_lag: Callable[..., Any]
    supercluster_single_lag: Callable[..., Any]
    evaluate_cluster_likelihood: Callable[..., Any]
    event_factory: Callable[..., Any]
