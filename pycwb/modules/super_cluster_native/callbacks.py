"""Callable contracts for optional replacements within this scientific module.

These describe individual operations, not a required workflow or stage order.
Prepared inputs are shared read-only; cluster mutations stay job-owned.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from pycwb.types.network_cluster import Cluster


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
