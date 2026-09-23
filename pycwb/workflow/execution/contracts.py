"""Version-one extension contracts for factories configured in ``execution``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from pycwb.types.job import WaveSegment

    from .executor import ExecutionContext
    from .planner import ExecutionPlan
    from .settings import ExecutionSettings


class JobPlanner(Protocol):
    """Metadata-only scheduling extension returned by a planner factory."""

    def plan(
        self, jobs: list[WaveSegment], config: Any, settings: ExecutionSettings
    ) -> ExecutionPlan:
        """Return metadata scheduling each supplied task exactly once."""
        ...


class JobExecutor(Protocol):
    """Allocation lifetime extension returned by an executor factory."""

    def execute(self, plan: ExecutionPlan, context: ExecutionContext) -> Any:
        """Own worker/output lifetime until success or a propagated exception."""
        ...


class InputProvider(Protocol):
    """Optional source of raw physical samples for a segment processor."""

    def read(self, filename: str, channel: str, start: float, end: float) -> Any:
        """Return an owned, mutable GWPy TimeSeries for this physical interval."""
        ...
