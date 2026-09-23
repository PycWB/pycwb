"""Deterministic, metadata-only scheduling. Job IDs and science windows never change."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pycwb.types.job import WaveSegment

    from .settings import ExecutionSettings


import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path

PLAN_VERSION = 1


def json_value(value: object) -> object:
    """Match catalog support for NumPy-valued injection/noise descriptors."""
    import numpy as np

    if isinstance(value, (np.ndarray, np.generic)):
        return value.tolist()
    raise TypeError(f"Unsupported execution metadata: {type(value).__name__}")


@dataclass(frozen=True)
class FrameRequest:
    path: str
    channel: str
    start: float
    end: float
    rate: float
    source_bytes: int = 0

    @property
    def key(self) -> tuple[str, str, float]:
        return self.path, self.channel, self.rate

    @property
    def estimated_bytes(self) -> int:
        # Reader dtype is checked at runtime. float64 is a conservative payload
        # estimate for supported real float32/float64 strain channels.
        return math.ceil((self.end - self.start) * self.rate) * 8

    @property
    def decode_bytes(self) -> int:
        return max(self.estimated_bytes, self.source_bytes)


def frame_requests(job: WaveSegment) -> tuple[FrameRequest, ...]:
    """Describe physical sample windows and conservative full-frame payloads."""
    requests = []
    for frame in job.frames or []:
        i = job.ifos.index(frame.ifo)
        start = max(frame.start_time, job.physical_padded_starts[frame.ifo])
        end = min(frame.end_time, job.physical_padded_ends[frame.ifo])
        if (
            not all(
                math.isfinite(v) for v in (start, end, job.sample_rate, frame.duration)
            )
            or end <= start
            or job.sample_rate <= 0
            or frame.duration <= 0
        ):
            raise ValueError(f"Invalid frame window for job {job.index}: {frame.path}")
        requests.append(
            FrameRequest(
                frame.path,
                job.channels[i],
                start,
                end,
                job.sample_rate,
                math.ceil(frame.duration * job.sample_rate) * 8,
            )
        )
    return tuple(requests)


@dataclass(frozen=True)
class ExecutionPlan:
    # Indices refer to the supplied job list, allowing explicit repeated trials
    # of the same scientific job without inventing new catalog identities.
    batches: tuple[tuple[int, ...], ...]
    requests: tuple[tuple[FrameRequest, ...], ...]
    job_ids: tuple[int, ...]
    version: int = PLAN_VERSION

    @property
    def order(self) -> tuple[int, ...]:
        return tuple(task for batch in self.batches for task in batch)

    def validate(self, jobs: Sequence[WaveSegment]) -> None:
        """Reject incompatible versions, task permutations or input identities."""
        if self.version != PLAN_VERSION:
            raise ValueError("Unsupported execution plan version")
        if sorted(self.order) != list(range(len(jobs))) or any(
            not batch for batch in self.batches
        ):
            raise ValueError("Execution plan must contain each task exactly once")
        if self.job_ids != tuple(job.index for job in jobs):
            raise ValueError("Execution plan job identities differ from requested jobs")
        if self.requests != tuple(frame_requests(job) for job in jobs):
            raise ValueError(
                "Execution plan input requests differ from scientific jobs"
            )

    def document(
        self, jobs: Sequence[WaveSegment], settings: ExecutionSettings
    ) -> dict[str, Any]:
        """Build serializable metadata with an integrity identity."""
        data = {
            "plan": asdict(self),
            "execution": asdict(settings),
            "jobs": [asdict(job) for job in jobs],
        }
        encoded = json.dumps(
            data,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            default=json_value,
        )
        return {"identity": hashlib.sha256(encoded.encode()).hexdigest(), **data}


def write_document(path: str | Path, document: object) -> None:
    """Publish a complete JSON document atomically in its destination directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, delete=False
        ) as stream:
            name = stream.name
            json.dump(
                document,
                stream,
                indent=2,
                sort_keys=True,
                allow_nan=False,
                default=json_value,
            )
            stream.write("\n")
        os.replace(name, path)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)


class SimplePlanner:
    """Preserve input ordering and legacy fixed-count grouping."""

    def plan(
        self, jobs: Sequence[WaveSegment], config: Any, settings: ExecutionSettings
    ) -> ExecutionPlan:
        """Schedule every supplied task once without altering scientific jobs."""
        size = max(1, int(getattr(config, "job_per_worker", 1)))
        return ExecutionPlan(
            tuple(
                tuple(range(i, min(i + size, len(jobs))))
                for i in range(0, len(jobs), size)
            ),
            tuple(frame_requests(job) for job in jobs),
            tuple(job.index for job in jobs),
        )


class SharedFramePlanner:
    """Prefer shared sources while limiting group size and candidate scoring."""

    def plan(
        self, jobs: Sequence[WaveSegment], config: Any, settings: ExecutionSettings
    ) -> ExecutionPlan:
        """Schedule every supplied task once without altering scientific jobs."""
        requests = tuple(frame_requests(job) for job in jobs)
        keys = [{request.key for request in group} for group in requests]
        consumers: dict[tuple[str, str, float], deque[int]] = defaultdict(deque)
        for task, source_keys in enumerate(keys):
            for key in source_keys:
                consumers[key].append(task)
        remaining = set(range(len(jobs)))
        fallback = deque(range(len(jobs)))
        batches = []
        for seed in range(len(jobs)):
            if seed not in remaining:
                continue
            task = seed
            group = []
            available_keys = set()
            candidates: set[int] = set()
            while True:
                remaining.remove(task)
                group.append(task)
                available_keys.update(keys[task])
                for key in keys[task]:
                    queue = consumers[key]
                    while queue and queue[0] not in remaining:
                        queue.popleft()
                    # Bound scoring work even when one file serves millions of
                    # jobs. This is a deterministic heuristic, not global packing.
                    candidates.update(islice(queue, 128))
                candidates.intersection_update(remaining)
                if len(group) >= settings.batch_size:
                    break
                if not candidates:
                    # Fill spare capacity with ordinary input order instead of
                    # creating one scheduler allocation per disjoint segment.
                    while fallback and fallback[0] not in remaining:
                        fallback.popleft()
                    if not fallback:
                        break
                    task = fallback[0]
                    continue

                def score(
                    candidate: int,
                    available_keys: set[tuple[str, str, float]] = available_keys,
                ) -> tuple[float, int, int]:
                    shared = sum(
                        r.estimated_bytes
                        for r in requests[candidate]
                        if r.key in available_keys
                    )
                    added = sum(
                        r.estimated_bytes
                        for r in requests[candidate]
                        if r.key not in available_keys
                    )
                    return shared / max(1, added), shared, -candidate

                task = max(candidates, key=score)
            batches.append(tuple(group))
        return ExecutionPlan(tuple(batches), requests, tuple(job.index for job in jobs))


def create_simple_planner() -> SimplePlanner:
    """Construct the compatibility planner."""
    return SimplePlanner()


def create_shared_frame_planner() -> SharedFramePlanner:
    """Construct the shared-input planner."""
    return SharedFramePlanner()


def prepare_plan(
    jobs: Sequence[WaveSegment], config: Any, settings: ExecutionSettings
) -> ExecutionPlan:
    """Resolve the configured factory and validate its metadata plan."""
    from pycwb.utils.module import import_function

    if settings.planner:
        planner = import_function(settings.planner)()
    else:
        planner = (
            SharedFramePlanner() if settings.profile == "scalable" else SimplePlanner()
        )
    plan = planner.plan(jobs, config, settings)
    if not isinstance(plan, ExecutionPlan):
        raise TypeError("Job planner must return an ExecutionPlan")
    plan.validate(jobs)
    return plan
