"""Conservative admission accounting; estimates are not an OS memory limiter."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .settings import ExecutionSettings


import os
import threading
from dataclasses import dataclass
from pathlib import Path

import psutil



def cgroup_headroom(
    root: Path = Path("/sys/fs/cgroup"), membership: Path = Path("/proc/self/cgroup")
) -> int | None:
    """Minimum available bytes in visible cgroup-v2 ancestors, or None.

    Walking ancestors matters when a parent slice, rather than the leaf, holds
    the allocation limit. Host availability remains the fallback on other hosts.
    """
    try:
        relative = next(
            line[3:]
            for line in membership.read_text().splitlines()
            if line.startswith("0::")
        )
        leaf = root / relative.lstrip("/")
        # A cgroup namespace can expose its own leaf as the mount root.
        if not leaf.is_dir():
            leaf = root
        limits = []
        for directory in (leaf, *leaf.parents):
            if directory != root and root not in directory.parents:
                break
            try:
                maximum = (directory / "memory.max").read_text().strip()
                if maximum != "max":
                    current = int((directory / "memory.current").read_text())
                    limits.append(max(0, int(maximum) - current))
            except (OSError, ValueError):
                continue
        return min(limits) if limits else None
    except (OSError, StopIteration):
        return None


def available_memory() -> int:
    """Return available host/cgroup bytes, whichever is smaller."""
    available = int(psutil.virtual_memory().available)
    cgroup = cgroup_headroom()
    return min(available, cgroup) if cgroup is not None else available


def available_cpus() -> list[int]:
    """Return affinity CPUs clipped by a declared scheduler allocation."""
    try:
        cpus = sorted(os.sched_getaffinity(0))
    except AttributeError:
        cpus = list(range(os.cpu_count() or 1))
    allocation = os.environ.get("SLURM_CPUS_PER_TASK")
    if allocation is not None:
        count = int(allocation)
        if count < 1:
            raise ValueError("Allocated CPU count must be positive")
        cpus = cpus[:count]
    return cpus


@dataclass(frozen=True)
class MemoryBudget:
    """Allocation ceiling and conservative worker/cache reservations, in bytes."""

    limit: int
    baseline: int
    headroom: int
    worker: int
    cache: int
    workers: int

    @classmethod
    def resolve(
        cls, settings: ExecutionSettings, requested_workers: int, input_peak: int = 0
    ) -> MemoryBudget:
        """Admit workers before assigning spare capacity to the raw cache."""
        if type(requested_workers) is not int or requested_workers < 1:
            raise ValueError("Number of execution workers must be positive")
        baseline = psutil.Process().memory_info().rss
        available = available_memory()
        limit = min(settings.memory_limit or available + baseline, available + baseline)
        # Two serialized-message buffers per worker may coexist with analysis.
        worker = max(settings.worker_memory, input_peak) + 2 * settings.message_limit
        usable = limit - baseline - settings.headroom
        workers = min(requested_workers, max(0, usable // worker))
        if workers < 1:
            raise MemoryError(
                f"Execution budget {limit} bytes cannot admit one worker: "
                f"baseline={baseline}, headroom={settings.headroom}, worker reservation={worker}. "
                "Increase memory_limit or supply a measured worker_memory estimate."
            )
        # During cache creation both a decoded buffer and file-backed pages can
        # exist. Keep a further payload allowance for decoder/conversion scratch.
        cache = min(settings.cache_limit, max(0, (usable - workers * worker) // 3))
        if settings.preload == "off":
            cache = 0
        return cls(limit, baseline, settings.headroom, worker, cache, workers)

    def can_decode(
        self, cached_bytes: int, new_bytes: int, decode_bytes: int | None = None
    ) -> bool:
        """Check payload, full-frame scratch and current-memory reservations."""
        decode_bytes = max(new_bytes, decode_bytes or 0)
        reserved = self.baseline + self.headroom + self.workers * self.worker
        return (
            cached_bytes + new_bytes <= self.cache
            and reserved + cached_bytes + new_bytes + 2 * decode_bytes <= self.limit
            and available_memory() >= self.headroom + new_bytes + 2 * decode_bytes
        )


def process_tree_memory() -> tuple[int, int]:
    """PSS avoids counting shared maps once per worker; RSS is diagnostic only."""
    root = psutil.Process()
    rss = pss = 0
    for process in [root, *root.children(recursive=True)]:
        try:
            info = process.memory_full_info()
            rss += info.rss
            pss += getattr(info, "pss", info.rss)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return rss, pss


class MemoryMonitor:
    """Sample during synchronous frame decoding as well as worker execution."""

    def __init__(
        self,
        cached_bytes: Callable[[], int],
        interval: float = 0.2,
        limit: int | None = None,
        on_pressure: Callable[[], None] | None = None,
        headroom: int = 0,
    ) -> None:
        self.cached_bytes = cached_bytes
        self.interval = interval
        self.limit = limit
        self.headroom = headroom
        self.on_pressure = on_pressure
        self.exceeded = False
        self.peak_rss = 0
        self.peak_pss = 0
        self.peak_accounted = 0
        self.min_available = available_memory()
        self.initial_swap = psutil.swap_memory().used
        self.peak_swap = self.initial_swap
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="execution-memory", daemon=True
        )

    def start(self) -> None:
        """Begin periodic sampling in a supervisor-owned thread."""
        self._thread.start()

    def close(self) -> None:
        """Stop sampling and wait for the monitor thread."""
        self._stop.set()
        self._thread.join()

    def _run(self) -> None:
        while not self._stop.is_set():
            rss, pss = process_tree_memory()
            self.peak_rss = max(self.peak_rss, rss)
            self.peak_pss = max(self.peak_pss, pss)
            # PSS can include mapped cache pages; adding logical cache bytes is
            # intentionally conservative and also covers currently unmapped pages.
            self.peak_accounted = max(self.peak_accounted, pss + self.cached_bytes())
            available = available_memory()
            self.min_available = min(self.min_available, available)
            self.peak_swap = max(self.peak_swap, psutil.swap_memory().used)
            if (
                (self.limit is not None and self.peak_accounted > self.limit)
                or available < self.headroom
            ) and not self.exceeded:
                self.exceeded = True
                if self.on_pressure is not None:
                    self.on_pressure()
            self._stop.wait(self.interval)
