"""Supervisor-owned bounded frame cache, shared through read-only array files."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gwpy.timeseries import TimeSeries

    from .resources import MemoryBudget


import math
import os
import tempfile
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .planner import FrameRequest


def local_source(path: str) -> str:
    """Use the existing transfer convention, also for staged absolute paths."""
    if os.path.isfile(path):
        return os.path.realpath(path)
    candidate = os.path.basename(path)
    if os.path.isfile(candidate):
        return os.path.realpath(candidate)
    raise FileNotFoundError(f"Frame file not found: {path}")


def fingerprint(path: str) -> tuple[int, int, int, int]:
    """Return the run-scoped stat identity of a local source."""
    stat = os.stat(path)
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def sample_index(seconds: float, rate: float) -> int:
    """Convert an aligned interval to samples with GPS-rounding tolerance."""
    offset = seconds * rate
    index = round(offset)
    # GPS subtraction can lose a few ulps. Never silently crop a partial sample.
    if not math.isclose(index, offset, rel_tol=0, abs_tol=0.01):
        raise ValueError(
            f"Frame window is not aligned to samples: {seconds}s at {rate}Hz"
        )
    return index


@dataclass(frozen=True)
class CachedFrame:
    """Immutable mapping descriptor for an unchanged physical source interval."""

    request: FrameRequest
    filename: str
    source: str
    signature: tuple[int, ...]
    nbytes: int

    def covers(self, request: FrameRequest) -> bool:
        """Check channel/rate identity and complete interval coverage."""
        return (
            self.request.key == request.key
            and self.request.start <= request.start
            and self.request.end >= request.end
        )


@dataclass(frozen=True)
class FrameProvider:
    """Picklable descriptors only. Returned arrays are owned by the consumer."""

    entries: tuple[CachedFrame, ...]
    metrics: dict[str, int] = field(
        default_factory=lambda: {
            "direct_reads": 0,
            "direct_bytes": 0,
            "cached_reads": 0,
        }
    )

    def read(self, filename: str, channel: str, start: float, end: float) -> TimeSeries:
        """Return owned raw samples from a mapping or the original GWF reader."""
        from gwpy.timeseries import TimeSeries

        from pycwb.modules.read_data.read_data import read_from_gwf

        for entry in self.entries:
            request = entry.request
            if (
                request.path != filename
                or request.channel != channel
                or not (request.start <= start < end <= request.end)
            ):
                continue
            if (
                local_source(filename) != entry.source
                or fingerprint(entry.source) != entry.signature
            ):
                raise RuntimeError(f"Frame source changed during execution: {filename}")
            first = sample_index(start - request.start, request.rate)
            last = sample_index(end - request.start, request.rate)
            array = np.load(entry.filename, mmap_mode="r", allow_pickle=False)
            try:
                data = np.array(array[first:last], copy=True)
            finally:
                array._mmap.close()
            self.metrics["cached_reads"] += 1
            return TimeSeries(
                data, t0=start, sample_rate=request.rate, channel=channel, copy=False
            )
        data = read_from_gwf(local_source(filename), channel, start=start, end=end)
        self.metrics["direct_reads"] += 1
        self.metrics["direct_bytes"] += data.value.nbytes
        return data


def merged_requests(
    groups: Iterable[Iterable[FrameRequest]],
) -> tuple[FrameRequest, ...]:
    """Coalesce overlapping/adjacent intervals only, never bridge large gaps."""
    by_key: dict[tuple[str, str, float], list[FrameRequest]] = {}
    for group in groups:
        for request in group:
            by_key.setdefault(request.key, []).append(request)
    result = []
    for requests in by_key.values():
        merged: list[FrameRequest] = []
        for request in sorted(requests, key=lambda r: (r.start, r.end)):
            if merged and request.start <= merged[-1].end:
                previous = merged.pop()
                request = FrameRequest(
                    request.path,
                    request.channel,
                    previous.start,
                    max(previous.end, request.end),
                    request.rate,
                    max(previous.source_bytes, request.source_bytes),
                )
            merged.append(request)
        result.extend(merged)
    return tuple(result)


class FrameCache:
    """Synchronous single-flight loading; live workers pin their input files.

    Logical cached bytes bound disk payload AND conservatively reserve resident
    pages. Source GWF files are never copied into this directory.
    """

    def __init__(
        self,
        directory: str | Path,
        budget: MemoryBudget,
        reader: Callable[..., Any] | None = None,
        decoder_cpus: Sequence[int] | None = None,
        max_entries: int = 256,
    ) -> None:
        self.budget = budget
        self.reader = reader
        self.decoder_cpus = decoder_cpus
        self.max_entries = max_entries
        self._temporary = tempfile.TemporaryDirectory(
            prefix=".frame-cache-", dir=directory
        )
        self.entries: OrderedDict[str, CachedFrame] = OrderedDict()
        self.pins: dict[str, int] = {}
        self.sources: dict[str, tuple[str, tuple[int, ...]]] = {}
        self.bytes = 0
        self.metrics = {
            "reads": 0,
            "hits": 0,
            "hit_bytes": 0,
            "decoded_bytes": 0,
            "evictions": 0,
            "bypasses": 0,
            "peak_cache_bytes": 0,
        }

    def close(self) -> None:
        """Remove temporary storage after all consumer leases are released."""
        if any(self.pins.values()):
            raise RuntimeError("Cannot close cache while workers still hold leases")
        self.entries.clear()
        self._temporary.cleanup()

    def _evict(self, key: str) -> None:
        entry = self.entries.pop(key)
        Path(entry.filename).unlink()
        self.bytes -= entry.nbytes
        self.pins.pop(key, None)
        self.metrics["evictions"] += 1

    def evict_idle(self) -> None:
        """Release every unpinned entry under memory pressure."""
        for key in list(self.entries):
            if not self.pins.get(key, 0):
                self._evict(key)

    def acquire(
        self, requests: Iterable[FrameRequest], planned: Sequence[FrameRequest] = ()
    ) -> FrameProvider:
        """Pin available inputs, loading bounded planned unions where possible."""
        entries = []
        try:
            for request in requests:
                entry = self._get(request, planned)
                if entry is not None:
                    self.pins[entry.filename] = self.pins.get(entry.filename, 0) + 1
                    entries.append(entry)
            return FrameProvider(tuple(entries))
        except BaseException:
            self.release(FrameProvider(tuple(entries)))
            raise

    def release(self, provider: FrameProvider) -> None:
        """Release one lease for each mapping given to a consumer."""
        for entry in provider.entries:
            self.pins[entry.filename] -= 1

    def _get(
        self, request: FrameRequest, planned: Sequence[FrameRequest]
    ) -> CachedFrame | None:
        source = local_source(request.path)
        signature = fingerprint(source)
        identity = (source, signature)
        previous = self.sources.setdefault(request.path, identity)
        if previous != identity:
            raise RuntimeError(f"Frame source changed during execution: {request.path}")
        for key, entry in list(self.entries.items()):
            if entry.covers(request):
                if entry.signature != signature:
                    raise RuntimeError(
                        f"Frame source changed during execution: {request.path}"
                    )
                self.entries.move_to_end(key)
                self.metrics["hits"] += 1
                self.metrics["hit_bytes"] += request.estimated_bytes
                return entry
        target = next(
            (
                r
                for r in planned
                if r.key == request.key
                and r.start <= request.start
                and r.end >= request.end
            ),
            request,
        )
        if target.estimated_bytes > self.budget.cache:
            target = request
        size = target.estimated_bytes
        while (
            self.bytes + size > self.budget.cache
            or len(self.entries) >= self.max_entries
        ):
            victim = next(
                (key for key in self.entries if not self.pins.get(key, 0)), None
            )
            if victim is None:
                break
            self._evict(victim)
        if len(self.entries) >= self.max_entries or not self.budget.can_decode(
            self.bytes, size, target.decode_bytes
        ):
            self.metrics["bypasses"] += 1
            return None
        filename = str(Path(self._temporary.name) / f"{self.metrics['reads']}.npy")
        from .decoder import decode_to_file, write_array

        if self.reader is None:
            size = decode_to_file(
                source,
                target,
                signature,
                filename,
                self.budget,
                self.bytes,
                self.decoder_cpus,
            )
        else:
            size = write_array(self.reader, source, target, signature, filename)
        entry = CachedFrame(target, filename, source, signature, size)
        self.entries[filename] = entry
        self.bytes += size
        self.metrics["reads"] += 1
        self.metrics["decoded_bytes"] += size
        self.metrics["peak_cache_bytes"] = max(
            self.metrics["peak_cache_bytes"], self.bytes
        )
        return entry
