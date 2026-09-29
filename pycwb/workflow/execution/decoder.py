"""Isolate GWF decoder allocations; transport only metadata and array filenames."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from multiprocessing.connection import Connection

    from .planner import FrameRequest
    from .resources import MemoryBudget


import multiprocessing
import os
import traceback
from pathlib import Path

import numpy as np


def write_array(
    reader: Callable[..., Any],
    source: str,
    request: FrameRequest,
    signature: tuple[int, ...],
    filename: str,
) -> int:
    """Validate decoded samples and save a non-pickled immutable cache payload."""
    from .cache import fingerprint, sample_index

    series = reader(source, request.channel, start=request.start, end=request.end)
    data = np.asarray(series.value)
    count = sample_index(request.end - request.start, request.rate)
    if (
        data.ndim != 1
        or data.size != count
        or data.dtype.kind not in "fiu"
        or data.dtype.itemsize > 8
        or data.nbytes > request.estimated_bytes
        or float(series.sample_rate.value) != request.rate
        or float(series.t0.value) != request.start
    ):
        raise ValueError(
            f"Unexpected decoded frame shape/rate/epoch/dtype: {request.path}"
        )
    if fingerprint(source) != signature:
        raise RuntimeError(f"Frame source changed while being read: {request.path}")
    np.save(filename, data, allow_pickle=False)
    return data.nbytes


def _decode(
    connection: Connection,
    source: str,
    request: FrameRequest,
    signature: tuple[int, ...],
    filename: str,
    cpus: Sequence[int] | None,
) -> None:
    try:
        if cpus and hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, cpus)
        from pycwb.modules.read_data.read_data import read_from_gwf

        connection.send(
            (write_array(read_from_gwf, source, request, signature, filename), None)
        )
    except BaseException:  # noqa: BLE001 -- report decoder exit to its supervisor
        connection.send((None, traceback.format_exc()))
    finally:
        connection.close()


def decode_to_file(
    source: str,
    request: FrameRequest,
    signature: tuple[int, ...],
    filename: str,
    budget: MemoryBudget,
    cached_bytes: int,
    cpus: Sequence[int] | None,
) -> int:
    """Check actual memory while a disposable decoder writes a local array."""
    from .resources import available_memory, process_tree_memory

    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(
        target=_decode, args=(child, source, request, signature, filename, cpus)
    )
    try:
        process.start()
        child.close()
        while not parent.poll(0.2):
            _, pss = process_tree_memory()
            if (
                pss + cached_bytes + request.estimated_bytes
                > budget.limit - budget.headroom
                or available_memory() < budget.headroom
            ):
                raise MemoryError(
                    "Frame decoding reached the execution memory safety margin"
                )
            if not process.is_alive():
                # The child can send its result and exit while we inspect the
                # process tree. Drain the pipe before treating its exit as a
                # failure; the poll at the top of this iteration is now stale.
                if parent.poll():
                    break
                raise RuntimeError(f"Frame decoder exited with code {process.exitcode}")
        try:
            size, error = parent.recv()
        except EOFError as exc:
            raise RuntimeError("Frame decoder exited without a result") from exc
        process.join(timeout=5)
        if error or process.exitcode != 0:
            raise RuntimeError(
                f"Frame decoder failed for {source}: {error or process.exitcode}"
            )
        return size
    except BaseException:
        if process.pid is not None:
            process.terminate()
            process.join(timeout=3)
            if process.is_alive():
                process.kill()
                process.join()
        Path(filename).unlink(missing_ok=True)
        raise
    finally:
        parent.close()
        child.close()
