"""Opt-in bounded frame reads, followed by unchanged native frame merging.

Runs in the job (parent) process before any lag worker exists. Each frame is
decoded by the unchanged native ``read_single_frame_from_job_segment``; only
the scheduling of the per-frame reads changes, and the merge is the native
``merge_frames`` on the frames in their original order.
"""

from __future__ import annotations

import logging
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from itertools import repeat
from typing import TYPE_CHECKING, Any

from pycwb.modules.read_data.read_data import (
    merge_frames,
    read_single_frame_from_job_segment,
)


if TYPE_CHECKING:
    from pycwb.config import Config
    from pycwb.types.job import WaveSegment

logger = logging.getLogger(__name__)


def _read_frame(config: Config, frame: Any, job_seg: WaveSegment) -> Any:
    """Module-level picklable wrapper so spawned read processes can import it."""
    return read_single_frame_from_job_segment(config, frame, job_seg)


def read_from_job_segment(
    config: Config,
    job_seg: WaveSegment,
    *,
    workers: int = 1,
    processes: bool = False,
    validate: bool = False,
) -> Any:
    """Read and merge every frame of ``job_seg`` with bounded parallelism.

    Parameters
    ----------
    config : Config
        Search configuration; ``segEdge`` is passed to the native merge.
    job_seg : WaveSegment
        Job segment whose ``frames`` are read.

    workers : int
        Maximum concurrent reads or detector tasks; positive, defaults to one.
    processes : bool
        Spawn independent frame readers instead of threads.
    validate : bool
        Also run the serial reference and compare exact outputs.

    Returns
    -------
    object
        The native ``merge_frames`` result (per-detector merged data).

    Raises
    ------
    ValueError
        If workers is not a positive integer.
    AssertionError
        If validation is enabled and the parallel result differs from the
        serial reads.
    """
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")

    def read(frame: Any) -> Any:
        return read_single_frame_from_job_segment(config, frame, job_seg)

    start = time.perf_counter()
    if processes:
        # Spawn avoids inheriting CUDA/JAX state. Native frame decoding is
        # unchanged; each child owns its independent reader and file handle.
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            frames = list(pool.map(_read_frame, repeat(config), job_seg.frames, repeat(job_seg)))
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            frames = list(pool.map(read, job_seg.frames))
    actual = merge_frames(job_seg, frames, config.segEdge)
    logger.info(
        "Parallel frame read: workers=%d processes=%s seconds=%.6f",
        workers,
        processes,
        time.perf_counter() - start,
    )
    if validate:
        from pycwb.utils.fingerprint import leaves

        expected = merge_frames(job_seg, [read(frame) for frame in job_seg.frames], config.segEdge)
        if leaves(expected) != leaves(actual):
            raise AssertionError("Parallel frame input differs from serial reads")
        logger.info("Frame read parity: all detector data and coordinates bitwise exact")
    return actual
