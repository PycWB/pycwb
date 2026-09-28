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

from pycwb.constants.gpu_options import gpu_options

if TYPE_CHECKING:
    from pycwb.config import Config
    from pycwb.types.job import WaveSegment

logger = logging.getLogger(__name__)

MAX_READ_WORKERS = 2
"""Largest ``gpu.read_workers``; frame files are read from shared storage where more did not help."""


def _read_frame(config: Config, frame: Any, job_seg: WaveSegment) -> Any:
    """Module-level picklable wrapper so spawned read processes can import it."""
    return read_single_frame_from_job_segment(config, frame, job_seg)


def read_from_job_segment(config: Config, job_seg: WaveSegment) -> Any:
    """Read and merge every frame of ``job_seg`` with bounded parallelism.

    Switches, read at call time:

    * ``gpu.read_workers`` (default 1, at most ``MAX_READ_WORKERS``):
      concurrent frame reads.
    * ``gpu.read_processes=true``: use spawned processes instead of threads
      so no CUDA/JAX state is inherited by the readers.
    * ``gpu.validate_read=true``: re-read serially and require the merged
      result to be bitwise identical.

    Parameters
    ----------
    config : Config
        Search configuration; ``segEdge`` is passed to the native merge.
    job_seg : WaveSegment
        Job segment whose ``frames`` are read.

    Returns
    -------
    object
        The native ``merge_frames`` result (per-detector merged data).

    Raises
    ------
    ValueError
        If ``gpu.read_workers`` is not an integer in
        ``[1, MAX_READ_WORKERS]``.
    AssertionError
        If validation is enabled and the parallel result differs from the
        serial reads.
    """
    workers = gpu_options(config).read_workers

    def read(frame: Any) -> Any:
        return read_single_frame_from_job_segment(config, frame, job_seg)

    start = time.perf_counter()
    processes = gpu_options(config).read_processes
    if processes:
        # Spawn avoids inheriting CUDA/JAX state. Native frame decoding is
        # unchanged; each child owns its independent reader and file handle.
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            frames = list(
                pool.map(_read_frame, repeat(config), job_seg.frames, repeat(job_seg))
            )
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            frames = list(pool.map(read, job_seg.frames))
    actual = merge_frames(job_seg, frames, config.segEdge)
    logger.info(
        "GPU job parallel frame read: workers=%d processes=%s seconds=%.6f",
        workers,
        processes,
        time.perf_counter() - start,
    )
    if gpu_options(config).validate_read:
        from .validation import leaves

        expected = merge_frames(
            job_seg, [read(frame) for frame in job_seg.frames], config.segEdge
        )
        if leaves(expected) != leaves(actual):
            raise AssertionError("Parallel frame input differs from serial reads")
        logger.info(
            "GPU frame read parity: all detector data and coordinates bitwise exact"
        )
    return actual
