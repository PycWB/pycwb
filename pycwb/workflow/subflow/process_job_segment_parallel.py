"""Run background lags in spawned processes with shared prepared input arrays.

The segment parent prepares data once and writes every output. Each worker maps
one job-local input file and receives only lag indices. At most twice the worker
count is in flight, bounding the number of queued tasks and retained results.

Select ``process_job_segment`` through ``segment_processer``. One worker and
injection trials run serially.
"""

from pycwb.constants.execution_profile import execution_profile
import logging
import multiprocessing
import os
from collections.abc import Callable, Iterable
from concurrent.futures import (
    FIRST_COMPLETED,
    Executor,
    Future,
    ProcessPoolExecutor,
    wait,
)
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib

from pycwb.workflow.subflow import process_job_segment_native as native
from pycwb.workflow.subflow.job_segment_resources import _parallel_inner_threads

logger = logging.getLogger(__name__)
_worker_context: native.LagAnalysisContext | None = None


def _consume_bounded(
    executor: Executor,
    analyze: Callable[[int], native.LagResult],
    pending_lags: Iterable[int],
    output_context: native.LagOutputContext,
    workers: int,
    *,
    save: Callable | None = None,
) -> None:
    """Save completed results in the parent, with bounded work in flight.

    A lag is committed only through the native output path. On failure, cancel
    queued work and propagate the exception; the caller joins running workers
    before removing their input maps. Previously saved lags remain resumable.
    """
    save = save or native._save_lag_outputs
    iterator = iter(pending_lags)
    futures: dict[Future, int] = {}
    try:
        while True:
            while len(futures) < 2 * workers:
                try:
                    lag = next(iterator)
                except StopIteration:
                    break
                futures[executor.submit(analyze, lag)] = lag
            if not futures:
                return

            completed, _ = wait(futures, return_when=FIRST_COMPLETED)
            # Drain and release results before submitting more work. Workers
            # never receive output_context or write the catalog/progress files.
            while completed:
                future = completed.pop()
                lag = futures.pop(future)
                result = future.result()
                if result.lag != lag:
                    raise RuntimeError(f"Lag worker returned {result.lag}, expected {lag}")
                save(output_context, result)
                del result, future
    finally:
        for future in futures:
            future.cancel()


def _initialize_process(path: str, inner_threads: int, log_directory: str) -> None:
    """Map prepared arrays once per worker, without copying them per lag."""
    import numba

    global _worker_context
    logging.basicConfig(
        filename=str(Path(log_directory) / f"lag_worker_{os.getpid()}.log"),
        level=logging.INFO,
        force=True,
        format="%(asctime)s - %(funcName)s - %(levelname)s - %(message)s",
    )
    # Copy-on-write maps share physical input pages but isolate worker writes.
    # Their writable array signatures also reuse the native Numba disk cache.
    _worker_context = replace(joblib.load(path, mmap_mode="c"), numba_threads=inner_threads)
    numba.set_num_threads(min(inner_threads, numba.config.NUMBA_NUM_THREADS))


def _analyze_process(lag: int) -> native.LagResult:
    """Use the native analysis; reclaim worker-local scratch after each lag."""
    if _worker_context is None:
        raise RuntimeError("Lag worker inputs were not initialized")
    try:
        return native._run_lag_analysis(_worker_context, lag)
    finally:
        # Analysis workers do not need to initialize JAX just to perform GC.
        native._cleanup_lag_output_state(release_jax=False, profile=execution_profile(_worker_context.config))


def _process_serial(
    context: native.LagAnalysisContext,
    output_context: native.LagOutputContext,
    pending_lags: Iterable[int],
) -> None:
    for lag in pending_lags:
        native._save_lag_outputs(output_context, native._run_lag_analysis(context, lag))


def _process_shared_inputs(
    context: native.LagAnalysisContext,
    output_context: native.LagOutputContext,
    pending_lags: Iterable[int],
    workers: int,
    inner_threads: int,
    *,
    initialize: Callable | None = None,
    analyze: Callable | None = None,
    consume: Callable | None = None,
) -> None:
    """Keep job-local input maps alive until every spawned worker exits."""
    initialize = initialize or _initialize_process
    analyze = analyze or _analyze_process
    consume = consume or _consume_bounded
    log_directory = Path(output_context.working_dir) / "log"
    log_directory.mkdir(parents=True, exist_ok=True)
    # Disk-backed job scratch avoids storing a second input copy on tmpfs.
    # The parent retains its original prepared arrays for output processing.
    with TemporaryDirectory(prefix=".lag-inputs-", dir=output_context.working_dir) as directory:
        path = Path(directory) / "context.joblib"
        joblib.dump(context, path, compress=0)
        logger.info("Shared lag input file: bytes=%d path=%s", path.stat().st_size, path)
        # Forking after JAX/OpenMP initialization can inherit unsafe runtime state.
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=initialize,
            initargs=(str(path), inner_threads, str(log_directory)),
        ) as executor:
            consume(executor, analyze, pending_lags, output_context, workers)


def process_lags(
    context: native.LagAnalysisContext,
    output_context: native.LagOutputContext,
    skip_lags: dict[int, set[int]] | None,
) -> None:
    """Run pending lags, sharing prepared inputs across background workers.

    ``parallel_lag_workers`` controls process count; inner Numba threads follow
    ``parallel_lag_inner_threads``. Set BLAS/OpenMP limits before job launch to
    avoid nested parallelism. Resume skips already committed lags before any
    input serialization or worker startup.
    """
    pending = list(native._iter_pending_lags(context, skip_lags))
    if not pending:
        return
    requested_workers = max(1, int(getattr(context.config, "parallel_lag_workers", 1) or 1))
    workers = min(len(pending), requested_workers)
    if context.sub_job_seg.injections:
        _process_serial(context, output_context, pending)
        return

    inner_threads = _parallel_inner_threads(context.config, workers)
    context = replace(context, numba_threads=inner_threads)
    logger.info(
        "Shared-input lag processes: workers=%d inner_threads=%d pending=%d",
        workers,
        inner_threads,
        len(pending),
    )
    if workers == 1:
        _process_serial(context, output_context, pending)
    else:
        _process_shared_inputs(context, output_context, pending, workers, inner_threads)


def process_job_segment(*args, **kwargs):
    """Native segment preparation/output with shared-input lag processes."""
    return native.process_job_segment(*args, **kwargs, lag_processor=process_lags)


process_job_segment.supports_input_provider = True
