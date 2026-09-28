"""Bounded spawned GPU lag workers with native shared inputs and parent output.

Workers are spawned (never forked) so that each owns its CUDA context, JAX
client, compiled kernels and device caches. Prepared inputs are shared through
the native memory-mapped context file; results return to the parent, which is
the only process that writes catalog files. Worker count is bounded by device
memory and must be measured for each workload before it is raised.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax

from pycwb.workflow.subflow import process_job_segment_native as native
from pycwb.workflow.subflow import process_job_segment_parallel as shared

from pycwb.workflow.subflow import gpu_worker_output as worker_output
from pycwb.constants.gpu_options import gpu_options
from pycwb.constants.execution_profile import execution_profile
from functools import partial
from pycwb.utils.function_binding import specialize
from pycwb.workflow.subflow.process_job_segment_gpu import _build_analyzer

logger = logging.getLogger(__name__)

# Per-worker state, populated once by the pool initializer.
_analyzer: Callable[[Any, int], Any] | None = None
_selector: Any | None = None
_profile_directory: Path | None = None


def _worker_needs_jax_gpu(options=None) -> bool:
    """Return whether any enabled lag stage runs JAX on the device.

    Only the JAX alignment/selection pair uses JAX on the GPU inside lag
    workers; every CUDA stage drives the device through Numba. When the CUDA
    selection backend is active the worker keeps JAX on the CPU, which avoids
    creating an unused CUDA client per worker.
    """
    return not gpu_options(options).selection_cuda


def _initialize(
    path: str, inner_threads: int, log_directory: str, *, options=None
) -> None:
    """Pool initializer: map shared inputs and build this worker's GPU bindings.

    Parameters
    ----------
    path : str
        Native shared-input context file.
    inner_threads : int
        Numba thread count for the worker; always ``1`` for GPU workers.
    log_directory : str
        Directory receiving ``lag_worker_<pid>.log``.
    """
    global _analyzer, _selector, _profile_directory
    _profile_directory = Path(log_directory).parent
    if not _worker_needs_jax_gpu(options):
        # Must precede the first JAX backend initialization in this process.
        jax.config.update("jax_platforms", "cpu")
    # JAX arrays in the prepared context, if present, must stay on the CPU on load.
    with jax.default_device(jax.devices("cpu")[0]):
        shared._initialize_process(path, inner_threads, log_directory)
        if gpu_options(options).quiet_driver:
            # Driver allocation tracing emitted over a million INFO messages in
            # a full job. Keep warnings/errors and the scientific stage logs.
            logging.getLogger("numba.cuda").setLevel(logging.WARNING)
        _analyzer, _selector = _build_analyzer(shared._worker_context.config)
        logger.info(
            "GPU worker source=%s analyzer_source=%s jax_platforms=%s",
            __file__,
            _build_analyzer.__code__.co_filename,
            jax.config.jax_platforms,
        )


def _analyze(lag: int) -> Any:
    """Analyze one lag in this worker and return the native ``LagResult``."""
    from pycwb.modules.gpu_utils.profiling import span

    if _analyzer is None or shared._worker_context is None:
        raise RuntimeError("GPU lag worker not initialized")
    try:
        with (
            span(
                lag,
                "analysis",
                _profile_directory,
                options=gpu_options(shared._worker_context.config),
            ),
            jax.default_device(jax.devices("cpu")[0]),
        ):
            result = _analyzer(shared._worker_context, lag)
            if worker_output.enabled(shared._worker_context.config):
                return worker_output.process(shared._worker_context, result)
            return result
    finally:
        native._cleanup_lag_output_state(
            release_jax=False, profile=execution_profile(shared._worker_context.config)
        )


def process_lags(
    context: Any,
    output_context: Any,
    skip_lags: dict[int, set[int]] | None,
    workers: int,
    save: Callable[[Any, Any], None],
) -> None:
    """Run pending lags on ``workers`` spawned GPU workers.

    Parameters
    ----------
    context
        Native ``LagAnalysisContext``; its Numba thread count is forced to one.
    output_context
        Native ``LagOutputContext`` used by ``save`` in the parent.
    skip_lags : dict or None
        Native resume record of committed lags.
    workers : int
        Requested worker count; reduced to the pending lag count.
    save : callable
        ``save(output_context, result)`` invoked in the parent for each
        completed lag, in completion order.
    """
    if worker_output.enabled(context.config):
        worker_output.validate(context)
    pending = list(native._iter_pending_lags(context, skip_lags))
    if not pending:
        return
    workers = min(workers, len(pending))
    context = replace(context, numba_threads=1)
    logger.info(
        "GPU shared-input lag workers: workers=%d pending=%d", workers, len(pending)
    )
    consume = specialize(
        shared._consume_bounded, native=SimpleNamespace(_save_lag_outputs=save)
    )
    launch = specialize(
        shared._process_shared_inputs,
        _initialize_process=partial(_initialize, options=gpu_options(context.config)),
        _analyze_process=_analyze,
        _consume_bounded=consume,
    )
    launch(context, output_context, pending, workers, 1)
