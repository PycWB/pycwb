"""Opt-in GPU job workflow assembled from independent scientific modules.

Select ``pycwb.workflow.subflow.process_job_segment_gpu.process_job_segment``
as ``segment_processer``. Native preparation, trial handling, resume records
and output handling reuse the supplied native recipe through ordinary function
arguments. Each process builds and owns its GPU callables and resident maps.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from functools import partial
from typing import Any

import jax

from pycwb.modules.coherence_gpu.coherence import build_coherence
from pycwb.modules.coherence_gpu.selection import GPUSelector
from pycwb.constants.gpu_options import gpu_options
from pycwb.modules.likelihood_gpu.likelihood import build_likelihood
from pycwb.modules.super_cluster_gpu.super_cluster import build_supercluster
from pycwb.workflow.subflow import process_job_segment_native as native

logger = logging.getLogger(__name__)


def _build_analyzer(config=None) -> tuple[Callable[[Any, int], Any], GPUSelector]:
    """Assemble per-lag functions; factories own each process's GPU resources."""
    options = gpu_options(config)
    coherence, selector = build_coherence(options)
    supercluster = build_supercluster(options)
    likelihood = build_likelihood(config)
    if options.validate_stages:
        from pycwb.modules.stage_validation import paired

        coherence = paired(
            coherence, native.coherence_single_lag, "coherence_single_lag", options=options,
        )
        supercluster = paired(
            supercluster, native.supercluster_single_lag, "supercluster_single_lag", 2,
            options=options,
        )
        likelihood = paired(
            likelihood, native.evaluate_cluster_likelihood, "evaluate_cluster_likelihood", 1,
            options=options,
        )
    return partial(
        native._run_lag_analysis,
        coherence=coherence,
        supercluster=supercluster,
        likelihood=likelihood,
    ), selector


def _process_lags(context: Any, output_context: Any, skip_lags: dict[int, set[int]] | None) -> None:
    """Run pending lags with process-owned functions and parent-only output."""
    from .gpu_output import OutputWriter

    workers = gpu_options(context.config).lag_workers
    writer = OutputWriter(output_context)
    if workers > 1 and not context.sub_job_seg.injections:
        from .process_job_segment_gpu_parallel import process_lags

        process_lags(context, output_context, skip_lags, workers, writer.save)
    else:
        if workers > 1:
            logger.info("GPU lag workers run serially for injection jobs")
        analyze, selector = _build_analyzer(context.config)
        try:
            pending: Iterable[int] = native._iter_pending_lags(context, skip_lags)
            for lag in pending:
                writer.save(output_context, analyze(context, lag))
        finally:
            selector.sessions.clear()
    # On failure, leave pending progress uncommitted for native resume cleanup.
    writer.close()


def process_job_segment(working_dir, config, job_seg, *args: Any, **kwargs: Any) -> Any:
    """Select accelerated functions and reuse the supplied native job recipe.

    This entry point owns its function choices and lag worker scheduling. Custom
    compositions use their own segment_processer, with ordinary scientific calls
    or selected helpers from the native recipe. Device resources remain local to
    the process that uses them.
    """
    if not jax.config.x64_enabled:
        raise RuntimeError("Set JAX_ENABLE_X64=1 for the experimental GPU processor")
    jax.devices("gpu")
    owned = {"lag_processor", "read_data", "condition_data", "prepare_coherence", "prepare_td"}
    if owned.intersection(kwargs):
        raise ValueError("This entry point owns its lag processor and preparation functions")

    options = gpu_options(config)
    read_data = native.read_from_job_segment
    condition_data = native.condition_strains
    prepare_coherence = native.setup_coherence
    prepare_td = native.build_td_inputs_cache

    # Providers own decode admission; avoid nested readers outside their budget.
    if options.read_workers > 1 and kwargs.get("input_provider") is None:
        from pycwb.modules.read_data.parallel import read_from_job_segment

        read_data = partial(
            read_from_job_segment, workers=options.read_workers,
            processes=options.read_processes, validate=options.validate_read,
        )
    if options.condition_workers > 1:
        from pycwb.modules.data_conditioning.parallel import condition_strains

        condition_data = partial(
            condition_strains, workers=options.condition_workers,
            validate=options.validate_conditioning,
        )
    if options.td_setup_workers > 1:
        from pycwb.modules.super_cluster_gpu.td_setup_parallel import build_td_inputs_cache

        prepare_td = build_td_inputs_cache
    if options.setup_workers > 1 or options.wdm_prefilter:
        from pycwb.modules.coherence_gpu.coherence import setup_coherence

        prepare_coherence = setup_coherence
    if options.overlap_setup:
        from .gpu_setup_overlap import OverlappedSetup

        overlapped = OverlappedSetup(prepare_coherence, prepare_td)
        prepare_coherence = overlapped.setup_coherence
        prepare_td = overlapped.build_td_inputs_cache

    with jax.default_device(jax.devices("cpu")[0]):
        return native.process_job_segment(
            working_dir, config, job_seg, *args, **kwargs,
            lag_processor=_process_lags,
            read_data=read_data,
            condition_data=condition_data,
            prepare_coherence=prepare_coherence,
            prepare_td=prepare_td,
        )


process_job_segment.supports_input_provider = True  # type: ignore[attr-defined]


def requested_cores(config):
    """CPU slots for the busiest configured GPU preparation/analysis stage."""
    options = gpu_options(config)
    setup = (options.setup_workers + options.td_setup_workers if options.overlap_setup
             else max(options.setup_workers, options.td_setup_workers))
    return max(options.lag_workers, options.read_workers, options.condition_workers, setup)


process_job_segment.requested_cores = requested_cores
