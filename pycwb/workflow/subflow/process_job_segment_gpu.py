"""Opt-in GPU job workflow assembled from independent scientific modules.

Select ``pycwb.workflow.subflow.process_job_segment_gpu.process_job_segment``
as ``segment_processer``. Native preparation, trial handling, resume records
and output contracts are shared with the CPU workflow through explicit stage
bundles. Each process builds and owns its GPU stages and resident maps.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import replace
from functools import partial
from typing import Any

import jax

from pycwb.modules.coherence_gpu.coherence import build_coherence
from pycwb.modules.coherence_gpu.selection import GPUSelector
from pycwb.constants.gpu_options import gpu_options
from pycwb.modules.likelihood_gpu.likelihood import build_likelihood
from pycwb.modules.super_cluster_gpu.super_cluster import build_supercluster
from pycwb.workflow.subflow import process_job_segment_native as native
from pycwb.workflow.subflow.job_segment_stages import LagStages, PreparationStages

logger = logging.getLogger(__name__)



def _build_analyzer(config=None) -> tuple[Callable[[Any, int], Any], GPUSelector]:
    """Assemble explicit lag stages; factories own each process's GPU resources."""
    options = gpu_options(config)
    coherence, selector = build_coherence(options)
    bindings = {
        "coherence_single_lag": coherence,
        "supercluster_single_lag": build_supercluster(options),
        "evaluate_cluster_likelihood": build_likelihood(config),
    }
    if options.validate_stages:
        from pycwb.modules.gpu_utils.validation import paired

        for name, mutable_arg in (
            ("coherence_single_lag", None),
            ("supercluster_single_lag", 2),
            ("evaluate_cluster_likelihood", 1),
        ):
            reference = getattr(native, name)
            bindings[name] = paired(
                bindings.get(name, reference),
                reference,
                name,
                mutable_arg,
                options=options,
            )
    stages = LagStages(**bindings, event_factory=native.Event)
    return partial(native._run_lag_analysis, stages=stages), selector


def _process_lags(context: Any, output_context: Any, skip_lags: dict[int, set[int]] | None) -> None:
    """Run pending lags with process-owned stages and parent-only output."""
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


def _build_preparation(config=None, *, input_provider: Any = None) -> PreparationStages:
    """Choose bounded preparation stages, respecting provider-owned reads."""
    options = gpu_options(config)
    stages = PreparationStages(
        native.read_from_job_segment, native.condition_strains,
        native.setup_coherence, native.build_td_inputs_cache,
    )
    # Providers own decode admission and supply job-local copies; do not start
    # nested readers outside the allocation cache's resource accounting.
    if options.read_workers > 1 and input_provider is None:
        from pycwb.modules.read_data.parallel import read_from_job_segment

        stages = replace(stages, read_from_job_segment=partial(read_from_job_segment, workers=options.read_workers, processes=options.read_processes, validate=options.validate_read))
    if options.condition_workers > 1:
        from pycwb.modules.data_conditioning.parallel import condition_strains

        stages = replace(stages, condition_strains=partial(condition_strains, workers=options.condition_workers, validate=options.validate_conditioning))
    if options.td_setup_workers > 1:
        from pycwb.modules.super_cluster_gpu.td_setup_parallel import build_td_inputs_cache

        stages = replace(stages, build_td_inputs_cache=build_td_inputs_cache)
    if options.setup_workers > 1 or options.wdm_prefilter:
        from pycwb.modules.coherence_gpu.coherence import setup_coherence

        stages = replace(stages, setup_coherence=setup_coherence)
    if options.overlap_setup:
        from .gpu_setup_overlap import OverlappedSetup

        preparation = OverlappedSetup(stages.setup_coherence, stages.build_td_inputs_cache)
        stages = replace(
            stages,
            setup_coherence=preparation.setup_coherence,
            build_td_inputs_cache=preparation.build_td_inputs_cache,
        )
    return stages


def process_job_segment(working_dir, config, job_seg, *args: Any, **kwargs: Any) -> Any:
    """Run the native job lifecycle with explicit GPU stage composition.

    Accepts the native processor arguments except ``lag_processor`` and
    ``preparation_stages``, which this workflow owns. Requires JAX x64 and a
    visible GPU before any preparation; never silently falls back to CPU.
    """
    if not jax.config.x64_enabled:
        raise RuntimeError("Set JAX_ENABLE_X64=1 for the experimental GPU processor")
    jax.devices("gpu")
    if "lag_processor" in kwargs or "preparation_stages" in kwargs:
        raise ValueError("This entry point owns its lag processor and preparation stages")
    preparation = _build_preparation(config, input_provider=kwargs.get("input_provider"))
    with jax.default_device(jax.devices("cpu")[0]):
        return native.process_job_segment(
            working_dir, config, job_seg, *args, **kwargs, lag_processor=_process_lags, preparation_stages=preparation,
        )


process_job_segment.supports_input_provider = True  # type: ignore[attr-defined]


def requested_cores(config):
    """CPU slots for the busiest configured GPU preparation/analysis stage."""
    options = gpu_options(config)
    setup = (options.setup_workers + options.td_setup_workers if options.overlap_setup
             else max(options.setup_workers, options.td_setup_workers))
    return max(options.lag_workers, options.read_workers, options.condition_workers, setup)


process_job_segment.requested_cores = requested_cores
