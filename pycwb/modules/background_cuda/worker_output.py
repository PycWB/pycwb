"""Opt-in native reconstruction in lag workers for catalog-only background.

With ``PYCWB_GPU_WORKER_OUTPUT=1`` each spawned lag worker runs the native
post-processing (waveform reconstruction and Q-veto) on its own lag and ships
only the resulting timings back to the parent, which then performs the file
writes. The worker never creates files: :func:`validate` rejects every
configuration that would make the native post-processing write waveforms,
plots or injection products.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from pycwb.workflow.subflow import process_job_segment_native as native

from . import flags

FILE_PRODUCING_FLAGS = (
    "save_waveform",
    "save_cluster",
    "save_sky_map",
    "plot_waveform",
    "plot_trigger",
    "plot_sky_map",
)
"""Config attributes whose truth makes the native post-processing produce files."""


def enabled() -> bool:
    """Return whether ``PYCWB_GPU_WORKER_OUTPUT=1``; read at call time."""
    return flags.enabled("WORKER_OUTPUT")


def validate(context: Any) -> None:
    """Reject configurations for which worker output would have to write files.

    Called by the parent before spawning workers and again by each worker and
    by the parent's writer before every use, so a context that drifts between
    processes still fails loudly.

    Parameters
    ----------
    context
        Native ``LagAnalysisContext`` or ``LagOutputContext``; only ``config``
        and ``sub_job_seg.injections`` are inspected.

    Raises
    ------
    ValueError
        If the segment carries injections or any of
        ``FILE_PRODUCING_FLAGS`` is enabled in ``context.config``.
    """
    config = context.config
    if context.sub_job_seg.injections or any(getattr(config, flag, False) for flag in FILE_PRODUCING_FLAGS):
        raise ValueError("GPU worker output requires catalog-only background without plots")


@dataclass
class ProcessedLag:
    """Native lag result together with the post-processing timings computed in a worker.

    Attributes
    ----------
    result : LagResult
        Native ``LagResult`` whose events have been reconstructed and Q-vetoed.
    timings : tuple of float
        ``(reconstruct_elapsed, qveto_elapsed, plot_elapsed)`` returned by the
        native ``_postprocess_saved_triggers``; the parent's writer substitutes
        these for its own post-processing so the work runs only once.
    """

    result: Any
    timings: tuple[float, float, float]

    @property
    def lag(self) -> int:
        """Lag index of the wrapped result, mirroring ``LagResult.lag``."""
        return self.result.lag


def process(context: Any, result: Any) -> ProcessedLag:
    """Run native post-processing on one lag inside the worker.

    Parameters
    ----------
    context
        Native ``LagAnalysisContext`` of the worker.
    result : LagResult
        Native result of the lag just analysed in this worker.

    Returns
    -------
    ProcessedLag
        ``result`` with the timings of the native post-processing.

    Raises
    ------
    ValueError
        If :func:`validate` rejects the context or any event is an injection.
    """
    validate(context)
    if any(event.injection for event, _, _ in result.events_data):
        raise ValueError("GPU worker output cannot process injections")
    # All file-producing branches were rejected above. The exact native flow
    # computes all six waveform products and Q-veto; only its owner changes.
    output = SimpleNamespace(
        config=context.config,
        sub_job_seg=context.sub_job_seg,
        wave_file=None,
        queue=None,
    )
    timings = native._postprocess_saved_triggers(output, result, [""] * len(result.events_data))
    return ProcessedLag(result, timings)
