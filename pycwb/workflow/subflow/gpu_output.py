"""Bounded parent-only catalog batches; commit progress after durable triggers.

Ownership
---------
Only the parent (job) process instantiates :class:`OutputWriter` and therefore
only the parent writes catalog and progress files. Lag workers never touch the
catalog: they return results to the parent, which saves them here in
completion order.

Durability
----------
:class:`BufferedCatalog` performs two native-format atomic replacements per
flush, triggers first and progress second. A crash between them leaves stale
triggers without progress, never completed progress pointing at missing
events. A failed or interrupted uncommitted batch is recomputed on native
resume; native stale-trigger cleanup removes events written without
corresponding progress.
"""

from __future__ import annotations

import logging
import time
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.parquet as pq
from filelock import SoftFileLock

from pycwb.modules.catalog.catalog import PROGRESS_SCHEMA, Catalog, _write_table_atomic
from pycwb.workflow.subflow import process_job_segment_native as native
from pycwb.workflow.subflow.job_segment_progress import _catalog_path

from pycwb.constants.gpu_options import gpu_options

if TYPE_CHECKING:
    from collections.abc import Callable

    from pycwb.workflow.subflow.process_job_segment_native import LagOutputContext

logger = logging.getLogger(__name__)

MIN_BATCH_LAGS = 1
"""Smallest accepted ``batch_lags``; one lag per commit is the native cadence."""

MAX_BATCH_LAGS = 256
"""Largest accepted ``batch_lags``; bounds the work lost to an uncommitted batch on interruption."""

DEFAULT_BATCH_LAGS = 32
"""``batch_lags`` used when a caller does not specify one."""

MAX_BUFFERED_TRIGGERS = 4096
"""Trigger count that forces a flush before ``batch_lags`` completes, bounding parent memory."""

PROGRESS_LOCK_TIMEOUT_SECONDS = 30
"""Soft file-lock wait on the progress file; matches the native catalog lock timeout."""


class BufferedCatalog:
    """Queue-compatible sink that batches triggers and progress into one catalog.

    The instance replaces the collector queue in the parent's
    ``LagOutputContext`` so the native save path sends its ``trigger`` and
    ``progress`` messages here instead of to the batch-runner collector.
    Messages accumulate in memory until :meth:`commit_if_ready` decides a batch
    is complete, then :meth:`flush` writes them in the native file formats.

    Parameters
    ----------
    path : str or os.PathLike
        Existing catalog ``.parquet`` file opened with
        :meth:`pycwb.modules.catalog.catalog.Catalog.open`.
    batch_lags : int, optional
        Number of completed lags per commit, in
        ``[MIN_BATCH_LAGS, MAX_BATCH_LAGS]``. Default is ``DEFAULT_BATCH_LAGS``.

    Attributes
    ----------
    catalog : Catalog
        Native catalog used for trigger appends; its ``progress_file`` names
        the companion progress table.
    batch_lags : int
        Validated commit cadence.
    triggers : list
        Uncommitted native ``Trigger`` objects.
    progress : list of dict
        Uncommitted progress rows following ``PROGRESS_SCHEMA``.
    flush_seconds : float
        Accumulated wall time spent inside :meth:`flush`.
    flush_count : int
        Number of completed flushes.

    Raises
    ------
    ValueError
        If ``batch_lags`` is outside ``[MIN_BATCH_LAGS, MAX_BATCH_LAGS]``.
    """

    def __init__(self, path: str | Path, batch_lags: int = DEFAULT_BATCH_LAGS) -> None:
        if not MIN_BATCH_LAGS <= batch_lags <= MAX_BATCH_LAGS:
            raise ValueError(
                f"Output batch must be between {MIN_BATCH_LAGS} and {MAX_BATCH_LAGS} lags"
            )
        self.catalog = Catalog.open(str(path))
        self.batch_lags = int(batch_lags)
        self.triggers: list[Any] = []
        self.progress: list[dict[str, Any]] = []
        self.flush_seconds = 0.0
        self.flush_count = 0

    def put(self, message: dict[str, Any]) -> None:
        """Buffer one native output message.

        Parameters
        ----------
        message : dict
            Native collector message. ``{"type": "trigger", "trigger": Trigger}``
            buffers the trigger; ``{"type": "progress", ...}`` buffers the
            remaining keys as a progress row stamped with the current time.

        Raises
        ------
        ValueError
            If ``message["type"]`` is neither ``"trigger"`` nor ``"progress"``.
        """
        kind = message["type"]
        if kind == "trigger":
            self.triggers.append(message["trigger"])
        elif kind == "progress":
            self.progress.append(
                {k: v for k, v in message.items() if k != "type"}
                | {"timestamp": time.time()}
            )
        else:
            raise ValueError(f"Unexpected output message {kind}")

    def commit_if_ready(self) -> None:
        """Flush when a batch of lags completed or the trigger buffer is large.

        Called once per saved lag, after the native save path has emitted that
        lag's progress record, so a flush never splits a lag between commits.
        """
        if (
            len(self.progress) >= self.batch_lags
            or len(self.triggers) >= MAX_BUFFERED_TRIGGERS
        ):
            self.flush()

    def _append_progress(self) -> None:
        """Append buffered progress rows to the progress table under the native soft lock."""
        path = Path(self.catalog.progress_file)
        new_rows = pa.Table.from_pylist(self.progress, schema=PROGRESS_SCHEMA)
        with SoftFileLock(str(path) + ".lock", timeout=PROGRESS_LOCK_TIMEOUT_SECONDS):
            if path.exists() and path.stat().st_size:
                old = pq.read_table(path, schema=PROGRESS_SCHEMA)
                new_rows = pa.concat_tables([old, new_rows])
            _write_table_atomic(new_rows, str(path), compression="snappy")

    def flush(self) -> None:
        """Commit buffered triggers, then buffered progress, and clear both.

        A buffer with progress rows is written as two native-format atomic
        replacements, triggers first. A buffer without progress rows is left
        untouched when it is empty.

        Raises
        ------
        RuntimeError
            If triggers are buffered without any completed lag record; such a
            commit could never be reconciled by native resume.
        """
        if not self.progress:
            if self.triggers:
                raise RuntimeError(
                    "Cannot commit triggers without completed lag records"
                )
            return
        start = time.perf_counter()
        nt, nl = len(self.triggers), len(self.progress)
        # Two native-format atomic replacements. A crash between them leaves
        # stale triggers, never completed progress pointing at missing events.
        self.catalog.add_triggers(self.triggers)
        self.triggers.clear()
        self._append_progress()
        self.progress.clear()
        elapsed = time.perf_counter() - start
        self.flush_seconds += elapsed
        self.flush_count += 1
        logger.info(
            "GPU output batch: lags=%d triggers=%d elapsed=%.6f", nl, nt, elapsed
        )


class OutputWriter:
    """Parent-side saver for completed lags with optional catalog batching.

    Constructed once per job in the parent process. Switches are read here at
    construction time:

    * ``gpu.q_reconstruction=true`` swaps the native save function for the
      Q-veto reconstruction variant from :mod:`pycwb.workflow.subflow.gpu_reconstruction`.
    * ``gpu.output_batch=n`` with ``n > 1`` routes trigger and progress
      messages to a :class:`BufferedCatalog` instead of the collector queue.

    Parameters
    ----------
    context : LagOutputContext
        Native output context of the parent. When batching is enabled the
        stored copy has its ``queue`` replaced by the :class:`BufferedCatalog`.

    Attributes
    ----------
    context : LagOutputContext
        Output context actually passed to the native save function.
    save_native : callable
        ``save(output_context, result)`` used for plain native ``LagResult``.
    sink : BufferedCatalog or None
        Batching sink, or ``None`` when ``gpu.output_batch`` is 1.

    Raises
    ------
    ValueError
        If batching is requested with saved waveforms or injections, without
        a catalog path, or with a non-positive batch size; or if
        ``gpu.output_batch`` is not an integer.
    """

    def __init__(self, context: LagOutputContext) -> None:
        self.context = context
        self.options = gpu_options(context.config)
        self.save_native: Callable[[Any, Any], None] = native._save_lag_outputs
        if self.options.q_reconstruction:
            from pycwb.workflow.subflow.gpu_reconstruction import make_save

            self.save_native = make_save(context)
        batch = self.options.output_batch
        self.sink: BufferedCatalog | None = None
        if batch > 1:
            if (
                getattr(context.config, "save_waveform", False)
                or context.sub_job_seg.injections
            ):
                raise ValueError(
                    "GPU catalog batching is currently validated only for background without saved waveforms"
                )
            # The batch runner supplies a collector queue. This background path
            # emits only triggers/progress, so route both to this single parent
            # sink. The collector receives no messages from this job and exits
            # normally at its sentinel. Catalog locks retain inter-job isolation.
            path = _catalog_path(
                context.working_dir, context.config, context.catalog_file
            )
            if path is None:
                raise ValueError("GPU output batching requires a catalog path")
            self.sink = BufferedCatalog(path, batch)
            self.context = replace(context, queue=self.sink)
        elif batch != 1:
            raise ValueError("Output batch must be positive")

    def save(self, unused_context: Any, result: Any) -> None:
        """Save one completed lag, profiling the output stage when requested.

        Parameters
        ----------
        unused_context
            Ignored. The parameter keeps the native ``save(output_context,
            result)`` call signature expected by :func:`~.processor` and
            :func:`~.process_parallel.process_lags`; the writer always uses the
            context it was constructed with, which may carry the batching sink.
        result : LagResult or ProcessedLag
            Product of one lag, either the native result or a worker-processed
            wrapper from :mod:`.worker_output`.
        """
        if not self.options.profile_lags:
            self._save(result)
            return
        from pycwb.modules.gpu_utils.profiling import span

        with span(result.lag, "output", self.context.working_dir, options=self.options):
            self._save(result)

    def _save(self, result: Any) -> None:
        """Dispatch to the native or worker-output save path, then commit if a batch is ready."""
        from pycwb.workflow.subflow.gpu_worker_output import ProcessedLag, validate

        if isinstance(result, ProcessedLag):
            from pycwb.utils.function_binding import specialize

            validate(self.context)
            timings = result.timings
            save = specialize(
                native._save_lag_outputs,
                _postprocess_saved_triggers=lambda *_: timings,
            )
            save(self.context, result.result)
        else:
            self.save_native(self.context, result)
        if self.sink is not None:
            self.sink.commit_if_ready()

    def close(self) -> None:
        """Flush any remaining batch and log the batching totals.

        Must be called by the parent after the last :meth:`save`; without it
        the final partial batch is never committed and is recomputed on resume.
        """
        if self.sink is not None:
            self.sink.flush()
            logger.info(
                "GPU output batches complete: batches=%d write_seconds=%.6f",
                self.sink.flush_count,
                self.sink.flush_seconds,
            )
