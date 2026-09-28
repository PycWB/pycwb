"""Supervised execution with persistent raw inputs and replaceable compute workers."""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple, TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from multiprocessing.process import BaseProcess

    from pycwb.types.job import WaveSegment

    from .planner import ExecutionPlan


import logging
import multiprocessing
import os
import pickle
import signal
import time
import traceback
from collections import Counter, deque
from contextlib import ExitStack
from dataclasses import asdict, dataclass, replace
from multiprocessing.connection import Connection, wait
from pathlib import Path

from .cache import FrameCache, FrameProvider, merged_requests
from .planner import prepare_plan, write_document
from .resources import (
    MemoryBudget,
    MemoryMonitor,
    available_cpus,
    available_memory,
)
from .settings import ExecutionSettings
from .writer import OutputClient, OutputWriter

logger = logging.getLogger(__name__)


@dataclass
class ExecutionContext:
    """Resolved scientific work and workflow-specific output/legacy adapters."""

    jobs: list[WaveSegment]
    config: Any
    processor: Callable[..., Any]
    working_dir: str
    catalog_file: str
    compress_json: bool = True
    workers: int = 1
    skip_lags: dict[int, dict[int, set[int]]] | None = None
    legacy: Callable[[list[WaveSegment]], Any] | None = None
    allocated_cores: int | None = None
    memory_limit: int | None = None


class SimpleExecutor:
    """Adapt planned ordering to the existing workflow implementation."""

    def execute(self, plan: ExecutionPlan, context: ExecutionContext) -> Any:
        """Execute the validated plan, propagating any worker or output failure."""
        if context.legacy is None:
            raise ValueError("Simple executor requires a legacy workflow adapter")
        return context.legacy([context.jobs[i] for i in plan.order])


def _segment_worker(
    connection: Connection,
    processor: Callable[..., Any],
    config: Any,
    job: WaveSegment,
    working_dir: str,
    catalog_file: str,
    compress_json: bool,
    skip_lags: dict[int, set[int]] | None,
    provider: FrameProvider | None,
    message_limit: int,
    cpus: list[int],
) -> None:
    """Spawn entry point; numerical state belongs to this disposable process."""
    try:
        if hasattr(os, "setsid"):
            os.setsid()
        if hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, cpus)
        from pycwb.modules.logger import logger_init

        logger_init(
            log_file=str(Path(working_dir) / "log" / f"job_{job.index}.log"),
            log_level="INFO",
        )
        kwargs = {
            "compress_json": compress_json,
            "catalog_file": catalog_file,
            "queue": OutputClient(connection, message_limit),
        }
        if skip_lags:
            kwargs["skip_lags"] = skip_lags
        if provider is not None:
            kwargs["input_provider"] = provider
        processor(working_dir, config, job, **kwargs)
        connection.send(("done", provider.metrics if provider is not None else {}))
    except BaseException:  # noqa: BLE001 -- report SystemExit/interrupt as failed work too
        connection.send(("error", traceback.format_exc()))
    finally:
        connection.close()


def _stop_process(process: BaseProcess) -> None:
    """Also stop nested lag workers, which inherit the segment's session."""
    if process.pid is None:
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            if hasattr(os, "killpg"):
                os.killpg(process.pid, sig)
            elif process.is_alive():
                process.terminate()
        except ProcessLookupError:
            if process.is_alive():
                process.terminate()
        process.join(timeout=3)
        if not process.is_alive():
            break


def pending_jobs(
    jobs: list[WaveSegment], config: Any, catalog_file: str
) -> tuple[list[WaveSegment], dict[int, dict[int, set[int]]]]:
    """Read durable progress before fetching inputs; never swallow catalog errors."""
    from pycwb.modules.catalog.catalog import Catalog

    catalog = Catalog.open(catalog_file)
    pending, skips = [], {}
    for job in jobs:
        if (
            job.injections is not None
            and not job.injections
            and getattr(config, "analyze_injection_only", False)
        ):
            continue
        trials = {inj.get("trial_idx", 0) for inj in job.injections or []} or {
            job.trial_idx
        }
        completed = catalog.get_completed_lags(job.index)
        expected = set(range(job.n_lag))
        if all(expected.issubset(completed.get(trial, set())) for trial in trials):
            continue
        catalog.remove_stale_triggers(job.index, completed)
        skips[job.index] = completed
        pending.append(job)
    return pending, skips



def estimate_input_peak(plan: ExecutionPlan, config: Any, supports_cache: bool) -> int:
    """Reserve simultaneous decode and merged-input bytes before launching workers."""
    input_peak = 0
    for requests in plan.requests:
        readers = (
            1
            if supports_cache
            else max(1, int(getattr(config, "nproc", 1) or 1))
        )
        decoding = sum(
            sorted((r.decode_bytes for r in requests), reverse=True)[:readers]
        )
        # Source-frame decoding can greatly exceed the requested slice.
        # Include simultaneous decoding and job-owned/merged input buffers.
        input_peak = max(
            input_peak, 3 * decoding + 2 * sum(r.estimated_bytes for r in requests)
        )
    return input_peak


def plan_cache_reuse(plan: ExecutionPlan) -> tuple[dict, dict]:
    """Return reusable frame unions and per-task requests without decoding data."""
    batch_inputs = {}
    cache_requests = {}
    for batch in plan.batches:
        counts = Counter(r.key for i in batch for r in set(plan.requests[i]))
        reusable = {key for key, count in counts.items() if count > 1}
        planned = merged_requests(
            tuple(r for r in plan.requests[i] if r.key in reusable) for i in batch
        )
        for task in batch:
            batch_inputs[task] = planned
            cache_requests[task] = tuple(
                r for r in plan.requests[task] if r.key in reusable
            )
    return batch_inputs, cache_requests

class WorkerState(NamedTuple):
    """Resources retained until a worker exits and its output is acknowledged."""

    process: BaseProcess
    task: int
    provider: FrameProvider
    slot: int


class ScalableExecutor:
    """Own allocation-wide admission, input lifetime and acknowledged output."""

    def execute(self, plan: ExecutionPlan, context: ExecutionContext) -> Any:
        """Execute the validated plan, propagating any worker or output failure."""
        settings = ExecutionSettings.from_config(context.config)
        if context.allocated_cores is not None:
            if type(context.allocated_cores) is not int or context.allocated_cores < 1:
                raise ValueError("allocated_cores must be a positive integer")
            settings = replace(settings, cores=min(settings.cores or context.allocated_cores, context.allocated_cores))
        if context.memory_limit is not None:
            if type(context.memory_limit) is not int or context.memory_limit < 1:
                raise ValueError("memory_limit must be positive integer bytes")
            settings = replace(settings, memory_limit=min(settings.memory_limit or context.memory_limit, context.memory_limit))
        cpus = available_cpus()
        if settings.cores is not None:
            cpus = cpus[: settings.cores]
        inner = max(
            1, int(getattr(context.config, "parallel_lag_inner_threads", 1) or 1)
        )
        lag_workers = max(
            1, int(getattr(context.config, "parallel_lag_workers", 1) or 1)
        )
        per_job_cores = max(
            1, int(getattr(context.config, "nproc", 1) or 1), lag_workers * inner
        )
        resource_request = getattr(context.processor, "requested_cores", None)
        if resource_request is not None:
            per_job_cores = max(per_job_cores, resource_request(context.config))
        if per_job_cores > len(cpus):
            raise ValueError(
                f"Segment requests {per_job_cores} cores but allocation has {len(cpus)}"
            )
        requested = min(context.workers, max(1, len(cpus) // per_job_cores))
        supports_cache = getattr(context.processor, "supports_input_provider", False)
        input_peak = estimate_input_peak(plan, context.config, supports_cache)
        budget = MemoryBudget.resolve(settings, requested, input_peak=input_peak)
        if not supports_cache:
            logger.warning(
                "Processor has no input-provider contract; using direct reads with supervised execution"
            )
        order = deque(plan.order)
        # Duplicate scientific IDs (explicit trial selections) must not race in
        # the same output namespace. Admission below serializes these tasks.
        batch_inputs, cache_requests = plan_cache_reuse(plan)
        cache = FrameCache(
            context.working_dir,
            budget,
            decoder_cpus=cpus,
            max_entries=settings.cache_entries,
        )
        writer = OutputWriter(
            context.config, context.catalog_file, settings.message_limit
        )
        process_context = multiprocessing.get_context("spawn")
        active: dict[Connection, WorkerState] = {}
        retiring: set[Connection] = set()
        completed = []
        started = time.monotonic()

        def stop_on_pressure() -> None:
            # Do not wait for a long decode or catalog flush to finish before
            # stopping workers whose actual selection/scratch exceeds estimates.
            for process, _, _, _ in list(active.values()):
                if process.pid is None:
                    continue
                try:
                    if hasattr(os, "killpg"):
                        os.killpg(process.pid, signal.SIGTERM)
                    else:
                        process.terminate()
                except ProcessLookupError:
                    if process.is_alive():
                        process.terminate()

        monitor = MemoryMonitor(
            lambda: cache.bytes,
            limit=budget.limit - budget.headroom,
            on_pressure=stop_on_pressure,
            headroom=budget.headroom,
        )
        failure = None
        input_metrics: Counter[str] = Counter()
        preloaded = set()
        logger.info("Execution memory reservations: %s", asdict(budget))
        monitor.start()
        try:
            while order or active:
                active_ids = {context.jobs[state.task].index for state in active.values()}
                while order and len(active) < budget.workers:
                    task = order[0]
                    if context.jobs[task].index in active_ids:
                        break
                    if available_memory() < settings.headroom + settings.worker_memory:
                        cache.evict_idle()
                        if active:
                            break
                        raise MemoryError(
                            "Insufficient available memory to admit the next segment"
                        )
                    planned = batch_inputs[task] if settings.preload != "off" else ()
                    # batch mode loads the entire bounded union before launch.
                    # If it does not fit, use demand/union loading automatically.
                    if (
                        settings.preload == "batch"
                        and supports_cache
                        and planned not in preloaded
                        and sum(r.estimated_bytes for r in planned) <= budget.cache
                    ):
                        lease = cache.acquire(planned, planned)
                        cache.release(lease)
                        preloaded.add(planned)
                    provider = (
                        cache.acquire(cache_requests[task], planned)
                        if supports_cache
                        else FrameProvider(())
                    )
                    parent, child = process_context.Pipe()
                    used_slots = {state.slot for state in active.values()}
                    slot = next(i for i in range(budget.workers) if i not in used_slots)
                    assigned = cpus[slot * per_job_cores : (slot + 1) * per_job_cores]
                    job = context.jobs[task]
                    process: BaseProcess = process_context.Process(
                        target=_segment_worker,
                        args=(
                            child,
                            context.processor,
                            context.config,
                            job,
                            context.working_dir,
                            context.catalog_file,
                            context.compress_json,
                            (context.skip_lags or {}).get(job.index),
                            provider if supports_cache else None,
                            settings.message_limit,
                            assigned,
                        ),
                    )
                    try:
                        process.start()
                    except BaseException:
                        parent.close()
                        cache.release(provider)
                        raise
                    finally:
                        child.close()
                    active[parent] = WorkerState(process, task, provider, slot)
                    active_ids.add(job.index)
                    order.popleft()
                ready = (
                    cast(
                        list[Connection],
                        wait([c for c in active if c not in retiring], timeout=0.2),
                    )
                    if active
                    else []
                )
                for connection in ready:
                    if monitor.exceeded:
                        raise MemoryError(
                            "Observed memory reached the safety margin; increase worker_memory or reduce concurrency"
                        )
                    process, task, provider, slot = active[connection]
                    try:
                        payload = connection.recv_bytes(settings.message_limit)
                    except (EOFError, OSError) as exc:
                        raise RuntimeError(
                            f"Job {context.jobs[task].index} worker exited without completion"
                        ) from exc
                    kind, item = pickle.loads(payload)
                    if kind == "write":
                        writer.handle(item, len(payload))
                        connection.send("ok")
                    elif kind == "error":
                        raise RuntimeError(
                            f"Job {context.jobs[task].index} failed:\n{item}"
                        )
                    elif kind == "done":
                        input_metrics.update(item)
                        writer.flush()
                        # Native runtimes can take longer than five seconds to
                        # shut down. Keep their reservations until they exit,
                        # while continuing to service other workers' output.
                        retiring.add(connection)
                    else:
                        raise ValueError(f"Invalid worker message: {kind!r}")
                for connection in list(retiring):
                    process, task, provider, _ = active[connection]
                    if process.is_alive():
                        continue
                    process.join()
                    if process.exitcode != 0:
                        raise RuntimeError(
                            f"Job {context.jobs[task].index} did not exit cleanly "
                            f"(exit code {process.exitcode})"
                        )
                    cache.release(provider)
                    connection.close()
                    del active[connection]
                    retiring.remove(connection)
                    completed.append(task)
                if monitor.exceeded:
                    cache.evict_idle()
                    raise MemoryError(
                        "Observed process memory exceeded execution.memory_limit; increase worker_memory reservation"
                    )
        except BaseException as exc:
            failure = str(exc)
            raise
        finally:
            for connection, (process, task, provider, _) in active.items():
                _stop_process(process)
                connection.close()
                cache.release(provider)
            monitor.close()
            metrics = {
                "status": "failed" if failure else "complete",
                "error": failure,
                "completed_tasks": completed,
                "elapsed_seconds": time.monotonic() - started,
                "peak_tree_rss": monitor.peak_rss,
                "peak_tree_pss": monitor.peak_pss,
                "peak_accounted_bytes": monitor.peak_accounted,
                "min_available_bytes": monitor.min_available,
                "host_swap_growth_bytes": monitor.peak_swap - monitor.initial_swap,
                "budget": asdict(budget),
                "cache": cache.metrics,
                "inputs": dict(input_metrics),
            }
            cache.close()
            stem = Path(context.catalog_file).stem
            try:
                write_document(
                    Path(context.working_dir) / "execution" / f"{stem}.metrics.json",
                    metrics,
                )
            except OSError:
                if failure is None:
                    raise
                logger.exception("Could not write execution failure metrics")
        return metrics


def create_simple_executor() -> SimpleExecutor:
    """Construct the legacy workflow adapter."""
    return SimpleExecutor()


def create_scalable_executor() -> ScalableExecutor:
    """Construct the allocation supervisor."""
    return ScalableExecutor()


def execute_jobs(context: ExecutionContext) -> Any:
    """Resolve extensions and hold exclusive output ownership through shutdown."""
    from filelock import FileLock

    from pycwb.utils.module import import_function

    settings = ExecutionSettings.from_config(context.config)
    executor = (
        import_function(settings.executor)()
        if settings.executor
        else ScalableExecutor()
        if settings.profile == "scalable"
        else SimpleExecutor()
    )
    if not callable(getattr(executor, "execute", None)):
        raise TypeError(
            "Execution factory must return an object with execute(plan, context)"
        )
    stem = Path(context.catalog_file).stem
    with ExitStack() as locks:
        locks.enter_context(
            FileLock(
                str(Path(context.working_dir) / f".execution-{stem}.lock"), timeout=0
            )
        )
        lock_directory = Path(context.working_dir) / ".execution-locks"
        lock_directory.mkdir(exist_ok=True)
        for job_id in sorted({job.index for job in context.jobs}):
            locks.enter_context(
                FileLock(str(lock_directory / f"job-{job_id}.lock"), timeout=0)
            )
        # The legacy catalog uses soft locks. Inspect stale locks only while
        # holding allocation-independent output ownership, including on resume.
        from pycwb.workflow.batch import _cleanup_stale_lock

        catalog_path = Path(context.catalog_file)
        for path in (
            catalog_path,
            catalog_path.with_name(catalog_path.name.replace("catalog", "progress", 1)),
        ):
            _cleanup_stale_lock(str(path) + ".lock")
        if isinstance(executor, ScalableExecutor):
            context.jobs, context.skip_lags = pending_jobs(
                context.jobs, context.config, context.catalog_file
            )
        plan = prepare_plan(context.jobs, context.config, settings)
        write_document(
            Path(context.working_dir) / "execution" / f"{stem}.plan.json",
            plan.document(context.jobs, settings),
        )
        if not context.jobs:
            return {"status": "complete", "completed_tasks": []}
        return executor.execute(plan, context)
