"""Experimental bounded, cross-job CPU preparation and analysis pipeline.

Only deterministic catalog-only background is accepted. Private adapters reuse
existing scientific functions; these experiment boundaries are not production APIs.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from pycwb.workflow.execution.planner import frame_requests, write_document
from pycwb.workflow.execution.resources import MemoryMonitor, available_memory

GiB = 1024**3


class BoundedWriter:
    """Reject oversized serialized payloads before writing beyond the quota."""

    def __init__(self, stream: io.BufferedWriter, limit: int):
        self.stream, self.limit = stream, limit

    def write(self, data: Any) -> int:
        if self.stream.tell() + memoryview(data).nbytes > self.limit:
            raise MemoryError("Prepared artifact exceeds its byte reservation")
        return self.stream.write(data)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.stream, name)


class Prepared(Exception):
    """End native preparation at an explicitly selected stage boundary."""


def source_identity(job: Any) -> list:
    """Record identities independently of cache/worker lifetime."""
    from pycwb.workflow.execution.cache import fingerprint, local_source

    return [
        (r.path, local_source(r.path), fingerprint(local_source(r.path)))
        for r in frame_requests(job)
    ]


def validate_background(config: Any, jobs: list) -> None:
    """Refuse scientific paths not covered by the experiment adapter."""
    if any(j.injections or j.noise or not j.frames for j in jobs):
        raise ValueError(
            "Pipeline experiments require deterministic frame-backed background"
        )
    for name in (
        "cwb_compare",
        "save_waveform",
        "plot_waveform",
        "plot_trigger",
        "save_sky_map",
        "plot_sky_map",
    ):
        if getattr(config, name, False):
            raise ValueError(f"Unsupported pipeline experiment output: {name}")


def _setup_functions(coherence: Any, td: Any) -> tuple[Any, Any]:
    """Allow serial setup inside a resource-constrained preparation process."""
    if os.environ.get("PYCWB_GPU_OVERLAP_SETUP", "1") != "1":
        return coherence, td

    from pycwb.modules.background_cuda.setup_overlap import OverlappedSetup

    overlap = OverlappedSetup(coherence, td)
    return overlap.setup_coherence, overlap.build_td_inputs_cache


def prepare(task: dict) -> None:
    """Run unchanged CPU functions, publishing a bounded stage checkpoint."""
    import jax
    import joblib

    from pycwb.modules.background_cuda.conditioning_parallel import data_conditioning
    from pycwb.modules.background_cuda.processor import _specialize
    from pycwb.modules.background_cuda.read_parallel import read_from_job_segment
    from pycwb.modules.background_cuda.setup_parallel import setup_coherence
    from pycwb.modules.background_cuda.td_setup_parallel import build_td_inputs_cache
    from pycwb.workflow.subflow import process_job_segment_native as native

    if any(device.platform != "cpu" for device in jax.devices()):
        raise RuntimeError("Preparation must run in a CPU-only process")
    identity = source_identity(task["job"])
    started = time.time()
    path = Path(task["artifact"])

    def checkpoint(value: Any) -> None:
        temporary = path.with_suffix(".partial")
        with temporary.open("wb") as stream:
            joblib.dump(value, BoundedWriter(stream, task["payload_limit"]), compress=0)
        if source_identity(task["job"]) != identity:
            raise RuntimeError("Input source changed during preparation")
        os.replace(temporary, path)
        write_document(
            path.with_suffix(".json"),
            {
                "token": task["token"],
                "stage": task["stage"],
                "job": task["job"].index,
                "source": identity,
                "bytes": path.stat().st_size,
                "started": started,
                "finished": time.time(),
            },
        )
        raise Prepared

    def read(config: Any, job: Any) -> Any:
        value = read_from_job_segment(config, job)
        if task["stage"] == "read":
            checkpoint(value)
        return value

    def condition(config: Any, data: Any) -> Any:
        value = data_conditioning(config, data)
        if task["stage"] == "conditioned":
            checkpoint(value)
        return value

    def capture(context: Any, output: Any, skip_lags: Any) -> None:
        checkpoint((context, output))

    coherence, td = _setup_functions(setup_coherence, build_td_inputs_cache)
    implementation = _specialize(
        native.process_job_segment,
        read_from_job_segment=read,
        data_conditioning=condition,
        setup_coherence=coherence,
        build_td_inputs_cache=td,
    )
    try:
        with jax.default_device(jax.devices("cpu")[0]):
            implementation(
                task["directory"],
                task["config"],
                task["job"],
                catalog_file=task["catalog"],
                compress_json=True,
                lag_processor=capture,
            )
    except Prepared:
        return
    raise RuntimeError("Preparation did not reach the requested checkpoint")


def consume(task: dict) -> None:
    """Consume owned/COW prepared data through the existing analysis/output path."""
    import jax
    import joblib

    from pycwb.modules.background_cuda import processor as gpu
    from pycwb.workflow.subflow import process_job_segment_native as native
    from pycwb.workflow.subflow import process_job_segment_parallel as cpu

    path = Path(task["artifact"])
    manifest = json.loads(path.with_suffix(".json").read_text())
    expected = json.loads(json.dumps(source_identity(task["job"])))
    if (
        manifest["token"] != task["token"]
        or manifest["stage"] != task["stage"]
        or manifest["job"] != task["job"].index
        or manifest["source"] != expected
        or path.stat().st_size != manifest["bytes"]
    ):
        raise ValueError("Prepared input identity mismatch")
    if task["backend"] == "gpu":
        if not jax.config.x64_enabled:
            raise RuntimeError("GPU consumption requires JAX_ENABLE_X64=1")
        jax.devices("gpu")
    started = time.time()

    def analyze(context: Any, output: Any, skip_lags: Any) -> None:
        write_document(path.with_suffix(".analysis"), {"token": task["token"]})
        (gpu._process_lags if task["backend"] == "gpu" else cpu.process_lags)(
            context, output, skip_lags=skip_lags
        )

    with jax.default_device(jax.devices("cpu")[0]):
        value = joblib.load(path, mmap_mode="c")
        if task["stage"] == "full":
            context, output = value
            analyze(context, output, skip_lags=None)
        else:
            # Restricted background adapter: resume at the selected boundary.
            # No global module mutation; GPU kernels and output functions retain
            # their original bindings. Unsupported trial paths were rejected.
            bindings = {"read_from_job_segment": lambda *_: value}
            if task["stage"] == "conditioned":
                strains, noise_rms = value
                bindings = {
                    "read_from_job_segment": lambda *_: [None] * len(task["job"].ifos),
                    "check_and_resample_py": lambda series, *_: series,
                    "data_conditioning": lambda *_: (strains, noise_rms),
                }
            process = gpu._specialize(native.process_job_segment, **bindings)
            if task["backend"] == "gpu":
                process = gpu._specialize(
                    gpu.process_job_segment,
                    native=SimpleNamespace(process_job_segment=process),
                    _process_lags=analyze,
                )
                process(
                    task["directory"],
                    task["config"],
                    task["job"],
                    catalog_file=task["catalog"],
                    compress_json=True,
                )
            else:
                process(
                    task["directory"],
                    task["config"],
                    task["job"],
                    catalog_file=task["catalog"],
                    compress_json=True,
                    lag_processor=analyze,
                )
    write_document(
        path.with_suffix(".consumed.json"),
        {"started": started, "finished": time.time()},
    )


def worker(role: str, filename: str, *, publish_completion: bool = False) -> None:
    """Child entry point; environment isolation precedes any JAX import."""
    import joblib

    from pycwb.modules.logger import logger_init

    task = joblib.load(filename)
    logger_init(
        log_file=str(
            Path(task["directory"]) / "log" / f"{role}_{task['job'].index}.log"
        ),
        log_level="INFO",
    )
    (prepare if role == "prepare" else consume)(task)
    if publish_completion:
        import gc

        gc.collect()
        write_document(
            Path(task["artifact"]).with_suffix(f".{role}.done"),
            {"token": task["token"]},
        )


def stop(process: subprocess.Popen) -> None:
    """Terminate a failed stage and every inherited process-group member."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            continue


def run_pipeline(
    directory: str,
    config: Any,
    jobs: list,
    catalog: str,
    stage: str,
    *,
    backend: str = "gpu",
    overlap: bool = True,
    memory_limit: int = 28 * GiB,
    producer_memory: int = 8 * GiB,
    consumer_memory: int = 12 * GiB,
    payload_limit: int = 3 * GiB,
    headroom: int = GiB,
    producer_cpus: str | None = None,
    consumer_cpus: str | None = None,
    persistent: bool = False,
    defer_prepare: bool = False,
) -> dict:
    """Run one producer plus one consumer with at most one job waiting ahead."""
    import tempfile

    import joblib
    import psutil

    validate_background(config, jobs)
    if stage not in {"read", "conditioned", "full"} or backend not in {"gpu", "cpu"}:
        raise ValueError("Unknown stage or backend")
    if (
        min(producer_memory, consumer_memory, payload_limit, memory_limit, headroom)
        <= 0
    ):
        raise ValueError("Pipeline reservations must be positive")
    baseline = psutil.Process().memory_info().rss
    limit = min(memory_limit, baseline + available_memory())
    # The consumer retains its mapping until exit while the producer publishes
    # the next artifact. Reserve both payloads, including their file-backed pages.
    required = baseline + producer_memory + consumer_memory + headroom
    if required >= limit:
        raise MemoryError(
            f"Pipeline reservations require {required} bytes, only {limit} available"
        )
    requested_payload_limit = payload_limit
    payload_limit = min(payload_limit, (limit - required) // 2)
    for job in jobs:
        requests = frame_requests(job)
        input_peak = 3 * sum(
            sorted((r.decode_bytes for r in requests), reverse=True)[:2]
        ) + 2 * sum(r.estimated_bytes for r in requests)
        if input_peak > producer_memory:
            raise MemoryError("Full-frame decode estimate exceeds producer reservation")
    started = time.time()
    events, active, streams, workers = [], {}, [], {}
    next_prepare = next_consume = 0
    ready = None
    error = None
    peak_payload = 0
    with tempfile.TemporaryDirectory(prefix=".pipeline-", dir=directory) as scratch:
        scratch = Path(scratch)
        tasks = []
        for index, job in enumerate(jobs):
            task = {
                "job": job,
                "config": config,
                "directory": directory,
                "catalog": catalog,
                "stage": stage,
                "backend": backend,
                "payload_limit": payload_limit,
                "artifact": str(scratch / f"{index}.joblib"),
                "token": uuid.uuid4().hex,
            }
            filename = scratch / f"task-{index}.pkl"
            joblib.dump(task, filename)
            tasks.append((task, filename))

        def payload_bytes() -> int:
            total = 0
            for pattern in ("*.joblib", "*.partial"):
                for path in scratch.glob(pattern):
                    try:
                        total += path.stat().st_size
                    except FileNotFoundError:
                        pass  # Publication/deletion may race with the sampler.
            return total

        def pressure() -> None:
            processes = set(workers.values()) | {p for p, _ in list(active.values())}
            for process in processes:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass

        monitor = MemoryMonitor(
            payload_bytes,
            limit=limit - headroom,
            headroom=headroom,
            on_pressure=pressure,
        )
        monitor.start()

        def launch(role: str, index: int) -> None:
            env = dict(os.environ)
            if role == "prepare":
                env.update(
                    JAX_PLATFORMS="cpu",
                    CUDA_VISIBLE_DEVICES="",
                    PYCWB_GPU_WDM_PREFILTER="0",
                    PYCWB_GPU_MAX_ENERGY="0",
                )
            else:
                if backend == "cpu":
                    env.update(JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="")
                env["PYCWB_GPU_READ_WORKERS"] = "1"
                if stage == "conditioned":
                    env["PYCWB_GPU_CONDITION_WORKERS"] = "1"
            command = [
                sys.executable,
                "-m",
                "benchmark.job_pipeline",
                role,
                *(["--serve"] if persistent else [str(tasks[index][1])]),
            ]
            cpus = producer_cpus if role == "prepare" else consumer_cpus
            if cpus:
                command = ["taskset", "-c", cpus, *command]
            process = workers.get(role) if persistent else None
            if process is None:
                stream = (
                    Path(directory) / "log" / f"{role}_{jobs[index].index}.console.log"
                ).open("w")
                streams.append(stream)
                process = subprocess.Popen(
                    command,
                    env=env,
                    stdin=subprocess.PIPE if persistent else None,
                    text=True,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                if persistent:
                    workers[role] = process
            if persistent:
                process.stdin.write(str(tasks[index][1]) + "\n")
                process.stdin.flush()
            active[role] = (process, index)
            events.append(
                {"event": "start", "role": role, "task": index, "time": time.time()}
            )

        try:
            while next_consume < len(jobs) or active:
                if monitor.exceeded:
                    raise MemoryError(
                        "Pipeline observed memory reached its safety margin"
                    )
                for role, process in list(workers.items()):
                    if process.poll() is not None:
                        raise RuntimeError(
                            f"Persistent {role} process exited unexpectedly"
                        )
                for role, (process, index) in list(active.items()):
                    code = process.poll()
                    task = tasks[index][0]
                    path = Path(task["artifact"])
                    done = path.with_suffix(f".{role}.done")
                    if code is None and not (persistent and done.exists()):
                        continue
                    if code:
                        raise RuntimeError(
                            f"{role} failed for job {jobs[index].index} with status {code}"
                        )
                    del active[role]
                    if persistent:
                        if json.loads(done.read_text())["token"] != task["token"]:
                            raise RuntimeError("Stage completion identity mismatch")
                        done.unlink()
                    events.append(
                        {
                            "event": "finish",
                            "role": role,
                            "task": index,
                            "time": time.time(),
                        }
                    )
                    if role == "prepare":
                        ready = index
                    else:
                        for file in (
                            path,
                            path.with_suffix(".json"),
                            path.with_suffix(".consumed.json"),
                            path.with_suffix(".analysis"),
                        ):
                            file.unlink(missing_ok=True)
                peak_payload = max(peak_payload, payload_bytes())
                if ready is not None and "consume" not in active:
                    assert ready == next_consume
                    launch("consume", ready)
                    next_consume += 1
                    ready = None
                if (
                    next_prepare < len(jobs)
                    and "prepare" not in active
                    and ready is None
                    and (overlap or "consume" not in active)
                    and (
                        not defer_prepare
                        or "consume" not in active
                        or Path(tasks[active["consume"][1]][0]["artifact"])
                        .with_suffix(".analysis")
                        .exists()
                    )
                ):
                    launch("prepare", next_prepare)
                    next_prepare += 1
                time.sleep(0.1)
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            processes = set(workers.values()) | {p for p, _ in list(active.values())}
            for process in processes:
                stop(process)
                if getattr(process, "stdin", None) is not None:
                    process.stdin.close()
            monitor.close()
            for stream in streams:
                stream.close()
            summary = {
                "stage": stage,
                "backend": backend,
                "overlap": overlap,
                "persistent": persistent,
                "defer_prepare": defer_prepare,
                "seconds": time.time() - started,
                "status": "failed" if error else "complete",
                "error": error,
                "events": events,
                "peak_payload": peak_payload,
                "peak_pss": monitor.peak_pss,
                "peak_accounted": monitor.peak_accounted,
                "swap_growth": monitor.peak_swap - monitor.initial_swap,
                "reservations": {
                    "limit": limit,
                    "producer": producer_memory,
                    "consumer": consumer_memory,
                    "ready_payload": payload_limit,
                    "requested_payload_limit": requested_payload_limit,
                    "retained_payloads": 2,
                    "headroom": headroom,
                },
            }
            write_document(Path(directory) / "pipeline.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=("prepare", "consume"))
    parser.add_argument("task", nargs="?")
    parser.add_argument("--serve", action="store_true")
    args = parser.parse_args()
    if args.serve:
        for filename in sys.stdin:
            worker(args.role, filename.rstrip("\n"), publish_completion=True)
    elif args.task:
        worker(args.role, args.task)
    else:
        parser.error("task is required without --serve")
