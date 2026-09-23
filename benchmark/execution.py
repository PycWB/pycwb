"""Reproducible GWF I/O/cache experiment using the real supervised runtime.

Run each mode in a fresh Python process. This measures input handling and worker
overhead, not scientific-analysis speedup. Use --manifest for existing frames;
otherwise create deterministic synthetic GWF channels with overlapping jobs.
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from pycwb.types.job import FrameFile, WaveSegment
from pycwb.workflow.execution.executor import ExecutionContext, execute_jobs


def input_probe(working_dir, config, job, *, input_provider=None, queue, **kwargs):
    from pycwb.modules.read_data.read_data import read_from_job_segment

    data = read_from_job_segment(config, job, input_provider=input_provider)
    scratch = np.ones(config.scratch_mib * 1024**2 // 8, dtype=np.float64)
    result = {}
    for ifo, series in zip(job.ifos, data):
        result[ifo] = {
            "sha256": hashlib.sha256(np.ascontiguousarray(series.data)).hexdigest(),
            "start": series.start_time,
            "rate": series.sample_rate,
            "samples": len(series.data),
            "dtype": str(series.data.dtype),
        }
        series.data[:] = -12345  # Exercise cache isolation for subsequent jobs.
    # Touch a configurable amount of private working memory, independently of
    # frame size. This is a stress model, not a prediction of pixel-selection RAM.
    scratch *= 1.01
    del scratch
    (Path(working_dir) / f"input-{job.index}.json").write_text(
        json.dumps(result, sort_keys=True)
    )
    queue.put(
        {
            "type": "progress",
            "job_id": job.index,
            "trial_idx": 0,
            "lag_idx": 0,
            "n_triggers": 0,
            "livetime": job.duration,
        }
    )


input_probe.supports_input_provider = True


def synthetic_jobs(directory):
    from gwpy.timeseries import TimeSeries

    directory.mkdir(parents=True, exist_ok=True)
    paths = [directory / f"frame-{i}.gwf" for i in range(2)]
    for i, path in enumerate(paths):
        if not path.exists():
            data = np.random.default_rng(100 + i).normal(size=512 * 4096)
            TimeSeries(data, t0=1000000000, sample_rate=4096, channel="H1:TEST").write(
                str(path), format="gwf"
            )
    jobs = []
    for i in range(8):
        start = 1000000000 + (i // 2) * 64
        jobs.append(
            WaveSegment(
                i + 1,
                ["H1"],
                start,
                start + 256,
                4096,
                0,
                channels=["H1:TEST"],
                frames=[FrameFile("H1", str(paths[i % 2]), 1000000000, 512)],
            )
        )
    return jobs


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--mode", choices=["off", "auto", "batch", "tiny"], required=True
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--scratch-mib", type=int, default=0)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Use a fresh output directory: {output}")
    if args.manifest:
        import pyarrow.parquet as pq
        from dacite import Config, from_dict

        jobs = [
            from_dict(
                WaveSegment, json.loads(row["job_json"]), config=Config(cast=[tuple])
            )
            for row in pq.read_table(args.manifest).to_pylist()
        ]
        for job in jobs:
            job.lag_size, job.lag_array, job.lag_file = 1, None, None
    else:
        jobs = synthetic_jobs(output.parent / "synthetic-inputs")
    output.mkdir(parents=True)
    (output / "log").mkdir()
    config = SimpleNamespace(
        nproc=1,
        segEdge=jobs[0].seg_edge,
        job_per_worker=4,
        scratch_mib=args.scratch_mib,
        execution={
            "profile": "scalable",
            "preload": "auto" if args.mode == "tiny" else args.mode,
            "memory_limit": "12GiB" if args.manifest else "3GiB",
            "worker_memory": "1GiB" if args.manifest else "512MiB",
            "headroom": "256MiB",
            "message_limit": "8MiB",
            "cache_limit": "1MiB"
            if args.mode == "tiny"
            else "2GiB"
            if args.manifest
            else "64MiB",
            "batch_size": 8,
            "cores": args.workers,
        },
    )
    from pycwb.modules.catalog.catalog import Catalog

    catalog = output / "catalog.parquet"
    Catalog.create(str(catalog), config, jobs)
    before = os.times()
    start = time.perf_counter()
    summary = execute_jobs(
        ExecutionContext(
            jobs, config, input_probe, str(output), str(catalog), workers=args.workers
        )
    )
    after = os.times()
    summary.update(
        mode=args.mode,
        workers=args.workers,
        jobs=len(jobs),
        complete_seconds=time.perf_counter() - start,
        cpu_seconds=sum(after[:4]) - sum(before[:4]),
        input_digests={
            str(job.index): json.loads((output / f"input-{job.index}.json").read_text())
            for job in jobs
        },
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(
        json.dumps({k: v for k, v in summary.items() if k != "input_digests"}, indent=2)
    )


if __name__ == "__main__":
    main()
