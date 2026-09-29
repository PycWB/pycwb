"""Small processors exercising the real reader, spawn, and output protocols."""

import os
from pathlib import Path

import numpy as np


def read_and_record(working_dir, config, job, *, queue, input_provider=None, **kwargs):
    from pycwb.modules.read_data.read_data import read_from_job_segment

    data = read_from_job_segment(config, job, input_provider=input_provider)
    np.savez(
        Path(working_dir) / f"input-{job.index}-{job.trial_idx}.npz",
        **{ifo: series.data for ifo, series in zip(job.ifos, data)},
    )
    # Mutating this job must not affect later readers of the same cache entry.
    for series in data:
        series.data[:] = -12345
    queue.put(
        {
            "type": "progress",
            "job_id": job.index,
            "trial_idx": job.trial_idx,
            "lag_idx": 0,
            "n_triggers": 0,
            "livetime": job.duration,
        }
    )


read_and_record.supports_input_provider = True


def crash(working_dir, config, job, **kwargs):
    os._exit(17)


def slow_shutdown(working_dir, config, job, **kwargs):
    import time
    from multiprocessing.util import Finalize

    Finalize(None, time.sleep, args=(6,), exitpriority=0)


def failed_shutdown(working_dir, config, job, **kwargs):
    from multiprocessing.util import Finalize

    Finalize(None, os._exit, args=(17,), exitpriority=0)


def bad_output(working_dir, config, job, *, queue, **kwargs):
    queue.put({"type": "invalid"})


def create_reverse_planner():
    from pycwb.workflow.execution.planner import ExecutionPlan, frame_requests

    class ReversePlanner:
        def plan(self, jobs, config, settings):
            return ExecutionPlan(
                (tuple(reversed(range(len(jobs)))),),
                tuple(frame_requests(job) for job in jobs),
                tuple(job.index for job in jobs),
            )

    return ReversePlanner()
