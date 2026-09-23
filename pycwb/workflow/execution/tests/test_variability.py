from dataclasses import replace

import numpy as np
import pytest

from pycwb.workflow.execution.cache import FrameCache, merged_requests
from pycwb.workflow.execution.planner import FrameRequest
from pycwb.workflow.execution.resources import MemoryBudget, MemoryMonitor
from pycwb.workflow.execution.settings import ExecutionSettings
from pycwb.workflow.execution.tests.test_execution import budget, config


@pytest.mark.parametrize("duration", [16, 64, 256, 1024])
@pytest.mark.parametrize("rate", [64, 257, 1024])
@pytest.mark.parametrize("capacity_fraction", [0, 0.1, 2])
def test_varied_frame_sizes_rates_channels_and_cache_pressure(
    tmp_path, duration, rate, capacity_fraction
):
    from gwpy.timeseries import TimeSeries

    paths = [tmp_path / f"frame-{i}" for i in range(3)]
    for path in paths:
        path.write_bytes(b"source")
    rng = np.random.default_rng(duration + rate)
    requests = []
    for _ in range(18):
        start = int(rng.integers(0, duration - 1))
        end = int(rng.integers(start + 1, duration + 1))
        requests.append(
            FrameRequest(
                str(paths[int(rng.integers(0, 3))]),
                f"H1:CHANNEL{int(rng.integers(0, 2))}",
                1000 + start,
                1000 + end,
                rate,
                duration * rate * 8,
            )
        )

    def reader(path, channel, start, end):
        values = np.arange(int(start * rate), int(end * rate), dtype=np.float64)
        values *= 1 + int(channel[-1])
        return TimeSeries(values, t0=start, sample_rate=rate, channel=channel)

    capacity = int(duration * rate * 8 * capacity_fraction)
    cache = FrameCache(tmp_path, budget(capacity), reader, max_entries=2)
    try:
        unions = merged_requests((requests,))
        for request in requests:
            provider = cache.acquire((request,), unions)
            try:
                expected = reader(
                    request.path, request.channel, request.start, request.end
                )
                actual = (
                    provider.read(
                        request.path, request.channel, request.start, request.end
                    )
                    if provider.entries
                    else expected.copy()
                )
                np.testing.assert_array_equal(actual.value, expected.value)
                assert actual.t0 == expected.t0
                actual.value[:] = -1
                assert cache.bytes <= capacity
                assert len(cache.entries) <= 2
            finally:
                cache.release(provider)
    finally:
        cache.close()


def test_small_slice_of_large_frame_reserves_full_decode(tmp_path):
    path = tmp_path / "large.gwf"
    path.write_bytes(b"metadata-only")
    request = FrameRequest(str(path), "H1:TEST", 0, 1, 128, 16 * 1024**3)

    def forbidden(*args, **kwargs):
        pytest.fail("A large decoder must not run inside a tiny reservation")

    cache = FrameCache(tmp_path, budget(1024**2), forbidden)
    try:
        assert not cache.acquire((request,)).entries
        assert cache.metrics["bypasses"] == 1
    finally:
        cache.close()


def test_larger_analysis_estimates_reduce_admitted_workers(monkeypatch):
    from pycwb.workflow.execution import resources

    monkeypatch.setattr(resources, "available_memory", lambda: 4 * 1024**3)
    base = ExecutionSettings.from_config(
        config(memory_limit="4GiB", cache_limit="1GiB")
    )
    counts = [
        MemoryBudget.resolve(replace(base, worker_memory=size), 8).workers
        for size in (256 * 1024**2, 512 * 1024**2, 2 * 1024**3)
    ]
    assert counts == sorted(counts, reverse=True)
    assert counts[-1] < counts[0]
    with pytest.raises(MemoryError):
        MemoryBudget.resolve(base, 1, input_peak=10 * 1024**3)


def test_unpredicted_peak_triggers_pressure_callback(monkeypatch):
    from pycwb.workflow.execution import resources

    monkeypatch.setattr(resources, "process_tree_memory", lambda: (100, 80))
    calls = []
    monitor = MemoryMonitor(lambda: 30, limit=100)

    def stop():
        calls.append(True)
        monitor._stop.set()

    monitor.on_pressure = stop
    monitor._run()
    assert monitor.exceeded
    assert monitor.peak_accounted == 110
    assert calls == [True]


def test_external_pressure_triggers_callback_below_process_limit(monkeypatch):
    from pycwb.workflow.execution import resources

    monkeypatch.setattr(resources, "process_tree_memory", lambda: (100, 80))
    monkeypatch.setattr(resources, "available_memory", lambda: 10)
    monitor = MemoryMonitor(lambda: 0, limit=1000, headroom=20)
    monitor.on_pressure = monitor._stop.set
    monitor._run()
    assert monitor.exceeded
    assert monitor.peak_accounted < monitor.limit
    assert monitor.min_available < monitor.headroom


def test_disjoint_workload_fills_bounded_batches():
    from pycwb.workflow.execution.planner import prepare_plan
    from pycwb.workflow.execution.tests.test_execution import job

    jobs = [job(i + 1, f"source-{i}.gwf") for i in range(10003)]
    cfg = config(batch_size=31)
    plan = prepare_plan(jobs, cfg, ExecutionSettings.from_config(cfg))
    assert plan.order == tuple(range(len(jobs)))
    assert len(plan.batches) == (len(jobs) + 30) // 31
    assert all(len(batch) == 31 for batch in plan.batches[:-1])
