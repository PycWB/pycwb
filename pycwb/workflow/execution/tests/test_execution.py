from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pycwb.types.job import FrameFile, WaveSegment
from pycwb.workflow.execution.cache import FrameCache, merged_requests, sample_index
from pycwb.workflow.execution.executor import ExecutionContext, execute_jobs
from pycwb.workflow.execution.planner import FrameRequest, prepare_plan
from pycwb.workflow.execution.resources import MemoryBudget, cgroup_headroom
from pycwb.workflow.execution.settings import ExecutionSettings, byte_size
from pycwb.workflow.execution.tests.helpers import bad_output, crash, read_and_record
from pycwb.workflow.execution.writer import OutputWriter


def job(index, path, start=100, end=104, rate=128, shift=None):
    return WaveSegment(
        index,
        ["H1"],
        start,
        end,
        rate,
        0,
        channels=["H1:TEST"],
        frames=[FrameFile("H1", str(path), 100, 16)],
        shift=shift,
    )


def config(**overrides):
    execution = {
        "profile": "scalable",
        "memory_limit": "3GiB",
        "worker_memory": "256MiB",
        "headroom": "64MiB",
        "cache_limit": "1MiB",
        "message_limit": "1MiB",
        "batch_size": 4,
    }
    execution.update(overrides)
    return SimpleNamespace(execution=execution, nproc=1, job_per_worker=2, segEdge=0)


@pytest.mark.parametrize(
    "value, expected", [(12, 12), ("1GiB", 1024**3), ("1.5 MB", 1500000), ("0B", 0)]
)
def test_byte_units(value, expected):
    assert byte_size(value) == expected


@pytest.mark.parametrize("value", [True, -1, 1.2, "32", "nanGiB", "-2MB", None])
def test_invalid_byte_units(value):
    with pytest.raises(ValueError):
        byte_size(value)


def test_settings_legacy_and_validation():
    assert not ExecutionSettings.from_config(SimpleNamespace()).enabled
    for values in (
        {"profile": "typo"},
        {"cache_limit": None},
        {"batch_size": 0},
        {"cores": True},
        {"unknown": 1},
        {"executor": "missingdot"},
    ):
        with pytest.raises(ValueError):
            ExecutionSettings.from_config(SimpleNamespace(execution=values))


def test_planner_groups_nonconsecutive_ids_without_changing_jobs():
    jobs = [job(10, "a"), job(20, "b"), job(30, "a"), job(40, "b")]
    cfg = config(batch_size=2)
    settings = ExecutionSettings.from_config(cfg)
    plan = prepare_plan(jobs, cfg, settings)
    assert plan.batches == ((0, 2), (1, 3))
    assert plan.job_ids == (10, 20, 30, 40)
    assert prepare_plan(jobs, cfg, settings) == plan
    assert [j.index for j in jobs] == [10, 20, 30, 40]


def test_plugin_and_invalid_plan():
    cfg = config(
        planner="pycwb.workflow.execution.tests.helpers.create_reverse_planner"
    )
    jobs = [job(1, "a"), job(2, "b")]
    plan = prepare_plan(jobs, cfg, ExecutionSettings.from_config(cfg))
    assert plan.order == (1, 0)
    with pytest.raises(ValueError, match="exactly once"):
        replace(plan, batches=((0, 0),)).validate(jobs)
    with pytest.raises(ValueError, match="input requests"):
        replace(plan, requests=((), ())).validate(jobs)


def test_superlag_physical_requests_and_disjoint_intervals():
    jobs = [job(1, "a", start=104, end=108, shift=[4]), job(2, "a", start=112, end=116)]
    cfg = config()
    plan = prepare_plan(jobs, cfg, ExecutionSettings.from_config(cfg))
    assert plan.requests[0][0].start == 100
    assert len(merged_requests(plan.requests)) == 2


def budget(cache=32768):
    return MemoryBudget(1024**3, 0, 0, 0, cache, 1)


def fake_reader(filename, channel, start, end):
    from gwpy.timeseries import TimeSeries

    return TimeSeries(
        np.arange(round(start * 128), round(end * 128), dtype=float),
        t0=start,
        sample_rate=128,
        channel=channel,
    )


def test_cached_union_owned_samples_and_epoch(tmp_path):
    source = tmp_path / "frame.gwf"
    source.write_bytes(b"identity")
    first = FrameRequest(str(source), "H1:TEST", 100, 104, 128)
    second = replace(first, start=103, end=108)
    planned = merged_requests(((first,), (second,)))
    cache = FrameCache(tmp_path, budget(), fake_reader)
    try:
        one = cache.acquire((first,), planned)
        data = one.read(first.path, first.channel, 100, 104)
        np.testing.assert_array_equal(data.value, np.arange(12800, 13312))
        data.value[:] = -1
        cache.release(one)
        two = cache.acquire((second,), planned)
        data2 = two.read(second.path, second.channel, 103, 108)
        np.testing.assert_array_equal(data2.value, np.arange(13184, 13824))
        assert float(data2.t0.value) == 103
        assert cache.metrics["reads"] == 1
        assert cache.metrics["hits"] == 1
        assert cache.bytes <= cache.budget.cache
        cache.release(two)
    finally:
        cache.close()
    assert not list(tmp_path.glob(".frame-cache-*"))


def test_cache_pin_eviction_and_low_memory_bypass(tmp_path):
    paths = [tmp_path / name for name in ("a", "b")]
    for path in paths:
        path.write_bytes(b"x")
    requests = [FrameRequest(str(path), "H1:TEST", 100, 104, 128) for path in paths]
    cache = FrameCache(tmp_path, budget(4096), fake_reader)
    first = cache.acquire((requests[0],))
    other = cache.acquire((requests[1],))
    assert not other.entries  # Cannot evict pinned data.
    with pytest.raises(RuntimeError, match="leases"):
        cache.close()
    cache.release(first)
    other = cache.acquire((requests[1],))
    assert len(other.entries) == 1
    assert cache.metrics["evictions"] == 1
    cache.release(other)
    cache.close()


def test_changed_source_rejected_and_sample_alignment(tmp_path):
    source = tmp_path / "frame"
    source.write_bytes(b"x")
    request = FrameRequest(str(source), "H1:TEST", 100, 104, 128)
    cache = FrameCache(tmp_path, budget(), fake_reader)
    provider = cache.acquire((request,))
    source.write_bytes(b"different")
    with pytest.raises(RuntimeError, match="changed"):
        provider.read(request.path, request.channel, request.start, request.end)
    cache.release(provider)
    cache.close()
    with pytest.raises(ValueError, match="aligned"):
        sample_index(0.1, 128)


def test_cgroup_parent_limits(tmp_path):
    root = tmp_path / "cgroup"
    leaf = root / "parent" / "child"
    leaf.mkdir(parents=True)
    membership = tmp_path / "membership"
    membership.write_text("0::/parent/child\n")
    for directory, limit, current in [
        (root, "1000", "100"),
        (leaf.parent, "500", "400"),
        (leaf, "max", "20"),
    ]:
        (directory / "memory.max").write_text(limit)
        (directory / "memory.current").write_text(current)
    assert cgroup_headroom(root, membership) == 100


def test_memory_reservations_clamp_workers_cache(monkeypatch):
    from pycwb.workflow.execution import resources

    monkeypatch.setattr(resources, "available_memory", lambda: 2 * 1024**3)
    settings = ExecutionSettings.from_config(
        config(memory_limit="2GiB", worker_memory="512MiB", cache_limit="8GiB")
    )
    result = MemoryBudget.resolve(settings, 20)
    assert 1 <= result.workers < 4
    assert (
        result.baseline
        + result.headroom
        + result.workers * result.worker
        + 3 * result.cache
        <= result.limit
    )
    with pytest.raises(MemoryError, match="cannot admit"):
        MemoryBudget.resolve(replace(settings, memory_limit=1), 1)


@pytest.fixture
def real_frames(tmp_path):
    from gwpy.timeseries import TimeSeries

    path = tmp_path / "test.gwf"
    samples = np.random.default_rng(13).normal(size=16 * 128)
    TimeSeries(samples, t0=100, sample_rate=128, channel="H1:TEST").write(
        str(path), format="gwf"
    )
    return path, samples


def make_context(tmp_path, jobs, cfg, processor=read_and_record):
    from pycwb.modules.catalog.catalog import Catalog

    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "log").mkdir(exist_ok=True)
    catalog = tmp_path / "catalog.parquet"
    Catalog.create(str(catalog), cfg, jobs)
    return ExecutionContext(jobs, cfg, processor, str(tmp_path), str(catalog))


def test_real_spawn_gwf_reuse_mutation_and_resume(tmp_path, real_frames):
    from pycwb.modules.catalog.catalog import Catalog

    source, samples = real_frames
    jobs = [job(1, source), job(2, source, start=102, end=106)]
    context = make_context(tmp_path / "run", jobs, config())
    summary = execute_jobs(context)
    assert summary["cache"]["reads"] == 1
    assert summary["cache"]["hits"] == 1
    for j in jobs:
        with np.load(Path(context.working_dir) / f"input-{j.index}-0.npz") as actual:
            np.testing.assert_array_equal(
                actual["H1"],
                samples[(j.analyze_start - 100) * 128 : (j.analyze_end - 100) * 128],
            )
        assert Catalog.open(context.catalog_file).get_completed_lags(j.index) == {
            0: {0}
        }
    assert not list(Path(context.working_dir).glob(".frame-cache-*"))
    source.unlink()  # Completed work must not even inspect inputs on resume.
    assert execute_jobs(context)["completed_tasks"] == []


@pytest.mark.parametrize(
    "processor, message",
    [(crash, "without completion"), (bad_output, "Unknown output")],
)
def test_worker_and_writer_failure_cleanup(tmp_path, processor, message):
    from pycwb.modules.catalog.catalog import Catalog

    jobs = [WaveSegment(1, ["H1"], 100, 104, 128, 0)]
    context = make_context(tmp_path / "run", jobs, config(), processor)
    with pytest.raises((RuntimeError, ValueError), match=message):
        execute_jobs(context)
    assert not list(Path(context.working_dir).glob(".frame-cache-*"))
    assert Catalog.open(context.catalog_file).get_completed_lags(1) == {}


def test_writer_does_not_commit_progress_after_failed_flush(monkeypatch):
    from pycwb.modules.catalog.catalog import Catalog

    progress = []

    def fail(triggers):
        raise OSError("disk full")

    monkeypatch.setattr(
        Catalog,
        "open",
        lambda _: SimpleNamespace(
            add_triggers=fail, add_lag_progress=lambda **kw: progress.append(kw)
        ),
    )
    writer = OutputWriter(None, "unused", 1000)
    writer.handle({"type": "trigger", "trigger": object()}, 1)
    with pytest.raises(OSError, match="disk full"):
        writer.handle({"type": "progress", "job_id": 1}, 1)
    assert not progress


def test_concurrent_runner_cannot_clean_live_catalog_locks(tmp_path, monkeypatch):
    from filelock import FileLock, Timeout

    from pycwb.workflow import batch

    context = make_context(tmp_path / "run", [], config())
    calls = []
    monkeypatch.setattr(batch, "_cleanup_stale_lock", calls.append)
    lock = Path(context.working_dir) / ".execution-catalog.lock"
    with FileLock(str(lock)), pytest.raises(Timeout):
        execute_jobs(context)
    assert calls == []
