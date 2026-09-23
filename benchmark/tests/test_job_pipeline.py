"""Resource and lifecycle contracts, independent of scientific benchmark inputs."""

import io
import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pytest

from benchmark import job_pipeline as pipeline


@pytest.mark.parametrize("setting", ["0", "1", None])
def test_setup_overlap_can_be_disabled_without_changing_results(monkeypatch, setting):
    import threading

    if setting is None:
        monkeypatch.delenv("PYCWB_GPU_OVERLAP_SETUP", raising=False)
    else:
        monkeypatch.setenv("PYCWB_GPU_OVERLAP_SETUP", setting)
    concurrent = setting != "0"
    td_started = threading.Event()
    calls = []
    config, strains, coherence_result, td_result = (object() for _ in range(4))

    def coherence(actual_config, actual_strains, *, nRMS):
        assert (actual_config, actual_strains, nRMS) == (config, strains, "noise")
        if concurrent:
            assert td_started.wait(timeout=5)
        else:
            assert not td_started.is_set()
        calls.append("coherence")
        return coherence_result

    def td(actual_config, actual_strains):
        assert (actual_config, actual_strains) == (config, strains)
        assert (threading.current_thread() is threading.main_thread()) != concurrent
        calls.append("td")
        td_started.set()
        return td_result

    setup, build_td = pipeline._setup_functions(coherence, td)
    assert setup(config, strains, nRMS="noise") is coherence_result
    assert build_td(config, strains) is td_result
    assert sorted(calls) == ["coherence", "td"]


@pytest.mark.parametrize("size", [0, 1, 127, 4097, 100003])
def test_serialization_quota_and_roundtrip(size):
    data = np.arange(size, dtype=np.float64)
    reference = io.BytesIO()
    joblib.dump(data, reference, compress=0)
    required = reference.tell()
    target = io.BytesIO()
    joblib.dump(data, pipeline.BoundedWriter(target, required), compress=0)
    target.seek(0)
    np.testing.assert_array_equal(joblib.load(target), data)
    target = io.BytesIO()
    with pytest.raises(MemoryError, match="byte reservation"):
        joblib.dump(data, pipeline.BoundedWriter(target, required - 1), compress=0)
    assert target.tell() <= required - 1


def test_rejected_write_does_not_partially_write():
    stream = io.BytesIO()
    writer = pipeline.BoundedWriter(stream, 5)
    writer.write(b"123")
    with pytest.raises(MemoryError):
        writer.write(b"456")
    assert stream.getvalue() == b"123"


def test_source_change_after_publication_rejects_consumption(monkeypatch, tmp_path):
    source = tmp_path / "frame.gwf"
    source.write_bytes(b"original source")
    monkeypatch.setattr(
        pipeline, "frame_requests", lambda _: [SimpleNamespace(path=str(source))]
    )
    job = SimpleNamespace(index=7)
    artifact = tmp_path / "prepared.joblib"
    joblib.dump(np.arange(7), artifact)
    task = {
        "artifact": str(artifact),
        "job": job,
        "token": "owner",
        "stage": "read",
        "backend": "cpu",
    }
    artifact.with_suffix(".json").write_text(
        json.dumps(
            {
                "token": task["token"],
                "stage": task["stage"],
                "job": job.index,
                "source": pipeline.source_identity(job),
                "bytes": artifact.stat().st_size,
            }
        )
    )
    source.write_bytes(b"a different source generation")
    with pytest.raises(ValueError, match="Prepared input identity mismatch"):
        pipeline.consume(task)


@pytest.mark.parametrize("field", ["injections", "noise", "frames"])
def test_unsupported_scientific_paths_rejected(field):
    job = SimpleNamespace(injections=None, noise=None, frames=[1])
    setattr(job, field, [] if field == "frames" else [1])
    with pytest.raises(ValueError, match="deterministic"):
        pipeline.validate_background(SimpleNamespace(), [job])


@pytest.fixture
def harness(monkeypatch, tmp_path):
    """Deterministic child lifetimes exercise scheduling without CUDA or frames."""
    (tmp_path / "log").mkdir()
    state = SimpleNamespace(
        children=[], fail=None, decode_bytes=1, pressure=False, require_analysis=False
    )
    monkeypatch.setattr(pipeline, "available_memory", lambda: 64 * pipeline.GiB)
    monkeypatch.setattr(
        pipeline,
        "time",
        SimpleNamespace(time=__import__("time").time, sleep=lambda _: None),
    )
    monkeypatch.setattr(
        pipeline,
        "frame_requests",
        lambda _: [
            SimpleNamespace(decode_bytes=state.decode_bytes, estimated_bytes=10)
        ],
    )

    class Monitor:
        peak_pss = peak_accounted = peak_swap = initial_swap = 0

        @property
        def exceeded(self):
            return state.pressure and len(state.children) >= 3

        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

        def close(self):
            pass

    class Child:
        def __init__(self, command, **kwargs):
            self.role = command[command.index("benchmark.job_pipeline") + 1]
            self.serving = "--serve" in command
            self.task = None
            self.stdin = (
                SimpleNamespace(
                    write=self.accept_task, flush=lambda: None, close=lambda: None
                )
                if self.serving
                else None
            )
            if not self.serving:
                self.accept_task(command[-1])
            self.pid = 99999999
            self.stopped = False
            self.done = False
            self.environment = kwargs["env"]
            state.children.append(self)

        def accept_task(self, filename):
            self.task = joblib.load(filename.rstrip("\n"))
            self.left = 2 if self.role == "prepare" else 7
            self.analysis_started = False
            if (
                state.require_analysis
                and self.role == "prepare"
                and self.task["job"].index
            ):
                assert any(
                    c.role == "consume" and c.analysis_started for c in state.children
                )

        def poll(self):
            if state.fail == "interrupt":
                raise KeyboardInterrupt
            if self.task is None:
                return None
            self.left -= 1
            if self.role == "consume" and self.left <= 4:
                self.analysis_started = True
                Path(self.task["artifact"]).with_suffix(".analysis").write_text("{}")
            if self.left > 0:
                return None
            if state.fail == (self.role, self.task["job"].index):
                return 1
            if self.role == "prepare":
                Path(self.task["artifact"]).write_bytes(b"ready")
            else:
                assert Path(self.task["artifact"]).read_bytes() == b"ready"
            self.done = True
            if self.serving:
                Path(self.task["artifact"]).with_suffix(
                    f".{self.role}.done"
                ).write_text(json.dumps({"token": self.task["token"]}))
                self.task = None
                return None
            return 0

    monkeypatch.setattr(pipeline, "MemoryMonitor", Monitor)
    monkeypatch.setattr(pipeline.subprocess, "Popen", Child)
    monkeypatch.setattr(pipeline, "stop", lambda p: setattr(p, "stopped", True))

    def run(**kwargs):
        jobs = [
            SimpleNamespace(index=i, injections=None, noise=None, frames=[1])
            for i in range(4)
        ]
        return pipeline.run_pipeline(
            str(tmp_path), SimpleNamespace(), jobs, "unused", "read", **kwargs
        )

    return state, run, tmp_path


@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("persistent", [False, True])
def test_bounded_queue_order_and_cpu_only_producer(harness, overlap, persistent):
    state, run, directory = harness
    summary = run(overlap=overlap, persistent=persistent)
    preparing, consuming, ready = set(), set(), set()
    observed_overlap = False
    completed = []
    for event in summary["events"]:
        index, role = event["task"], event["role"]
        if role == "prepare":
            if event["event"] == "start":
                preparing.add(index)
            else:
                preparing.remove(index)
                ready.add(index)
        elif event["event"] == "start":
            ready.remove(index)
            consuming.add(index)
        else:
            consuming.remove(index)
            completed.append(index)
        assert len(preparing) + len(ready) <= 1
        assert len(consuming) <= 1
        observed_overlap |= bool(preparing and consuming)
    assert observed_overlap == overlap
    assert completed == list(range(4))
    assert len(state.children) == (2 if persistent else 8)
    assert not list(directory.glob(".pipeline-*"))
    for child in state.children:
        if child.role == "prepare":
            assert child.environment["CUDA_VISIBLE_DEVICES"] == ""
            assert child.environment["JAX_PLATFORMS"] == "cpu"


def test_failure_cancels_other_stage_and_cleans_payloads(harness):
    state, run, directory = harness
    state.fail = ("prepare", 1)
    with pytest.raises(RuntimeError, match="prepare failed"):
        run()
    live = [c for c in state.children if not c.done]
    assert {c.role for c in live} == {"prepare", "consume"}
    assert all(c.stopped for c in live)
    assert not list(directory.glob(".pipeline-*"))
    assert json.loads((directory / "pipeline.json").read_text())["status"] == "failed"


def test_consumer_failure_removes_already_prepared_payload(harness):
    state, run, directory = harness
    state.fail = ("consume", 0)
    with pytest.raises(RuntimeError, match="consume failed"):
        run()
    assert not list(directory.glob(".pipeline-*"))
    assert all(c.done or c.stopped for c in state.children)


def test_persistent_process_death_stops_pipeline(harness):
    state, run, directory = harness
    state.fail = ("consume", 0)
    with pytest.raises(RuntimeError, match="exited unexpectedly"):
        run(persistent=True)
    assert all(c.stopped for c in state.children)
    assert not list(directory.glob(".pipeline-*"))


def test_preparation_can_wait_for_analysis_phase(harness):
    state, run, _ = harness
    state.require_analysis = True
    summary = run(persistent=True, defer_prepare=True)
    assert summary["status"] == "complete"
    assert summary["defer_prepare"]


def test_observed_pressure_cancels_both_stages(harness):
    state, run, directory = harness
    state.pressure = True
    with pytest.raises(MemoryError, match="observed memory"):
        run()
    assert all(c.done or c.stopped for c in state.children)
    assert not list(directory.glob(".pipeline-*"))


def test_interrupt_is_failed_and_releases_children(harness):
    state, run, directory = harness
    state.fail = "interrupt"
    with pytest.raises(KeyboardInterrupt):
        run()
    assert all(c.stopped for c in state.children)
    assert not list(directory.glob(".pipeline-*"))
    summary = json.loads((directory / "pipeline.json").read_text())
    assert summary["status"] == "failed"
    assert "KeyboardInterrupt" in summary["error"]


def test_insufficient_ram_launches_nothing(harness):
    state, run, _ = harness
    with pytest.raises(MemoryError, match="reservations require"):
        run(memory_limit=pipeline.GiB)
    assert not state.children


def test_available_ram_bounds_two_retained_artifacts(harness):
    _, run, _ = harness
    summary = run(memory_limit=24 * pipeline.GiB)
    reserved = summary["reservations"]
    assert 0 < reserved["ready_payload"] < reserved["requested_payload_limit"]
    assert reserved["retained_payloads"] == 2
    assert (
        reserved["producer"]
        + reserved["consumer"]
        + reserved["headroom"]
        + 2 * reserved["ready_payload"]
    ) < reserved["limit"]


def test_variable_frame_decode_size_admission(harness):
    state, run, _ = harness
    state.decode_bytes = 4 * pipeline.GiB
    with pytest.raises(MemoryError, match="Full-frame decode"):
        run()
    assert not state.children
