"""Batching, flush and commit-ordering rules of the parent-only catalog batcher."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pyarrow.parquet as pq
import pytest

from pycwb.modules.background_cuda.output_buffer import BufferedCatalog, OutputWriter
from pycwb.modules.catalog.catalog import Catalog
from pycwb.types.trigger import Trigger
from pycwb.workflow.subflow.process_job_segment_native import LagOutputContext


@pytest.fixture
def catalog_path(tmp_path: Path) -> Path:
    """An empty native-schema catalog fragment that ``Catalog.open`` accepts."""
    path = tmp_path / "catalog_1.parquet"
    pq.write_table(Trigger.arrow_schema().empty_table(), path)
    return path


def _trigger(lag: int, index: int) -> Trigger:
    return Trigger(id=f"t{lag}_{index}", job_id=1, lag_idx=lag, trial_idx=0, cluster_id=index, rho=float(lag + index))


def _progress(lag: int, n_triggers: int, status: str = "completed") -> dict:
    return {
        "type": "progress",
        "job_id": 1,
        "trial_idx": 0,
        "lag_idx": lag,
        "n_triggers": n_triggers,
        "livetime": 10.0,
        "status": status,
    }


def _enqueue_lag(sink: BufferedCatalog, lag: int, n_triggers: int) -> None:
    for index in range(n_triggers):
        sink.put({"type": "trigger", "trigger": _trigger(lag, index)})
    sink.put(_progress(lag, n_triggers))


@pytest.mark.parametrize("batch", [0, -1, 257])
def test_batch_size_bounds(catalog_path: Path, batch: int) -> None:
    with pytest.raises(ValueError, match="between 1 and 256"):
        BufferedCatalog(catalog_path, batch)


def test_open_requires_existing_catalog(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        BufferedCatalog(tmp_path / "missing.parquet", 2)


def test_put_rejects_unknown_messages(catalog_path: Path) -> None:
    sink = BufferedCatalog(catalog_path, 2)
    with pytest.raises(ValueError, match="Unexpected output message"):
        sink.put({"type": "waveform"})


def test_progress_rows_drop_type_and_gain_timestamp(catalog_path: Path) -> None:
    sink = BufferedCatalog(catalog_path, 2)
    sink.put(_progress(4, 0))
    assert len(sink.progress) == 1
    row = sink.progress[0]
    assert "type" not in row
    assert row["lag_idx"] == 4
    assert isinstance(row["timestamp"], float)


def test_commit_waits_for_full_batch_then_flushes(catalog_path: Path) -> None:
    sink = BufferedCatalog(catalog_path, 2)
    _enqueue_lag(sink, 0, 2)
    sink.commit_if_ready()
    assert sink.flush_count == 0
    assert pq.read_table(catalog_path).num_rows == 0
    assert not Path(sink.catalog.progress_file).exists()
    _enqueue_lag(sink, 1, 1)
    sink.commit_if_ready()
    assert sink.flush_count == 1
    assert sink.triggers == [] and sink.progress == []
    assert pq.read_table(catalog_path).num_rows == 3
    assert sink.catalog.get_completed_lags(1) == {0: {0, 1}}


def test_final_flush_commits_partial_tail(catalog_path: Path) -> None:
    sink = BufferedCatalog(catalog_path, 4)
    for lag in range(5):
        _enqueue_lag(sink, lag, 1)
        sink.commit_if_ready()
    assert sink.flush_count == 1
    sink.flush()
    assert sink.flush_count == 2
    assert sink.flush_seconds > 0.0
    assert pq.read_table(catalog_path).num_rows == 5
    assert sink.catalog.get_completed_lags(1) == {0: set(range(5))}
    table = pq.read_table(catalog_path)
    assert sorted(table.column("lag_idx").to_pylist()) == list(range(5))


def test_flush_without_progress_is_a_no_op(catalog_path: Path) -> None:
    sink = BufferedCatalog(catalog_path, 2)
    sink.flush()
    assert sink.flush_count == 0
    assert not Path(sink.catalog.progress_file).exists()


def test_triggers_without_completed_lag_cannot_commit(catalog_path: Path) -> None:
    sink = BufferedCatalog(catalog_path, 2)
    sink.put({"type": "trigger", "trigger": _trigger(0, 0)})
    with pytest.raises(RuntimeError, match="without completed lag records"):
        sink.flush()
    assert pq.read_table(catalog_path).num_rows == 0


def test_large_trigger_count_forces_flush_before_batch(catalog_path: Path) -> None:
    sink = BufferedCatalog(catalog_path, 256)
    for index in range(4096):
        sink.put({"type": "trigger", "trigger": _trigger(0, index)})
    sink.put(_progress(0, 4096))
    sink.commit_if_ready()
    assert sink.flush_count == 1
    assert pq.read_table(catalog_path).num_rows == 4096


def test_trigger_write_failure_cannot_commit_progress(catalog_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sink = BufferedCatalog(catalog_path, 1)
    _enqueue_lag(sink, 0, 2)

    def fail(*_: object) -> None:
        raise OSError("injected")

    monkeypatch.setattr(sink.catalog, "add_triggers", fail)
    with pytest.raises(OSError, match="injected"):
        sink.flush()
    assert not Path(sink.catalog.progress_file).exists()
    assert pq.read_table(catalog_path).num_rows == 0
    # Nothing was cleared, so a retry can still commit the batch.
    assert len(sink.triggers) == 2 and len(sink.progress) == 1


def test_progress_failure_is_recoverable_by_native_resume(catalog_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sink = BufferedCatalog(catalog_path, 1)
    _enqueue_lag(sink, 3, 2)

    def fail() -> None:
        raise OSError("injected")

    monkeypatch.setattr(sink, "_append_progress", fail)
    with pytest.raises(OSError, match="injected"):
        sink.flush()
    catalog = Catalog.open(str(catalog_path))
    assert catalog.get_completed_lags(1) == {}
    assert pq.read_table(catalog_path).num_rows == 2
    assert catalog.remove_stale_triggers(1, catalog.get_completed_lags(1)) == 2
    assert pq.read_table(catalog_path).num_rows == 0
    resumed = BufferedCatalog(catalog_path, 1)
    _enqueue_lag(resumed, 3, 2)
    resumed.flush()
    assert resumed.catalog.get_completed_lags(1) == {0: {3}}
    assert pq.read_table(catalog_path).num_rows == 2


def test_empty_skipped_lag_is_committed(catalog_path: Path) -> None:
    sink = BufferedCatalog(catalog_path, 2)
    sink.put(_progress(999, 0, status="skipped_segTHR"))
    sink.flush()
    assert sink.catalog.get_completed_lags(1) == {0: {999}}
    assert pq.read_table(catalog_path).num_rows == 0
    progress = pq.read_table(sink.catalog.progress_file)
    assert progress.column("status").to_pylist() == ["skipped_segTHR"]


def _context(tmp_path: Path, catalog_path: Path, **config: object) -> LagOutputContext:
    return LagOutputContext(
        str(tmp_path),
        SimpleNamespace(save_waveform=False, catalog_dir="", **config),
        SimpleNamespace(injections=None),
        str(catalog_path),
        None,
        object(),
        None,
        None,
    )


def test_output_writer_defaults_to_direct_native_saves(
    tmp_path: Path, catalog_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PYCWB_GPU_OUTPUT_BATCH", raising=False)
    monkeypatch.delenv("PYCWB_GPU_Q_RECONSTRUCTION", raising=False)
    context = _context(tmp_path, catalog_path)
    writer = OutputWriter(context)
    assert writer.sink is None
    assert writer.context is context
    writer.close()


def test_output_writer_replaces_queue_only_in_private_context(
    tmp_path: Path, catalog_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PYCWB_GPU_OUTPUT_BATCH", "2")
    monkeypatch.delenv("PYCWB_GPU_Q_RECONSTRUCTION", raising=False)
    context = _context(tmp_path, catalog_path)
    queue = context.queue
    writer = OutputWriter(context)
    assert context.queue is queue
    assert isinstance(writer.sink, BufferedCatalog)
    assert writer.context.queue is writer.sink
    assert writer.context is not context
    for lag in range(3):
        _enqueue_lag(writer.sink, lag, 1)
        writer.sink.commit_if_ready()
    assert writer.sink.flush_count == 1
    writer.close()
    assert writer.sink.flush_count == 2
    assert pq.read_table(catalog_path).num_rows == 3
    assert writer.sink.catalog.get_completed_lags(1) == {0: {0, 1, 2}}


@pytest.mark.parametrize("value", ["0", "-2"])
def test_output_writer_rejects_non_positive_batch(
    tmp_path: Path, catalog_path: Path, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("PYCWB_GPU_OUTPUT_BATCH", value)
    with pytest.raises(ValueError, match="positive"):
        OutputWriter(_context(tmp_path, catalog_path))


def test_output_writer_batching_rejects_saved_waveforms_and_injections(
    tmp_path: Path, catalog_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PYCWB_GPU_OUTPUT_BATCH", "2")
    context = _context(tmp_path, catalog_path)
    context = LagOutputContext(
        context.working_dir,
        SimpleNamespace(save_waveform=True, catalog_dir=""),
        context.sub_job_seg,
        context.catalog_file,
        None,
        None,
        None,
        None,
    )
    with pytest.raises(ValueError, match="without saved waveforms"):
        OutputWriter(context)
    context = LagOutputContext(
        context.working_dir,
        SimpleNamespace(save_waveform=False, catalog_dir=""),
        SimpleNamespace(injections=[object()]),
        context.catalog_file,
        None,
        None,
        None,
        None,
    )
    with pytest.raises(ValueError, match="without saved waveforms"):
        OutputWriter(context)


def test_output_writer_batching_requires_catalog_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYCWB_GPU_OUTPUT_BATCH", "2")
    context = LagOutputContext(
        str(tmp_path),
        SimpleNamespace(save_waveform=False, catalog_dir=""),
        SimpleNamespace(injections=None),
        None,
        None,
        None,
        None,
        None,
    )
    with pytest.raises(ValueError, match="requires a catalog path"):
        OutputWriter(context)


def test_output_writer_save_routes_through_native_saver(
    tmp_path: Path, catalog_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PYCWB_GPU_OUTPUT_BATCH", "1")
    monkeypatch.delenv("PYCWB_GPU_PROFILE_LAGS", raising=False)
    context = _context(tmp_path, catalog_path)
    writer = OutputWriter(context)
    saved: list[tuple] = []
    writer.save_native = lambda ctx, res: saved.append((ctx, res))
    result = SimpleNamespace(lag=5)
    writer.save(object(), result)
    assert saved == [(context, result)]
    assert os.environ.get("PYCWB_GPU_OUTPUT_BATCH") == "1"
