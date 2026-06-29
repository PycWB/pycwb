"""Tests for post-production selection actions."""

from __future__ import annotations

import json
import os

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pycwb.modules.postprocess import selection as selection_module
from pycwb.modules.postprocess.selection import (
    filter_real_simulation,
    trigger_selection,
)


def _write_catalog_with_jobs(path, jobs, rows=None):
    rows = rows or {"job_id": [], "rho": [], "lag_idx": []}
    arrays = {
        name: pa.array(values)
        for name, values in rows.items()
    }
    table = pa.table(arrays)
    metadata = {
        b"config": b"{}",
        b"jobs": json.dumps(jobs).encode(),
    }
    pq.write_table(table.replace_schema_metadata(metadata), path)


def test_trigger_selection_split_is_disjoint(tmp_path):
    work_dir = str(tmp_path)
    catalog_dir = tmp_path / "BKG" / "catalog"
    catalog_dir.mkdir(parents=True)

    progress = pd.DataFrame({
        "job_id": [1, 1, 2, 2, 3, 3, 4, 4],
        "lag_idx": [0, 1, 0, 1, 0, 1, 0, 1],
        "livetime": [10.0, 100.0, 10.0, 100.0, 10.0, 100.0, 10.0, 100.0],
    })
    catalog = pd.DataFrame({
        "job_id": [1, 1, 2, 2, 3, 3, 4, 4],
        "lag_idx": [0, 1, 0, 1, 0, 1, 0, 1],
        "rho": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    })
    progress.to_parquet(catalog_dir / "progress.M1.parquet", index=False)
    _write_catalog_with_jobs(
        catalog_dir / "catalog.M1.parquet",
        [{"index": jid, "shift": [0.0, 0.0]} for jid in [1, 2, 3, 4]],
        rows=catalog.to_dict(orient="list"),
    )

    result = trigger_selection(
        work_dir=work_dir,
        catalog_file="BKG/catalog/catalog.M1.parquet",
        progress_file="BKG/catalog/progress.M1.parquet",
        split={"fractions": {"train": 0.5, "far": 0.5}, "seed": 1},
        returns=["jobs", "triggers", "livetime"],
        outputs={
            "train": {
                "jobs_file": "tmp/train_jobs.txt",
                "triggers_file": "tmp/train.parquet",
            },
            "far": {
                "jobs_file": "tmp/far_jobs.txt",
                "triggers_file": "tmp/far.parquet",
            },
        },
        exclude_zero_lag=True,
    )

    train_jobs = set(result["train"]["job_ids"])
    far_jobs = set(result["far"]["job_ids"])
    assert train_jobs
    assert far_jobs
    assert train_jobs.isdisjoint(far_jobs)
    assert result["train"]["livetime"]["seconds"] > 0
    assert result["far"]["livetime"]["seconds"] > 0
    assert os.path.exists(tmp_path / result["train"]["triggers_file"])
    assert os.path.exists(tmp_path / result["far"]["triggers_file"])


def test_trigger_selection_interval_split_is_disjoint_by_shift_and_lag(tmp_path):
    work_dir = str(tmp_path)
    catalog_dir = tmp_path / "BKG" / "catalog"
    catalog_dir.mkdir(parents=True)

    progress = pd.DataFrame({
        "job_id": [1, 1, 2, 2],
        "lag_idx": [1, 2, 1, 2],
        "livetime": [100.0, 100.0, 100.0, 100.0],
        "status": ["completed", "completed", "completed", "completed"],
    })
    catalog = pd.DataFrame({
        "job_id": [1, 1, 2, 2],
        "lag_idx": [1, 2, 1, 2],
        "rho": [10.0, 20.0, 30.0, 40.0],
    })
    progress.to_parquet(catalog_dir / "progress.M1.parquet", index=False)
    _write_catalog_with_jobs(
        catalog_dir / "catalog.M1.parquet",
        [
            {"index": 1, "shift": [0.0, 0.0]},
            {"index": 2, "shift": [0.0, 0.0]},
        ],
        rows=catalog.to_dict(orient="list"),
    )

    result = trigger_selection(
        work_dir=work_dir,
        catalog_file="BKG/catalog/catalog.M1.parquet",
        progress_file="BKG/catalog/progress.M1.parquet",
        split={"by": "interval_livetime", "fractions": {"train": 0.5, "far": 0.5}, "seed": 1},
        returns=["jobs", "triggers", "livetime"],
        outputs={
            "train": {
                "jobs_file": "tmp/train_jobs.txt",
                "progress_file": "tmp/train_progress.parquet",
                "intervals_file": "tmp/train_intervals.parquet",
                "intervals_csv_file": "tmp/train_intervals.csv",
                "triggers_file": "tmp/train.parquet",
            },
            "far": {
                "jobs_file": "tmp/far_jobs.txt",
                "progress_file": "tmp/far_progress.parquet",
                "intervals_file": "tmp/far_intervals.parquet",
                "intervals_csv_file": "tmp/far_intervals.csv",
                "triggers_file": "tmp/far.parquet",
            },
        },
        exclude_zero_lag=True,
    )

    train_intervals = pd.read_parquet(tmp_path / result["train"]["intervals_file"])
    far_intervals = pd.read_parquet(tmp_path / result["far"]["intervals_file"])
    train_intervals_csv = pd.read_csv(tmp_path / result["train"]["intervals_csv_file"])
    train_keys = set(zip(train_intervals["shift_key"], train_intervals["lag_idx"]))
    far_keys = set(zip(far_intervals["shift_key"], far_intervals["lag_idx"]))

    assert result["split_by"] == "interval_livetime"
    assert train_keys
    assert far_keys
    assert train_keys.isdisjoint(far_keys)
    assert train_intervals_csv[["shift_key", "lag_idx", "livetime", "n_rows", "n_jobs"]].equals(
        train_intervals[["shift_key", "lag_idx", "livetime", "n_rows", "n_jobs"]]
    )
    assert set(result["train"]["job_ids"]) == {1, 2}
    assert set(result["far"]["job_ids"]) == {1, 2}

    train_progress = pd.read_parquet(tmp_path / result["train"]["progress_file"])
    far_progress = pd.read_parquet(tmp_path / result["far"]["progress_file"])
    train_triggers = pd.read_parquet(tmp_path / result["train"]["triggers_file"])
    far_triggers = pd.read_parquet(tmp_path / result["far"]["triggers_file"])

    assert set(zip(train_triggers["job_id"], train_triggers["lag_idx"])) == set(zip(train_progress["job_id"], train_progress["lag_idx"]))
    assert set(zip(far_triggers["job_id"], far_triggers["lag_idx"])) == set(zip(far_progress["job_id"], far_progress["lag_idx"]))
    assert result["train"]["livetime"]["seconds"] + result["far"]["livetime"]["seconds"] == 400.0
    assert result["train"]["livetime"]["n_jobs"] == 2
    assert result["far"]["livetime"]["n_jobs"] == 2
    assert result["train"]["livetime"]["n_intervals"] + result["far"]["livetime"]["n_intervals"] == 2


def test_trigger_selection_streams_trigger_file_outputs(tmp_path, monkeypatch):
    progress = pd.DataFrame({
        "job_id": [1, 1, 2, 2],
        "lag_idx": [1, 2, 1, 2],
        "livetime": [100.0, 100.0, 100.0, 100.0],
        "status": ["completed", "completed", "completed", "completed"],
    })
    catalog = pd.DataFrame({
        "job_id": [1, 1, 2, 2],
        "lag_idx": [1, 2, 1, 2],
        "rho": [10.0, 20.0, 30.0, 40.0],
    })
    progress.to_parquet(tmp_path / "progress.parquet", index=False)
    _write_catalog_with_jobs(
        tmp_path / "catalog.parquet",
        [{"index": 1, "shift": [0.0, 0.0]}, {"index": 2, "shift": [0.0, 0.0]}],
        rows=catalog.to_dict(orient="list"),
    )

    def _fail_full_read(*args, **kwargs):
        raise AssertionError("full catalog read should not be used for trigger file outputs")

    monkeypatch.setattr(selection_module, "_read_catalog_filtered", _fail_full_read)

    result = trigger_selection(
        work_dir=str(tmp_path),
        catalog_file="catalog.parquet",
        progress_file="progress.parquet",
        split={"by": "interval_livetime", "fractions": {"train": 0.5, "far": 0.5}, "seed": 1},
        returns=["jobs", "triggers", "livetime"],
        outputs={
            "train": {"triggers_file": "train.parquet"},
            "far": {"triggers_file": "far.parquet"},
        },
        exclude_zero_lag=True,
    )

    train = pd.read_parquet(tmp_path / result["train"]["triggers_file"])
    far = pd.read_parquet(tmp_path / result["far"]["triggers_file"])
    assert len(train) + len(far) == len(catalog)
    assert result["train"]["livetime"]["n_triggers"] + result["far"]["livetime"]["n_triggers"] == len(catalog)


def test_trigger_selection_interval_split_excludes_physical_zero_lag(tmp_path):
    progress = pd.DataFrame({
        "job_id": [1, 2, 3, 4],
        "lag_idx": [0, 0, 1, 1],
        "livetime": [10.0, 20.0, 30.0, 40.0],
        "status": ["completed", "completed", "completed", "completed"],
    })
    catalog = pd.DataFrame({
        "job_id": [1, 2, 3, 4],
        "lag_idx": [0, 0, 1, 1],
        "rho": [1.0, 2.0, 3.0, 4.0],
    })
    progress.to_parquet(tmp_path / "progress.parquet", index=False)
    _write_catalog_with_jobs(
        tmp_path / "catalog.parquet",
        [
            {"index": 1, "shift": [0.0, 0.0]},
            {"index": 2, "shift": [0.0, 1200.0]},
            {"index": 3, "shift": [0.0, 0.0]},
            {"index": 4, "shift": [0.0, 1200.0]},
        ],
        rows=catalog.to_dict(orient="list"),
    )

    result = trigger_selection(
        work_dir=str(tmp_path),
        catalog_file="catalog.parquet",
        progress_file="progress.parquet",
        split={"by": "interval_livetime", "fractions": {"train": 0.5, "far": 0.5}, "seed": 4},
        returns=["triggers", "livetime"],
        outputs={
            "train": {"triggers_file": "train.parquet", "intervals_file": "train_intervals.parquet"},
            "far": {"triggers_file": "far.parquet", "intervals_file": "far_intervals.parquet"},
        },
        exclude_zero_lag=True,
    )

    selected = pd.concat([
        pd.read_parquet(tmp_path / result["train"]["triggers_file"]),
        pd.read_parquet(tmp_path / result["far"]["triggers_file"]),
    ])
    assert set(zip(selected["job_id"], selected["lag_idx"])) == {(2, 0), (3, 1), (4, 1)}
    assert result["train"]["livetime"]["seconds"] + result["far"]["livetime"]["seconds"] == 90.0


def test_trigger_selection_counts_only_completed_progress_livetime(tmp_path):
    progress = pd.DataFrame({
        "job_id": [1, 1, 2, 2],
        "lag_idx": [1, 2, 1, 2],
        "livetime": [100.0, 1000.0, 100.0, 2000.0],
        "status": ["completed", "skipped_segTHR", "completed", "skipped_segTHR"],
    })
    progress.to_parquet(tmp_path / "progress.parquet", index=False)
    _write_catalog_with_jobs(
        tmp_path / "catalog.parquet",
        [{"index": 1, "shift": [0.0, 0.0]}, {"index": 2, "shift": [0.0, 0.0]}],
    )

    result = trigger_selection(
        work_dir=str(tmp_path),
        catalog_file="catalog.parquet",
        progress_file="progress.parquet",
        returns=["livetime"],
        fraction=1.0,
    )

    assert result["livetime"]["seconds"] == 200.0
    assert result["total"]["seconds"] == 200.0


def test_trigger_selection_excludes_only_unshifted_zero_lag(tmp_path):
    progress = pd.DataFrame({
        "job_id": [1, 2, 3],
        "lag_idx": [0, 0, 1],
        "livetime": [10.0, 20.0, 30.0],
        "status": ["completed", "completed", "completed"],
    })
    progress.to_parquet(tmp_path / "progress.parquet", index=False)
    _write_catalog_with_jobs(
        tmp_path / "catalog.parquet",
        [
            {"index": 1, "shift": [0.0, 0.0]},
            {"index": 2, "shift": [0.0, 1200.0]},
            {"index": 3, "shift": [0.0, 0.0]},
        ],
    )

    result = trigger_selection(
        work_dir=str(tmp_path),
        catalog_file="catalog.parquet",
        progress_file="progress.parquet",
        returns=["livetime"],
        fraction=1.0,
        exclude_zero_lag=True,
    )

    assert result["livetime"]["seconds"] == 50.0
    assert result["total"]["seconds"] == 50.0


def test_trigger_selection_filters_zero_lag_from_materialized_triggers(tmp_path):
    progress = pd.DataFrame({
        "job_id": [1, 2, 3],
        "lag_idx": [0, 0, 1],
        "livetime": [10.0, 20.0, 30.0],
        "status": ["completed", "completed", "completed"],
    })
    catalog = pd.DataFrame({
        "job_id": [1, 2, 3],
        "lag_idx": [0, 0, 1],
        "rho": [8.0, 9.0, 10.0],
    })
    progress.to_parquet(tmp_path / "progress.parquet", index=False)
    _write_catalog_with_jobs(
        tmp_path / "catalog.parquet",
        [
            {"index": 1, "shift": [0.0, 0.0]},
            {"index": 2, "shift": [0.0, 1200.0]},
            {"index": 3, "shift": [0.0, 0.0]},
        ],
        rows=catalog.to_dict(orient="list"),
    )

    result = trigger_selection(
        work_dir=str(tmp_path),
        catalog_file="catalog.parquet",
        progress_file="progress.parquet",
        returns=["jobs", "triggers", "livetime"],
        fraction=1.0,
        exclude_zero_lag=True,
        outputs={"triggers_file": "selected.parquet"},
    )

    selected = pd.read_parquet(tmp_path / result["triggers_file"])
    assert selected["job_id"].tolist() == [2, 3]
    assert selected["lag_idx"].tolist() == [0, 1]


def test_filter_real_simulation_keeps_recovered_clean_rows(tmp_path):
    work_dir = str(tmp_path)
    matched = pd.DataFrame({
        "id": [1.0, 2.0, None, 4.0],
        "rho": [8.0, 9.0, None, 10.0],
        "sim_sim_idx": [11.0, 12.0, 13.0, 14.0],
        "sim_vetoed_cat0": [False, True, False, False],
        "sim_vetoed_cat1": [False, False, False, False],
        "sim_vetoed_cat2": [False, False, False, False],
        "sim_across_segments": [False, False, False, True],
    })
    matched.to_parquet(tmp_path / "matched_outer.parquet", index=False)

    result = filter_real_simulation(
        work_dir=work_dir,
        matched_file="matched_outer.parquet",
        output_file="clean.parquet",
    )
    clean = pd.read_parquet(tmp_path / "clean.parquet")

    assert result["n_before"] == 4
    assert result["n_after"] == 1
    assert clean["id"].tolist() == [1.0]
