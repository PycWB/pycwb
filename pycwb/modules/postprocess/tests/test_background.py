"""Selection owns zero lag; background rates use their inputs as given."""

import logging

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pycwb.config import Config
from pycwb.modules.catalog.catalog import Catalog
from pycwb.modules.postprocess.background import process_background
from pycwb.modules.postprocess.selection import trigger_selection


def write_run(tmp_path, jobs, inline=False):
    """Catalog with lag 0 and 1 triggers per job, plus matching progress."""
    path = tmp_path / "catalog.parquet"
    Catalog.create(str(path), Config(), jobs, jobs_in_metadata=inline)
    ids = [job["index"] for job in jobs]
    rows = [(job_id, lag) for job_id in ids for lag in (0, 1)]
    table = pa.table({
        "id": [f"{job_id}-{lag}" for job_id, lag in rows],
        "job_id": [job_id for job_id, _ in rows],
        "lag_idx": [lag for _, lag in rows],
        "time_lag_L1": [float(lag) for _, lag in rows],
        "segment_lag_L1": [float(next(j for j in jobs if j["index"] == job_id)["shift"][1])
                           for job_id, _ in rows],
        "rho": [6. + i for i in range(len(rows))],
    })
    pq.write_table(table.replace_schema_metadata(pq.read_schema(path).metadata), path)
    progress = tmp_path / "progress.parquet"
    pd.DataFrame({
        "job_id": [job_id for job_id, _ in rows],
        "lag_idx": [lag for _, lag in rows],
        "livetime": [10. * (i + 1) for i in range(len(rows))],
    }).to_parquet(progress)
    return path, progress


def test_inputs_are_used_as_given_and_zero_lag_is_reported(tmp_path, caplog):
    catalog, progress = write_run(tmp_path, [{"index": 1, "shift": [0, 0]}])
    with caplog.at_level(logging.WARNING, logger="pycwb.modules.postprocess.background"):
        result = process_background(catalog, progress, thresholds=[5], exclude_zero_lag=True)
    assert result["livetime"] == 30.
    assert result["n_triggers"] == 2
    messages = " ".join(record.getMessage() for record in caplog.records)
    assert "ignores exclude_zero_lag" in messages
    assert "1 background triggers have no time or segment shift" in messages


@pytest.mark.parametrize("inline", [False, True])
def test_selected_background_keeps_superlag_lag_zero(tmp_path, inline):
    jobs = [{"index": 1, "shift": [0, 0]}, {"index": 2, "shift": [0, 100]}]
    catalog, progress = write_run(tmp_path, jobs, inline=inline)
    selected = trigger_selection(
        str(tmp_path), str(catalog), str(progress), exclude_zero_lag=True,
        outputs={"triggers_file": "bkg.parquet", "progress_file": "bkg_progress.parquet"},
    )
    result = process_background("bkg.parquet", "bkg_progress.parquet",
                                thresholds=[5], work_dir=str(tmp_path))
    # Only job 1 lag 0 (10 s) is zero lag; job 2 lag 0 is shifted by the superlag.
    assert result["livetime"] == 90.
    assert sorted(result["triggers"].id) == ["1-1", "2-0", "2-1"]
    assert selected["livetime"]["seconds"] == result["livetime"]


def test_superlag_only_run_keeps_every_lag_as_background(tmp_path):
    jobs = [{"index": 1, "shift": [0, 100]}, {"index": 2, "shift": [0, 200]}]
    catalog, progress = write_run(tmp_path, jobs)
    selected = trigger_selection(
        str(tmp_path), str(catalog), str(progress), exclude_zero_lag=True,
        outputs={"triggers_file": "bkg.parquet", "progress_file": "bkg_progress.parquet"},
    )
    result = process_background("bkg.parquet", "bkg_progress.parquet",
                                thresholds=[5], work_dir=str(tmp_path))
    assert selected["livetime"]["seconds"] == result["livetime"] == 100.
    assert result["n_triggers"] == 4


def test_missing_job_manifest_is_not_ignored(tmp_path):
    catalog, progress = write_run(tmp_path, [{"index": 1, "shift": [0, 0]}])
    (tmp_path / "jobs.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="job manifest is missing"):
        trigger_selection(str(tmp_path), str(catalog), str(progress))
