"""Integration checks for manifest-backed postproduction outputs."""

from types import SimpleNamespace

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pycwb.config import Config
from pycwb.modules.catalog.catalog import Catalog
from pycwb.modules.catalog.provenance import (
    MANIFEST_KEY,
    catalog_provenance,
)
from pycwb.modules.postprocess import evaluate
from pycwb.modules.postprocess.lag_filters import try_unshifted_job_ids_from_catalog
from pycwb.modules.postprocess.report_summaries import _read_catalog_metadata
from pycwb.modules.postprocess.selection import (
    filter_real_simulation,
    trigger_selection,
)


def _run(root, name="run", *, inline=False, first_job=1):
    directory = root / name / "catalog"
    directory.mkdir(parents=True)
    path = directory / "catalog.parquet"
    jobs = [
        {"index": first_job, "shift": [0.0, 0.0]},
        {"index": first_job + 1, "shift": [0.0, 100.0]},
    ]
    Catalog.create(str(path), Config(), jobs, jobs_in_metadata=inline)
    # Both jobs contribute exposure; the first has no recovered triggers.
    rows = pa.table(
        {
            "id": ["event"],
            "job_id": [first_job + 1],
            "lag_idx": [1],
            "rho": [8.0],
            "trial_idx": [0],
            "gps_time": [1000.0],
        }
    )
    pq.write_table(rows.replace_schema_metadata(pq.read_schema(path).metadata), path)
    progress = directory / "progress.parquet"
    pd.DataFrame(
        {
            "job_id": [first_job, first_job + 1],
            "lag_idx": [1, 1],
            "livetime": [100.0, 200.0],
            "status": ["completed", "completed"],
        }
    ).to_parquet(progress, index=False)
    return path, progress, jobs


@pytest.mark.parametrize("inline", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_selection_then_scoring_preserves_provenance_and_exposure(
    tmp_path, monkeypatch, inline, empty
):
    path, progress, jobs = _run(tmp_path, inline=inline)
    result = trigger_selection(
        str(tmp_path),
        str(path),
        str(progress),
        selection={"fraction": 1.0},
        trigger_filter={"query": "rho > 100"} if empty else None,
        outputs={"triggers_file": "tmp/selected.parquet"},
    )
    assert result["livetime"]["seconds"] == 300.0
    selected = tmp_path / result["triggers_file"]
    assert Catalog.open(str(selected)).jobs == jobs
    assert try_unshifted_job_ids_from_catalog(str(selected)) == {1}
    monkeypatch.setattr(
        evaluate.xgb,
        "XGBClassifier",
        lambda: SimpleNamespace(load_model=lambda path: None),
    )
    monkeypatch.setattr(
        evaluate,
        "_score_catalog_dataframe",
        lambda frame, *args: frame.assign(xgb_prob=0.9),
    )
    evaluate.score_catalog(
        str(tmp_path),
        str(selected),
        "unused-model",
        output_file="scored/result.parquet",
    )
    scored = tmp_path / "scored/result.parquet"
    assert Catalog.open(str(scored)).jobs == jobs
    assert pq.read_table(scored).num_rows == (0 if empty else 1)
    assert "job_id" in pq.read_schema(scored).names


def test_simulation_filter_then_selection_keeps_provenance(tmp_path):
    path, progress, jobs = _run(tmp_path)
    pytest.importorskip("duckdb")
    from pycwb.modules.postprocess.matching import match_simulations

    simulations = tmp_path / "simulations.parquet"
    pq.write_table(
        pa.table(
            {
                "sim_idx": [5, 6],
                "job_id": [2, 1],
                "trial_idx": [0, 0],
                "gps_time": [1000.0, 2000.0],
                "real_start": [999.5, 1999.5],
                "real_end": [1000.5, 2000.5],
            }
        ),
        simulations,
    )
    match_simulations(
        str(tmp_path), str(path), str(simulations), "matched.parquet", how="right"
    )
    matched = tmp_path / "matched.parquet"
    assert Catalog.open(str(matched)).jobs == jobs
    filtered = filter_real_simulation(str(tmp_path), str(matched), "tmp/real.parquet")
    result = trigger_selection(
        str(tmp_path),
        filtered["triggers_file"],
        str(progress),
        exclude_zero_lag=False,
        selection={"fraction": 1.0},
        outputs={"triggers_file": "tmp/selected.parquet"},
    )
    assert result["livetime"]["seconds"] == 300.0
    assert Catalog.open(str(tmp_path / result["triggers_file"])).jobs == jobs


def test_two_runs_can_share_output_directory(tmp_path):
    for name, first in [("first", 1), ("second", 10)]:
        path, progress, jobs = _run(tmp_path, name, first_job=first)
        output = f"tmp/{name}.parquet"
        trigger_selection(
            str(tmp_path),
            str(path),
            str(progress),
            selection={"fraction": 1.0},
            outputs={"triggers_file": output},
        )
        assert Catalog.open(str(tmp_path / output)).jobs == jobs
    assert not (tmp_path / "tmp/jobs.parquet").exists()


def test_rebasing_does_not_load_manifest_rows(tmp_path, monkeypatch):
    path, _, _ = _run(tmp_path)
    monkeypatch.setattr(
        pq, "read_table", lambda *a, **k: pytest.fail("manifest rows were loaded")
    )
    metadata = catalog_provenance(str(path), str(tmp_path / "tmp/selected.parquet"))
    assert MANIFEST_KEY in metadata


def test_report_does_not_hide_broken_manifest(tmp_path):
    path, _, _ = _run(tmp_path)
    (path.parent / "jobs.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="job manifest is missing"):
        _read_catalog_metadata(str(path))


def test_interval_partitions_preserve_selected_exposure(tmp_path):
    path, progress, jobs = _run(tmp_path)
    result = trigger_selection(
        str(tmp_path),
        str(path),
        str(progress),
        split={
            "by": "interval_livetime",
            "fractions": {"train": 0.5, "far": 0.5},
            "seed": 1,
        },
        outputs={
            name: {"triggers_file": f"tmp/{name}.parquet"} for name in ["train", "far"]
        },
    )
    assert (
        sum(result[name]["livetime"]["seconds"] for name in ["train", "far"]) == 300.0
    )
    for name in ["train", "far"]:
        assert Catalog.open(str(tmp_path / result[name]["triggers_file"])).jobs == jobs
        selected_ids = result[name]["job_ids"]
        assert result[name]["livetime"]["seconds"] == sum(
            {1: 100.0, 2: 200.0}[index] for index in selected_ids
        )


def test_plain_simulation_filter_output_remains_usable(tmp_path):
    path, progress, _ = _run(tmp_path)
    matched = tmp_path / "plain.parquet"
    pq.write_table(
        pa.table({"id": ["event"], "job_id": [2], "lag_idx": [1], "sim_sim_idx": [5]}),
        matched,
    )
    filtered = filter_real_simulation(str(tmp_path), str(matched), "tmp/real.parquet")
    result = trigger_selection(
        str(tmp_path),
        filtered["triggers_file"],
        str(progress),
        exclude_zero_lag=False,
        selection={"fraction": 1.0},
    )
    assert result["livetime"]["seconds"] == 300.0


def test_moving_whole_tree_keeps_relative_references(tmp_path):
    import shutil

    original = tmp_path / "original"
    original.mkdir()
    path, progress, jobs = _run(original)
    trigger_selection(
        str(original),
        str(path),
        str(progress),
        selection={"fraction": 1.0},
        outputs={"triggers_file": "tmp/selected.parquet"},
    )
    relocated = tmp_path / "relocated"
    shutil.move(str(original), str(relocated))
    assert Catalog.open(str(relocated / "tmp/selected.parquet")).jobs == jobs
