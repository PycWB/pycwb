"""Tests for post-production simulation matching actions."""

from __future__ import annotations

import os

import pytest
import pyarrow as pa
import pyarrow.parquet as pq

from pycwb.modules.postprocess import matching as matching_module
from pycwb.modules.postprocess.matching import match_simulations
from pycwb.post_production.workflow_config import (
    prepare_step_args,
    resolve_reference,
    store_result,
    workflow_base_context,
    workflow_runtime,
)


class _FakeMatchedTable:
    num_rows = 7


def test_match_simulations_resolves_paths_and_returns_reference_shape(tmp_path, monkeypatch):
    captured = {}

    def fake_match_simulations_parquet(
        catalog_parquet,
        sim_parquet,
        *,
        window_buffer,
        how,
        output_parquet,
    ):
        captured.update({
            "catalog_parquet": catalog_parquet,
            "sim_parquet": sim_parquet,
            "window_buffer": window_buffer,
            "how": how,
            "output_parquet": output_parquet,
        })
        return _FakeMatchedTable()

    monkeypatch.setattr(
        matching_module,
        "match_simulations_parquet",
        fake_match_simulations_parquet,
    )

    workflow = {
        "vars": {
            "work_dir": str(tmp_path),
            "paths": {
                "catalog": "SIM/catalog/catalog.parquet",
                "simulations": "SIM/catalog/simulations.parquet",
            },
        },
        "runtime": {"tmp_dir": "${work_dir}/tmp/postprod"},
    }
    context = workflow_base_context(workflow)
    runtime = workflow_runtime(workflow, context)
    step = {
        "id": "sim_match",
        "inputs": {
            "catalog_file": "${paths.catalog}",
            "simulation_file": "${paths.simulations}",
        },
        "args": {
            "how": "outer",
            "window_buffer": 0.25,
        },
        "outputs": {
            "output_file": "tmp://matched_outer.parquet",
        },
    }

    args = prepare_step_args(step, context, runtime)
    result = match_simulations(**args)
    store_result(context, step, result)

    assert captured == {
        "catalog_parquet": os.path.join(str(tmp_path), "SIM/catalog/catalog.parquet"),
        "sim_parquet": os.path.join(str(tmp_path), "SIM/catalog/simulations.parquet"),
        "window_buffer": 0.25,
        "how": "outer",
        "output_parquet": os.path.join(str(tmp_path), "tmp/postprod/matched_outer.parquet"),
    }
    assert result == {
        "matched_file": os.path.join(str(tmp_path), "tmp/postprod/matched_outer.parquet"),
        "output_file": os.path.join(str(tmp_path), "tmp/postprod/matched_outer.parquet"),
        "how": "outer",
        "window_buffer": 0.25,
        "n_rows": 7,
    }
    assert resolve_reference("@sim_match.matched_file", context) == result["matched_file"]


def test_match_simulations_duckdb_right_and_outer_outputs(tmp_path):
    pytest.importorskip("duckdb")

    triggers = pa.table({
        "id": ["t1", "t2"],
        "job_id": [1, 3],
        "trial_idx": [0, 0],
        "gps_time": [1000.0, 3000.0],
        "rho": [8.0, 9.0],
    })
    simulations = pa.table({
        "sim_idx": [10, 20],
        "job_id": [1, 2],
        "trial_idx": [0, 0],
        "gps_time": [1000.0, 2000.0],
        "real_start": [999.5, 1999.5],
        "real_end": [1000.5, 2000.5],
    })
    catalog_path = tmp_path / "catalog.parquet"
    simulation_path = tmp_path / "simulations.parquet"
    pq.write_table(triggers, catalog_path)
    pq.write_table(simulations, simulation_path)

    right = match_simulations(
        work_dir=str(tmp_path),
        catalog_file="catalog.parquet",
        simulation_file="simulations.parquet",
        output_file="matched_right.parquet",
        how="right",
    )
    right_df = pq.read_table(tmp_path / "matched_right.parquet").to_pandas()
    assert right["n_rows"] == 2
    assert set(right_df["sim_sim_idx"]) == {10, 20}
    assert right_df["id"].notna().sum() == 1

    outer = match_simulations(
        work_dir=str(tmp_path),
        catalog_file="catalog.parquet",
        simulation_file="simulations.parquet",
        output_file="matched_outer.parquet",
        how="outer",
    )
    outer_df = pq.read_table(tmp_path / "matched_outer.parquet").to_pandas()
    assert outer["n_rows"] == 3
    assert len(outer_df) == 3
    assert outer_df["id"].notna().sum() == 2
    assert outer_df["sim_sim_idx"].notna().sum() == 2
