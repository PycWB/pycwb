"""Tests for the postproduction multi-tab HTML report action."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pycwb.modules.postprocess.report_builder import (
    _livetime_dict,
    _parquet_file_info,
    _zero_lag_livetime_summary,
    postproduction_report,
)


def _write_catalog_with_metadata(path: Path) -> None:
    table = pa.table({"rho": pa.array([1.0, 2.0])})
    metadata = {
        b"pycwb_version": b"prod-1.2.3",
        b"config": json.dumps({
            "ifo": ["H1", "L1"],
            "cfg_search": "blf",
            "fLow": 24.0,
            "fHigh": 1024.0,
            "DQF": [["H1", "cat1"]],
        }).encode(),
        b"jobs": json.dumps([
            {"index": 1, "start": 100.0, "stop": 200.0},
            {"index": 2, "start": 200.0, "stop": 300.0},
        ]).encode(),
    }
    pq.write_table(table.replace_schema_metadata(metadata), path)


def _write_catalog_with_jobs(path: Path, jobs: list[dict]) -> None:
    table = pa.table({"rho": pa.array([1.0, 2.0])})
    metadata = {
        b"pycwb_version": b"prod-1.2.3",
        b"config": json.dumps({"ifo": ["H1", "L1"], "cfg_search": "blf"}).encode(),
        b"jobs": json.dumps(jobs).encode(),
    }
    pq.write_table(table.replace_schema_metadata(metadata), path)


def _write_dummy_png(path: Path) -> None:
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06"
        b"\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\n"
        b"IDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n"
        b"-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )


def test_postproduction_report_builds_multitab_html(tmp_path):
    _write_catalog_with_metadata(tmp_path / "production.parquet")
    (tmp_path / "public").mkdir()
    (tmp_path / "models").mkdir()

    workflow = tmp_path / "workflow.yaml"
    workflow.write_text(
        """
vars:
  bkg_train_fraction: 0.5
  bkg_far_fraction: 0.5
steps:
  - id: bkg_split
    name: Split BKG Train/FAR
    action: postprocess.selection.trigger_selection
    args:
      split:
        by: interval_livetime
        seed: 42
        fractions:
          train: ${bkg_train_fraction}
          far: ${bkg_far_fraction}
  - id: model
    name: Train XGBoost
    action: postprocess.train_xgboost.train_xgboost
    inputs:
      config_file: xgb_config.py
    args:
      model_file: model.ubj
""".strip()
    )
    (tmp_path / "workflow_diagram.html").write_text("<html><body>diagram</body></html>")

    bkg = pd.DataFrame({
        "id": [f"event-{i}" for i in range(12)],
        "job_id": [1, 1, 2, 2] * 3,
        "lag_idx": [1, 2, 1, 2] * 3,
        "trial_idx": [0] * 12,
        "rho": [5.0, 7.0, 6.5, 8.2, 4.5, 9.1, 3.2, 6.8, 8.8, 5.5, 4.9, 7.3],
        "xgb_prob": [0.01, 0.2, 0.1, 0.7, 0.02, 0.9, 0.03, 0.4, 0.8, 0.05, 0.04, 0.5],
        "gps_time": [1000.0 + i * 10 for i in range(12)],
        "central_freq_H1": [40.0 + i for i in range(12)],
        "central_freq_L1": [42.0 + i for i in range(12)],
        "net_cc": [0.8] * 12,
        "likelihood": [20.0 + i for i in range(12)],
        "coherent_energy": [10.0 + i for i in range(12)],
    })
    bkg.to_parquet(tmp_path / "bkg_scored.parquet", index=False)
    bkg.head(5).to_parquet(tmp_path / "train_bkg.parquet", index=False)
    bkg.tail(5).to_parquet(tmp_path / "train_sim.parquet", index=False)
    bkg.tail(4).to_parquet(tmp_path / "sim_scored.parquet", index=False)

    pd.DataFrame({
        "job_id": [1, 2, 3],
        "lag_idx": [1, 1, 2],
        "livetime": [10.0, 20.0, 30.0],
        "status": ["completed", "completed", "completed"],
        "n_triggers": [2, 3, 4],
    }).to_parquet(tmp_path / "progress.parquet", index=False)
    pd.DataFrame({
        "job_id": [1, 2, 3],
        "lag_idx": [0, 0, 1],
        "livetime": [5.0, 7.0, 11.0],
        "status": ["completed", "completed", "completed"],
        "n_triggers": [1, 1, 1],
    }).to_parquet(tmp_path / "zero_progress.parquet", index=False)
    pd.DataFrame({
        "shift_key": ["0,0", "0,1"],
        "lag_idx": [1, 2],
        "livetime": [30.0, 30.0],
        "n_rows": [2, 1],
        "n_jobs": [2, 1],
    }).to_parquet(tmp_path / "intervals.parquet", index=False)
    pd.DataFrame({
        "id": ["zero-1", "zero-2"],
        "job_id": [1, 2],
        "lag_idx": [0, 0],
        "rho": [6.0, 8.0],
        "ifar_years": [0.1, 0.2],
        "significance": [1.0, 2.0],
        "gps_time": [1000.0, 1010.0],
    }).to_csv(tmp_path / "zero_lag.csv", index=False)
    with open(tmp_path / "far_rho.json", "w") as f:
        json.dump({
            "bins": [5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "far": [10.0, 5.0, 2.0, 1.0, 0.5, 0.25],
            "cum_events": [12, 9, 5, 3, 1, 1],
            "ranking_par": "rho",
        }, f)
    pd.DataFrame({
        "sim_sim_idx": [1, 2, 3],
        "sim_name": ["wfA", "wfA", "wfB"],
        "sim_hrss": [1e-22, 2e-22, 1e-21],
        "id": ["event-1", None, "event-2"],
    }).to_parquet(tmp_path / "matched.parquet", index=False)
    pd.DataFrame({
        "ifar": ["1yr", "1yr"],
        "waveform": ["wfA", "wfB"],
        "status": ["ok", "ok"],
        "hrss10": [1e-22, 2e-22],
        "hrss50": [3e-22, 4e-22],
        "hrss90": [5e-22, 6e-22],
        "chi2": [0.1, 0.2],
    }).to_csv(tmp_path / "fit.csv", index=False)
    _write_dummy_png(tmp_path / "plot.png")
    (tmp_path / "xgb_config.py").write_text("def update_config(*args):\n    return None\n")
    (tmp_path / "model.ubj").write_text("model")
    (tmp_path / "training_settings.cfg").write_text("training settings\n")
    (tmp_path / "training_output.out").write_text("training output\n")

    result = postproduction_report(
        work_dir=str(tmp_path),
        workflow_file="workflow.yaml",
        production_catalog_file="production.parquet",
        title="Tiny postproduction report",
        output_file="public/index.html",
        bkg={
            "scored_catalog": "bkg_scored.parquet",
            "binned_far_json": "far_rho.json",
            "progress_file": "progress.parquet",
            "intervals_file": "intervals.parquet",
            "zero_lag_progress_file": "zero_progress.parquet",
            "zero_lag_catalog_file": "production.parquet",
            "livetime": 1.0,
            "zero_lag_csv": "zero_lag.csv",
            "plots": ["plot.png", "missing.png"],
        },
        training={
            "bkg_catalog": "train_bkg.parquet",
            "bkg_progress_file": "progress.parquet",
            "bkg_intervals_file": "intervals.parquet",
            "sim_catalog": "train_sim.parquet",
            "model_file": "model.ubj",
            "config_file": "xgb_config.py",
            "training_settings_file": "training_settings.cfg",
            "training_output_file": "training_output.out",
        },
        simulation_runs=[{
            "label": "STDINJs mini",
            "scored_catalog": "sim_scored.parquet",
            "matched_file": "matched.parquet",
            "plots": ["plot.png"],
            "fit_parameter_files": ["fit.csv"],
        }],
        max_plot_points=5,
        max_bins=4,
        table_limit=3,
    )

    assert result["n_tabs"] == 5
    assert (tmp_path / "public" / "index.html").exists()
    assert (tmp_path / "public" / "report_data.json").exists()
    assert (tmp_path / "public" / "workflow.yaml").exists()
    assert (tmp_path / "public" / "workflow_diagram.html").exists()
    html = (tmp_path / "public" / "index.html").read_text()
    assert "Summary" in html
    assert "Run Parameters" in html
    assert "Post-Production Selection" in html
    assert "Live Time" in html
    assert "BKG" in html
    assert "Training" in html
    assert "Training Review Files" in html
    assert "training_settings.cfg" in html
    assert "training_output.out" in html
    assert "STDINJs mini" in html
    assert "Workflow / YAML" in html

    data = json.loads((tmp_path / "public" / "report_data.json").read_text())
    assert data["basic_info"]["run_parameters"]
    assert any(item["label"] == "Frequency band" for item in data["basic_info"]["run_parameters"])
    assert any(item["label"] == "Ranking statistic" for item in data["basic_info"]["selection_cuts"])
    bkg_split = next(item for item in data["basic_info"]["selection_cuts"] if item["label"] == "BKG split")
    assert "${" not in bkg_split["value"]
    assert "train: 0.5" in bkg_split["value"]
    assert any(item["label"] == "Zero lag" for item in data["basic_info"]["livetimes"])
    assert data["metadata"]["production_pycwb_version"] == "prod-1.2.3"
    assert data["metadata"]["config_summary"][0]["key"] == "ifo"
    assert data["workflow"]["workflow_artifact"]["href"] == "workflow.yaml"
    assert data["workflow"]["workflow_artifact"]["path"] == "workflow.yaml"
    assert data["workflow"]["diagram_html"]["href"] == "workflow_diagram.html"
    assert data["workflow"]["diagram_html"]["path"] == "workflow_diagram.html"
    assert data["bkg"]["livetime"]["seconds"] == 60.0
    assert data["bkg"]["livetime_source"] == "progress_file"
    assert data["bkg"]["zero_lag_livetime"]["livetime"]["seconds"] == 12.0
    assert data["bkg"]["zero_lag_livetime"]["livetime"]["days_label"] == "0.000 d"
    assert any(item["path"] == "../missing.png" for item in data["missing_artifacts"])
    far_fig = next(fig for fig in data["bkg"]["figures"] if fig["id"] == "far_rho_curve")
    assert len(far_fig["traces"][0]["x"]) <= 5
    assert data["simulation_runs"][0]["fit_tables"][0]["n_rows"] == 2
    assert any(item["label"] == "XGBoost training settings" for item in data["basic_info"]["review_links"])
    assert any(item["label"] == "XGBoost training output" for item in data["training"]["artifacts"])


def test_report_livetime_and_row_labels_are_compact_for_cards(tmp_path):
    live = _livetime_dict(480166.0)
    assert live["years_label"] == "0.015 yr"
    assert live["days_label"] == "5.557 d"
    assert live["seconds_label"] == "480166 s"
    assert live["zero_lag_compact_label"] == "0.015 yr / 5.557 d / 480166 s"
    assert "," not in live["seconds_label"]

    table = pa.table({"rho": pa.array(range(1234))})
    pq.write_table(table, tmp_path / "catalog.parquet")
    assert _parquet_file_info(str(tmp_path / "catalog.parquet"))["rows_label"] == "1,234"


def test_zero_lag_livetime_uses_production_catalog_metadata(tmp_path):
    _write_catalog_with_jobs(
        tmp_path / "production.parquet",
        [
            {"index": 1, "shift": [0, 0]},
            {"index": 2, "shift": [0, 1200]},
        ],
    )
    pd.DataFrame({
        "rho": [1.0],
        "job_id": [1],
        "lag_idx": [0],
    }).to_parquet(tmp_path / "zero_scored.parquet", index=False)
    pd.DataFrame({
        "job_id": [1, 2, 1],
        "lag_idx": [0, 0, 1],
        "livetime": [5.0, 100.0, 11.0],
        "status": ["completed", "completed", "completed"],
    }).to_parquet(tmp_path / "zero_progress.parquet", index=False)
    (tmp_path / "workflow.yaml").write_text("steps: []\n")

    postproduction_report(
        work_dir=str(tmp_path),
        workflow_file="workflow.yaml",
        production_catalog_file="production.parquet",
        output_file="public/index.html",
        bkg={
            "zero_lag_progress_file": "zero_progress.parquet",
            "zero_lag_catalog_file": "zero_scored.parquet",
        },
    )

    data = json.loads((tmp_path / "public" / "report_data.json").read_text())
    zero_lag = data["bkg"]["zero_lag_livetime"]
    assert zero_lag["livetime"]["seconds"] == 5.0
    assert zero_lag["n_jobs"] == 1
    assert zero_lag["warning"] == ""


def test_zero_lag_livetime_warns_when_shift_metadata_is_unavailable(tmp_path):
    progress = tmp_path / "zero_progress.parquet"
    pd.DataFrame({
        "job_id": [1, 2],
        "lag_idx": [0, 0],
        "livetime": [5.0, 7.0],
        "status": ["completed", "completed"],
    }).to_parquet(progress, index=False)

    summary = _zero_lag_livetime_summary(
        progress_path=str(progress),
        catalog_path=None,
    )

    assert summary["livetime"]["seconds"] == 12.0
    assert "shifted jobs may be included" in summary["warning"]


def test_report_uses_only_explicit_simulation_tabs(tmp_path):
    _write_catalog_with_metadata(tmp_path / "production.parquet")
    (tmp_path / "public").mkdir()
    (tmp_path / "workflow.yaml").write_text("steps: []\n")
    _write_dummy_png(tmp_path / "unrelated_simulation_plot.png")

    result = postproduction_report(
        work_dir=str(tmp_path),
        workflow_file="workflow.yaml",
        production_catalog_file="production.parquet",
        output_file="public/index.html",
        simulation_runs=None,
    )

    assert result["n_tabs"] == 4
    data = json.loads((tmp_path / "public" / "report_data.json").read_text())
    assert data["simulation_runs"] == []
    assert "unrelated_simulation_plot.png" not in (tmp_path / "public" / "index.html").read_text()
