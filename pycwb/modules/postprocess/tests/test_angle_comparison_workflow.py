import json
import math
import os

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pycwb.modules.postprocess.angle_comparison import plot_angle_error_comparison
from pycwb.modules.postprocess.generic_report import generic_web_report
from pycwb.modules.postprocess.multi_run import read_catalog_runs


def _injection(sim_idx, ra_deg, dec_deg, gps_time):
    return {
        "ra": math.radians(ra_deg),
        "dec": math.radians(dec_deg),
        "gps_time": gps_time,
        "parameters": json.dumps({"sim_idx": sim_idx, "trial_idx": 0}),
    }


def _write_catalog(path, recovered_ra_offset):
    injections = [
        _injection(0, 10.0, -20.0, 100.0),
        _injection(1, 200.0, 45.0, 130.0),
    ]
    rows = [
        {
            "id": [0],
            "ra": 10.0 + recovered_ra_offset,
            "dec": -20.0,
            "rho": 12.0,
            "net_cc": 0.8,
            "injection": injections[0],
        },
        # Lower-ranked duplicate: the plotting action must keep the rho=12 row.
        {
            "id": [1],
            "ra": 60.0,
            "dec": -20.0,
            "rho": 6.0,
            "net_cc": 0.6,
            "injection": injections[0],
        },
    ]
    jobs = [{"index": 0, "injections": injections}]
    metadata = {
        b"jobs": json.dumps(jobs).encode(),
        b"config": json.dumps({"ifo": ["L1", "H1", "V1"]}).encode(),
        b"pycwb_version": b"test",
    }
    table = pa.Table.from_pylist(rows).replace_schema_metadata(metadata)
    pq.write_table(table, path)


def test_multi_run_angle_plots_and_generic_report(tmp_path):
    _write_catalog(tmp_path / "far.parquet", recovered_ra_offset=10.0)
    _write_catalog(tmp_path / "near.parquet", recovered_ra_offset=1.0)

    combined = read_catalog_runs(
        work_dir=str(tmp_path),
        runs=[
            {"name": "reference", "catalog_file": "far.parquet"},
            {"name": "comparison", "catalog_file": "near.parquet"},
        ],
        output_file="tmp/combined.parquet",
        require_injections=True,
    )
    triggers = pd.read_parquet(combined["triggers_file"])
    injections = pd.read_parquet(combined["injections_file"])
    assert list(triggers["run_index"]) == [0, 0, 1, 1]
    assert list(triggers["run_name"]) == [
        "reference", "reference", "comparison", "comparison"
    ]
    assert list(triggers["run_label"]) == list(triggers["run_name"])
    assert list(triggers.index) == [0, 1, 2, 3]
    assert len(injections) == 4  # Includes sim 1, which was missed in both runs.

    plotted = plot_angle_error_comparison(
        work_dir=str(tmp_path),
        triggers_file=combined["triggers_file"],
        injections_file=combined["injections_file"],
        manifest_file=combined["manifest_file"],
        output_dir="public/plots",
    )
    assert plotted["n_runs"] == 2
    assert plotted["summary"][0]["n_injections"] == 2
    assert plotted["summary"][0]["n_recovered"] == 1
    assert plotted["summary"][1]["median_error_deg"] < plotted["summary"][0]["median_error_deg"]
    for key in ["map_file", "histogram_file", "summary_file", "data_file"]:
        assert os.path.exists(tmp_path / plotted[key])
    map_page = (tmp_path / plotted["map_file"]).read_text()
    assert "plotly_click" in map_page
    assert "shared-injection" in map_page
    assert "recovery-path" in map_page
    assert "selected-recovery-path" in map_page
    assert "selectedLon" in map_page
    assert "dimmedPointOpacity" in map_page
    assert "Click a shared injection to highlight its recoveries" in map_page

    report = generic_web_report(
        work_dir=str(tmp_path),
        plots=plotted["plots"],
        output_file="public/report/index.html",
    )
    page = (tmp_path / "public/report/index.html").read_text()
    assert report["n_plots"] == 2
    assert "Injected and recovered sky positions" in page
    assert "Angular-error distribution" in page
    assert "<table" not in page
    assert "<th>Run</th>" not in page
    assert len(list((tmp_path / "public/report/assets").glob("*.html"))) == 2


def test_angle_plot_rejects_changed_injection_truth(tmp_path):
    _write_catalog(tmp_path / "a.parquet", recovered_ra_offset=2.0)
    _write_catalog(tmp_path / "b.parquet", recovered_ra_offset=2.0)
    table = pq.read_table(tmp_path / "b.parquet")
    jobs = json.loads(table.schema.metadata[b"jobs"])
    jobs[0]["injections"][0]["ra"] += math.radians(0.1)
    metadata = {**table.schema.metadata, b"jobs": json.dumps(jobs).encode()}
    pq.write_table(table.replace_schema_metadata(metadata), tmp_path / "b.parquet")

    combined = read_catalog_runs(
        work_dir=str(tmp_path),
        runs=["a.parquet", "b.parquet"],
        output_file="tmp/combined.parquet",
        require_injections=True,
    )
    with pytest.raises(ValueError, match="Injection truth differs"):
        plot_angle_error_comparison(
            work_dir=str(tmp_path),
            triggers_file=combined["triggers_file"],
            injections_file=combined["injections_file"],
            manifest_file=combined["manifest_file"],
        )
