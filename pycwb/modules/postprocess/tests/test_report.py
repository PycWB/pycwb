"""Tests for post-production report actions."""

from __future__ import annotations

import pandas as pd

from pycwb.modules.postprocess import fake_openbox
from pycwb.modules.postprocess import far
from pycwb.modules.postprocess import report_plots
from pycwb.modules.postprocess import zero_lag


def test_far_rho_plot_streams_histogram_without_materializing_catalog(tmp_path, monkeypatch):
    catalog = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "job_id": [1, 2, 3, 4],
        "lag_idx": [0, 1, 2, 3],
        "rho": [99.0, 1.0, 1.2, 2.0],
        "time_lag_L1": [0.0, 0.0, 0.1, 0.2],
        "time_lag_H1": [0.0, 0.2, 0.0, 0.1],
        "segment_lag_L1": [0.0, 0.0, 0.0, 1200.0],
        "segment_lag_H1": [0.0, 1200.0, -1200.0, 0.0],
    })
    catalog.to_parquet(tmp_path / "catalog.parquet", index=False)
    (tmp_path / "progress.parquet").touch()
    monkeypatch.setattr(report_plots, "plot_far_rho", lambda *args, **kwargs: None)
    monkeypatch.setattr(report_plots, "plot_n_events", lambda *args, **kwargs: None)

    result = far.far_rho_plot(
        work_dir=str(tmp_path),
        catalog_file="catalog.parquet",
        progress_file="progress.parquet",
        livetime=31557600.0,
        ranking_par="rho",
        bin_size=0.5,
        output_dir="public",
        exclude_zero_lag=True,
    )

    data = result["far_rho"]
    assert sum(data["n_events"]) == 3
    assert data["cum_events"][0] == 3
    assert data["ranking_par"] == "rho"
    assert (tmp_path / "public" / "far_rho.json").exists()
    loudest = pd.read_csv(tmp_path / "public" / "loudest_background_triggers.csv")
    assert result["loudest_background_triggers"]["n"] == 3
    assert result["loudest_background_triggers"]["csv"].endswith("loudest_background_triggers.csv")
    assert loudest["bkg_rank"].tolist() == [1, 2, 3]
    assert loudest["rho"].tolist() == [2.0, 1.2, 1.0]
    assert loudest["id"].tolist() == [4, 3, 2]
    assert {"far_attached", "ifar_years"}.issubset(loudest.columns)


def test_zero_lag_report_streams_only_zero_lag_rows(tmp_path, monkeypatch):
    catalog = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "job_id": [1, 2, 3, 4],
        "lag_idx": [0, 0, 1, 0],
        "rho": [8.0, 9.0, 10.0, 11.0],
        "ifar": [0.0, 0.0, 0.0, 0.0],
        "gps_time": [100.0, 200.0, 300.0, 400.0],
        "net_cc": [0.8, 0.9, 0.7, 0.95],
        "likelihood": [4.0, 5.0, 6.0, 7.0],
        "coherent_energy": [10.0, 11.0, 12.0, 13.0],
        "time_lag_L1": [0.0, 0.0, 0.1, 0.0],
        "time_lag_H1": [0.0, 0.0, 0.1, 0.0],
        "segment_lag_L1": [0.0, 0.0, 1200.0, 0.0],
        "segment_lag_H1": [0.0, 0.0, 0.0, 1200.0],
    })
    progress = pd.DataFrame({
        "job_id": [1, 2, 3, 4],
        "trial_idx": [0, 0, 0, 0],
        "lag_idx": [0, 0, 1, 0],
        "n_triggers": [1, 1, 1, 1],
        "livetime": [10.0, 20.0, 30.0, 40.0],
        "timestamp": [0.0, 0.0, 0.0, 0.0],
        "status": ["completed", "completed", "completed", "completed"],
    })
    catalog.to_parquet(tmp_path / "catalog.parquet", index=False)
    progress.to_parquet(tmp_path / "progress.parquet", index=False)
    plot_calls = []
    monkeypatch.setattr(report_plots, "plot_zero_lag", lambda *args, **kwargs: plot_calls.append(args[0].copy()))

    result = zero_lag.zero_lag_report(
        work_dir=str(tmp_path),
        catalog_file="catalog.parquet",
        progress_file="progress.parquet",
        far_rho_data={"bins": [0.0, 10.0, 20.0], "far": [1.0, 0.1, 0.01]},
        ranking_par="rho",
        output_dir="public",
    )

    triggers = pd.read_csv(tmp_path / "public" / "zero_lag_triggers.csv")
    assert result["zero_lag_n"] == 2
    assert set(triggers["id"]) == {1, 2}
    assert {"far_attached", "ifar_years", "significance", "p_value"}.issubset(triggers.columns)
    assert len(plot_calls) == 1


def test_fake_openbox_report_selects_intervals_and_matching_triggers(tmp_path, monkeypatch):
    catalog = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "job_id": [10, 11, 20, 21],
        "lag_idx": [1, 1, 2, 2],
        "rho": [8.0, 9.0, 10.0, 11.0],
        "segment_lag_L1": [0.0, 0.0, 0.0, 0.0],
        "segment_lag_H1": [1200.0, 1200.0, -1200.0, -1200.0],
    })
    intervals = pd.DataFrame({
        "shift_key": ["0,1200", "0,-1200"],
        "lag_idx": [1, 2],
        "livetime": [100.0, 200.0],
        "n_rows": [2, 2],
        "n_jobs": [2, 2],
        "shift_0": [0.0, 0.0],
        "shift_1": [1200.0, -1200.0],
    })
    catalog.to_parquet(tmp_path / "far.parquet", index=False)
    intervals.to_csv(tmp_path / "far_intervals.csv", index=False)
    plot_calls = []
    monkeypatch.setattr(report_plots, "plot_zero_lag", lambda *args, **kwargs: plot_calls.append(kwargs))

    result = fake_openbox.fake_openbox_report(
        work_dir=str(tmp_path),
        catalog_file="far.parquet",
        intervals_file="far_intervals.csv",
        far_rho_data={"bins": [0.0, 10.0, 20.0], "far": [1.0, 0.1, 0.01]},
        ranking_par="rho",
        output_dir="public",
        fake_openbox_n=2,
        fake_openbox_seed=7,
        exclude_zero_lag=False,
    )

    selected_intervals = pd.read_csv(tmp_path / "public" / "fake_openbox_intervals.csv")
    selected_triggers = pd.read_csv(tmp_path / "public" / "fake_openbox_triggers.csv")
    selected_keys = set(zip(selected_intervals["shift_key"], selected_intervals["lag_idx"]))
    trigger_keys = set(zip(selected_triggers["shift_key"], selected_triggers["lag_idx"]))

    assert result["fake_openbox_n_intervals"] == 2
    assert selected_keys
    assert trigger_keys == selected_keys
    assert result["fake_openbox_n"] == 4
    assert result["livetime"] == float(selected_intervals["livetime"].sum())
    assert len(result["reports"]) == 2
    assert len(plot_calls) == 2
    assert all("slag " in call["plot_label"] and "lag " in call["plot_label"] for call in plot_calls)
    assert (tmp_path / "public" / "fake_openbox_01_triggers.csv").exists()
    assert (tmp_path / "public" / "fake_openbox_02_triggers.csv").exists()
    assert {"shift_key", "far_attached", "ifar_years", "significance", "p_value"}.issubset(
        selected_triggers.columns
    )
    for report_item in result["reports"]:
        interval_triggers = pd.read_csv(report_item["triggers_csv"])
        assert {"shift_key", "far_attached", "ifar_years", "significance", "p_value"}.issubset(
            interval_triggers.columns
        )


def test_fake_openbox_report_matches_shift_columns_by_config_ifo_order(tmp_path, monkeypatch):
    catalog = pd.DataFrame({
        "id": [1, 2],
        "job_id": [10, 11],
        "lag_idx": [1, 1],
        "rho": [8.0, 9.0],
        "segment_lag_H1": [1200.0, 1200.0],
        "segment_lag_L1": [0.0, 0.0],
    })
    intervals = pd.DataFrame({
        "shift_key": ["0,1200"],
        "lag_idx": [1],
        "livetime": [100.0],
        "n_rows": [2],
        "n_jobs": [2],
        "shift_0": [0.0],
        "shift_1": [1200.0],
    })
    catalog.to_parquet(tmp_path / "far.parquet", index=False)
    intervals.to_csv(tmp_path / "far_intervals.csv", index=False)

    monkeypatch.setattr(report_plots, "plot_zero_lag", lambda *args, **kwargs: None)

    result = fake_openbox.fake_openbox_report(
        work_dir=str(tmp_path),
        catalog_file="far.parquet",
        intervals_file="far_intervals.csv",
        far_rho_data={"bins": [0.0, 10.0], "far": [1.0, 0.1]},
        ranking_par="rho",
        output_dir="public",
        ifo_order=["L1", "H1"],
        fake_openbox_n=1,
        fake_openbox_seed=7,
        exclude_zero_lag=False,
    )

    selected_triggers = pd.read_csv(tmp_path / "public" / "fake_openbox_triggers.csv")
    assert result["fake_openbox_n"] == 2
    assert selected_triggers["id"].tolist() == [1, 2]
