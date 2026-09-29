import json

import numpy as np
import pandas as pd
import pytest

from pycwb.modules.postprocess.cwb_report import attach_cwb_ifar, compare_cwb_report
from pycwb.modules.postprocess.model_io import import_cwb_model
from pycwb.modules.postprocess.simulation_report import simulation_efficiency


def test_efficiency_keeps_misses_prediction_rejections_and_equal_threshold(tmp_path):
    pd.DataFrame(
        {
            "sim_sim_idx": range(4),
            "sim_name": ["SG"] * 4,
            "sim_hrss": [1.0, 1.0, 2.0, 2.0],
            "id": ["a", None, "b", "c"],
            "rho_alt": [8.0, np.nan, 9.0, 10.0],
        }
    ).to_parquet(tmp_path / "matched.parquet")
    pd.DataFrame({"id": ["a", "c"], "rhor": [5.0, 6.0]}).to_parquet(
        tmp_path / "scores.parquet"
    )
    result = simulation_efficiency(
        tmp_path,
        "matched.parquet",
        "eff",
        scored_file="scores.parquet",
        ranking_par="rhor",
        threshold=5.0,
    )
    assert result["n_injected"] == 4
    assert result["n_detected"] == 1
    curve = pd.read_csv(tmp_path / "eff/efficiency.csv")
    assert curve.n_injected.tolist() == [2, 2]
    assert curve.n_detected.tolist() == [0, 1]
    assert result["waveforms"][0]["hrss50"] == 2.0
    assert result["waveforms"][0]["status"] == "measured"
    ref = tmp_path / "reference"
    ref.mkdir()
    (ref / "eff_SG.txt").write_text("1 0 2 0\n2 1 2 0.5\n")
    checked = compare_cwb_report(
        tmp_path,
        "checks.json",
        efficiency_file="eff/efficiency.csv",
        simulation_reference_dir="reference",
    )
    assert checked["checks"][0]["status"] == "PASS"
    (ref / "eff_SG.txt").write_text("1 1 2 0.5\n2 1 2 0.5\n")
    with pytest.raises(ValueError, match="comparison failed"):
        compare_cwb_report(
            tmp_path,
            "checks.json",
            efficiency_file="eff/efficiency.csv",
            simulation_reference_dir="reference",
        )
    assert (
        json.loads((tmp_path / "checks.json").read_text())["checks"][0]["status"]
        == "FAIL"
    )


def test_legacy_ifar_boundaries_and_no_infinite_tail(tmp_path):
    pd.DataFrame({"rhor": [0.0, 1.5, 2.0, 2.5, 100.0]}).to_parquet(
        tmp_path / "events.parquet"
    )
    (tmp_path / "far.txt").write_text("1 0.1\n2 0.05\n3 0\n")
    attach_cwb_ifar(tmp_path, "events.parquet", "far.txt", "scored.parquet")
    x = pd.read_parquet(tmp_path / "scored.parquet")
    np.testing.assert_allclose(x.far_hz, [0.1, 0.1, 0.075, 0.05, 0.05])
    np.testing.assert_allclose(x.ifar, [10, 10, 1 / 0.075, 20, 20])


def test_pickle_import_requires_explicit_trust_before_open(tmp_path):
    with pytest.raises(ValueError, match="trusted_pickle"):
        import_cwb_model(tmp_path, "does-not-exist", "model.json")


def test_simulation_import_unique_before_time_cut_and_full_denominator(tmp_path):
    uproot = pytest.importorskip("uproot")
    from pycwb.modules.postprocess.root_simulation import import_cwb_simulation

    with uproot.recreate(tmp_path / "sim.root") as root:
        root.mktree(
            "mdc",
            {
                "time": np.array([[100.0, 100.0], [200.0, 200.0], [300.0, 300.0]]),
                "type": np.ones(3, dtype="i4"),
                "name": ["SG"] * 3,
                "strain": np.array([1e-21] * 3),
                "factor": np.ones(3),
            },
        )
        root.mktree(
            "waveburst",
            {
                "ndim": np.full(3, 2, dtype="i4"),
                "run": np.ones(3, dtype="i4"),
                "time": np.array(
                    [
                        [100.0, 100.0, 100.0, 100.0],
                        [101.0, 101.0, 100.0, 100.0],
                        [200.0, 200.0, 200.0, 200.0],
                    ]
                ),
                "type": np.ones((3, 2), dtype="i4"),
                "factor": np.ones(3),
                "rho": np.array([[8.0, 8.0], [9.0, 9.0], [10.0, 10.0]], dtype="f4"),
                "slag": np.zeros((3, 3)),
            },
        )
    result = import_cwb_simulation(tmp_path, "sim.root", ["L1", "H1"], "out")
    assert [
        result[k] for k in ["injected", "raw", "unique", "recovered", "missed"]
    ] == [3, 3, 2, 1, 2]
    matched = pd.read_parquet(result["matched_file"])
    assert matched.sim_sim_idx.tolist() == [0, 1, 2]
    assert matched.id.notna().tolist() == [False, True, False]
