import numpy as np
import pandas as pd
import pytest

from pycwb.modules.postprocess import evaluate


def test_ranking_hook_receives_prediction_features(tmp_path, monkeypatch):
    config = tmp_path / "ranking.py"
    config.write_text(
        'def getrhor(events, search):\n    events["rhor"] = events.ecor ** .5 * events.Qa * events.MLstat\n    return events\n'
    )
    raw = pd.DataFrame(
        {"id": ["a"], "coherent_energy": [25.0], "rho": [8.0]}, index=[17]
    )
    processed = pd.DataFrame({"ecor": [25.0], "Qa": [2.0], "rho0": [5.0]}, index=[17])
    monkeypatch.setattr(
        evaluate,
        "_preprocess_for_scoring",
        lambda *args: (processed, processed[["ecor"]], {}, str(config)),
    )

    class Model:
        def predict_proba(self, features):
            assert features.index.tolist() == [17]
            return np.array([[0.25, 0.75]])

    result = evaluate._score_catalog_dataframe(
        raw, 2, "blf", str(config), str(tmp_path), Model()
    )
    assert result.loc[17, "rhor"] == 7.5
    assert result.loc[17, "rho"] == 8.0
    assert result.loc[17, "coherent_energy"] == 25.0
    assert result.loc[17, "id"] == "a"
    assert "rhor" not in raw


def test_root_prediction_cut_uses_stored_rho_and_double_arithmetic():
    from pycwb.modules.postprocess.prediction_cuts import prediction_mask

    frame = pd.DataFrame(
        {
            "ecor": np.array([36.0, 64.0], dtype="float32"),
            "penalty": [1.0, 1.0],
            "rho0_std": [8.0, 9.0],
            "rho0": [6.0, 8.0],
        }
    )
    mask = prediction_mask(
        frame,
        "sqrt(ecor/(1+penalty*(TMath::Max((float)1,(float)penalty)-1)))>6.5 && rho[0]>8",
    )
    assert mask.tolist() == [False, True]
    assert prediction_mask(frame, "rho[0]>7").tolist() == [True, True]
    assert prediction_mask(frame, "rho0>7").tolist() == [False, True]


def test_q_features_match_cwb_array_flattening_precision():
    from pycwb.modules.cwb_xgboost.read_data import preprocess_events

    events = pd.DataFrame(
        {
            "qveto": np.array([1.234567], dtype="f4"),
            "qfactor": np.array([2.345678], dtype="f4"),
            "ecor": np.array([91.23456], dtype="f4"),
        }
    )
    expected_qa = np.sqrt(events.qveto.astype("f8"))
    expected_qp = events.qfactor.astype("f8") / (
        2 * np.sqrt(np.log10(np.minimum(200, events.ecor)))
    )
    result = preprocess_events(events, 2, {"readfile(vars)": [], "Qp(index)": 1}, {})
    np.testing.assert_array_equal(result.Qa, expected_qa)
    np.testing.assert_array_equal(result.Qp, expected_qp)


@pytest.mark.parametrize("expression", [
    "sqrt(ecor/(1+penalty*(max(1.0,penalty)-1)))>6.5",
    "min(1.0, penalty)<0.5",
    "max(penalty, min(1.0, ecor))>1.5",
    "max_value > max (1.0, penalty)",
])
def test_prediction_min_max_match_root_and_training_selection(expression):
    from pycwb.modules.cwb_xgboost.read_data import apply_training_cuts
    from pycwb.modules.postprocess.prediction_cuts import prediction_mask

    frame = pd.DataFrame({
        "ecor": [25., 100., 400., 81.],
        "penalty": [1., 1., 2., .25],
        "max_value": [0., 2., 3., 0.],
    }, index=[3, 8, 13, 21])
    root_expression = expression.replace("max(", "TMath::Max(").replace(
        "min(", "TMath::Min("
    ).replace("max (", "TMath::Max(")
    mask = prediction_mask(frame, expression)
    pd.testing.assert_series_equal(mask, prediction_mask(frame, root_expression))
    pd.testing.assert_frame_equal(
        frame.loc[mask].reset_index(drop=True), apply_training_cuts(frame, expression)
    )
