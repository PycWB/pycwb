from __future__ import annotations

import json

import numpy as np
import pandas as pd

from pycwb.modules.cwb_xgboost.read_data import apply_user_ranking_statistics
from pycwb.modules.postprocess import evaluate
from pycwb.modules.postprocess.train_xgboost import (
    _compact_training_frame,
    _read_and_concat,
    _training_working_columns,
    _xgb_required_input_columns,
)


def test_apply_user_ranking_statistics_getrhor(tmp_path):
    config = tmp_path / "xgb_config.py"
    config.write_text(
        """
import numpy as np

def getrhor(xdp, search):
    xdp["MLstat_stretched"] = xdp["MLstat"]
    xdp.loc[xdp.MLstat_stretched > 0.99999999, "MLstat_stretched"] = 0.99999999
    xdp["rhor"] = -np.log(1.0 - xdp["MLstat_stretched"])
    return xdp
""".strip()
    )

    df = pd.DataFrame({"MLstat": [0.5, 0.999999999]})
    result = apply_user_ranking_statistics(df, "blf", str(config))

    assert list(result["MLstat_stretched"]) == [0.5, 0.99999999]
    np.testing.assert_allclose(result["rhor"], [-np.log(0.5), -np.log(1e-8)])


def test_evaluate_far_rho_can_rank_by_config_defined_rhor(tmp_path, monkeypatch):
    catalog = tmp_path / "catalog.parquet"
    model = tmp_path / "model.ubj"
    config = tmp_path / "xgb_config.py"
    far_json = tmp_path / "far.json"
    scored = tmp_path / "scored.parquet"

    pd.DataFrame({
        "id": [1, 2, 3],
        "rho": [30.0, 10.0, 20.0],
        "job_id": [1, 1, 1],
        "lag_idx": [1, 1, 1],
    }).to_parquet(catalog, index=False)
    model.write_text("dummy")
    config.write_text(
        """
import numpy as np

def update_config(*args):
    return None

def getrhor(xdp, search):
    xdp["MLstat_stretched"] = xdp["MLstat"]
    xdp.loc[xdp.MLstat_stretched > 0.99999999, "MLstat_stretched"] = 0.99999999
    xdp["rhor"] = -np.log(1.0 - xdp["MLstat_stretched"])
    return xdp
""".strip()
    )

    class _Booster:
        feature_names = []

    class _Classifier:
        def load_model(self, path):
            return None

        def get_booster(self):
            return _Booster()

        def predict_proba(self, X):
            probs = np.array([0.2, 0.95, 0.6])[:len(X)]
            return np.column_stack([1.0 - probs, probs])

    monkeypatch.setattr(evaluate.xgb, "XGBClassifier", _Classifier)
    monkeypatch.setattr(
        "pycwb.modules.cwb_xgboost.read_data.preprocess_events",
        lambda df, nifo, ML_options, ML_caps: df,
    )

    result = evaluate.evaluate_far_rho(
        work_dir=str(tmp_path),
        catalog_file="catalog.parquet",
        model_file="model.ubj",
        livetime=10.0,
        config_file="xgb_config.py",
        ranking_par="rhor",
        output_file="far.json",
        exclude_zero_lag=False,
        scored_catalog="scored.parquet",
    )

    scored_df = pd.read_parquet(scored)
    assert {"xgb_prob", "MLstat", "MLstat_stretched", "rhor"} <= set(scored_df.columns)
    assert scored_df.loc[1, "rhor"] == max(scored_df["rhor"])
    assert result["far_rho"][0]["ranking_par"] == "rhor"
    assert result["far_rho"][0]["ranking_value"] == scored_df.loc[1, "rhor"]
    assert result["far_rho"][0]["rho"] == scored_df.loc[1, "rhor"]

    written = json.loads(far_json.read_text())
    assert written[0]["ranking_par"] == "rhor"


def test_score_catalog_streams_zero_lag_with_config_defined_rhor(tmp_path, monkeypatch):
    catalog = tmp_path / "catalog.parquet"
    model = tmp_path / "model.ubj"
    config = tmp_path / "xgb_config.py"
    output = tmp_path / "zero_lag_scored.parquet"

    pd.DataFrame({
        "id": [1, 2, 3, 4],
        "rho": [30.0, 10.0, 20.0, 40.0],
        "job_id": [1, 1, 1, 1],
        "lag_idx": [0, 1, 0, 2],
    }).to_parquet(catalog, index=False)
    model.write_text("dummy")
    config.write_text(
        """
import numpy as np

def update_config(*args):
    return None

def getrhor(xdp, search):
    xdp["MLstat_stretched"] = xdp["MLstat"]
    xdp["rhor"] = -np.log(1.0 - xdp["MLstat_stretched"])
    return xdp
""".strip()
    )

    class _Booster:
        feature_names = []

    class _Classifier:
        def load_model(self, path):
            return None

        def get_booster(self):
            return _Booster()

        def predict_proba(self, X):
            probs = np.linspace(0.2, 0.8, len(X))
            return np.column_stack([1.0 - probs, probs])

    monkeypatch.setattr(evaluate.xgb, "XGBClassifier", _Classifier)
    monkeypatch.setattr(
        "pycwb.modules.cwb_xgboost.read_data.preprocess_events",
        lambda df, nifo, ML_options, ML_caps: df,
    )

    result = evaluate.score_catalog(
        work_dir=str(tmp_path),
        catalog_file="catalog.parquet",
        model_file="model.ubj",
        config_file="xgb_config.py",
        output_file="zero_lag_scored.parquet",
        lag_selection="zero_lag",
        batch_size=2,
    )

    scored = pd.read_parquet(output)
    assert result["n_input"] == 4
    assert result["n_scored"] == 2
    assert list(scored["id"]) == [1, 3]
    assert {"xgb_prob", "MLstat", "rhor"} <= set(scored.columns)


def test_score_mdc_catalog_uses_config_defined_rhor_for_ifar(tmp_path, monkeypatch):
    mdc_catalog = tmp_path / "mdc.parquet"
    bkg_scored = tmp_path / "bkg_scored.parquet"
    model = tmp_path / "model.ubj"
    config = tmp_path / "xgb_config.py"
    output = tmp_path / "mdc.csv"

    pd.DataFrame({
        "id": ["mdc-1", "mdc-2", "mdc-3"],
        "rho": [10.0, 11.0, 12.0],
        "job_id": [1, 1, 1],
        "lag_idx": [0, 0, 0],
        "gps_time": [1.0, 2.0, 3.0],
        "ifar": [0.0, 0.0, 0.0],
    }).to_parquet(mdc_catalog, index=False)

    bkg_probs = np.array([0.99, 0.90, 0.80, 0.70])
    pd.DataFrame({
        "id": ["bkg-1", "bkg-2", "bkg-3", "bkg-4"],
        "job_id": [1, 1, 1, 1],
        "lag_idx": [1, 1, 1, 1],
        "xgb_prob": bkg_probs,
        "rhor": -np.log(1.0 - bkg_probs),
    }).to_parquet(bkg_scored, index=False)

    model.write_text("dummy")
    config.write_text(
        """
import numpy as np

def update_config(*args):
    return None

def getrhor(xdp, search):
    xdp["MLstat_stretched"] = xdp["MLstat"]
    xdp["rhor"] = -np.log(1.0 - xdp["MLstat_stretched"])
    return xdp
""".strip()
    )

    class _Booster:
        feature_names = []

    class _Classifier:
        def load_model(self, path):
            return None

        def get_booster(self):
            return _Booster()

        def predict_proba(self, X):
            probs = np.array([0.995, 0.95, 0.85])[:len(X)]
            return np.column_stack([1.0 - probs, probs])

    monkeypatch.setattr(evaluate.xgb, "XGBClassifier", _Classifier)
    monkeypatch.setattr(
        "pycwb.modules.cwb_xgboost.read_data.preprocess_events",
        lambda df, nifo, ML_options, ML_caps: df,
    )

    livetime = 4 * 31557600
    result = evaluate.score_mdc_catalog(
        work_dir=str(tmp_path),
        mdc_catalog="mdc.parquet",
        model_file="model.ubj",
        bkg_scored_catalog="bkg_scored.parquet",
        livetime=livetime,
        config_file="xgb_config.py",
        ranking_par="rhor",
        ifar_threshold=str(2 * 31557600),
        output_csv="mdc.csv",
    )

    detections = pd.read_csv(output)
    assert result["ranking_par"] == "rhor"
    assert result["n_detections"] == 3
    assert {"rhor", "xgb_prob", "ifar", "ifar_yr", "ifar_years", "ifar_sec"} <= set(detections.columns)
    assert list(detections["id"]) == ["mdc-1", "mdc-2", "mdc-3"]
    assert np.isinf(detections.loc[0, "ifar_yr"])
    assert detections.loc[1, "ifar"] == detections.loc[1, "ifar_yr"]
    np.testing.assert_allclose(detections.loc[1, "ifar_yr"], 4.0)
    np.testing.assert_allclose(detections.loc[2, "ifar_yr"], 2.0)


def test_score_mdc_catalog_keeps_events_in_background_rank_gaps(tmp_path, monkeypatch):
    mdc_catalog = tmp_path / "mdc.parquet"
    bkg_scored = tmp_path / "bkg_scored.parquet"
    model = tmp_path / "model.ubj"
    config = tmp_path / "xgb_config.py"
    output = tmp_path / "mdc.csv"

    pd.DataFrame({
        "id": ["louder-than-bkg", "in-gap", "below-gap"],
        "rho": [10.0, 9.0, 8.0],
        "job_id": [1, 1, 1],
        "lag_idx": [0, 0, 0],
        "gps_time": [1.0, 2.0, 3.0],
    }).to_parquet(mdc_catalog, index=False)

    pd.DataFrame({
        "id": ["bkg-1", "bkg-2"],
        "job_id": [1, 1],
        "lag_idx": [1, 1],
        "rhor": [3.0, 1.0],
    }).to_parquet(bkg_scored, index=False)

    model.write_text("dummy")
    config.write_text(
        """
import numpy as np

def update_config(*args):
    return None

def getrhor(xdp, search):
    xdp["rhor"] = -np.log(1.0 - xdp["MLstat"])
    return xdp
""".strip()
    )

    desired_rhor = np.array([3.5, 2.5, 0.5])

    class _Booster:
        feature_names = []

    class _Classifier:
        def load_model(self, path):
            return None

        def get_booster(self):
            return _Booster()

        def predict_proba(self, X):
            probs = 1.0 - np.exp(-desired_rhor[:len(X)])
            return np.column_stack([1.0 - probs, probs])

    monkeypatch.setattr(evaluate.xgb, "XGBClassifier", _Classifier)
    monkeypatch.setattr(
        "pycwb.modules.cwb_xgboost.read_data.preprocess_events",
        lambda df, nifo, ML_options, ML_caps: df,
    )

    result = evaluate.score_mdc_catalog(
        work_dir=str(tmp_path),
        mdc_catalog="mdc.parquet",
        model_file="model.ubj",
        bkg_scored_catalog="bkg_scored.parquet",
        livetime=10 * 31557600,
        config_file="xgb_config.py",
        ranking_par="rhor",
        ifar_threshold="10yr",
        output_csv="mdc.csv",
    )

    detections = pd.read_csv(output)
    assert result["n_detections"] == 2
    assert list(detections["id"]) == ["louder-than-bkg", "in-gap"]
    np.testing.assert_allclose(detections.loc[detections["id"] == "in-gap", "ifar_yr"], [10.0])


def test_xgb_training_helpers_project_and_compact_catalog_columns(tmp_path):
    from pycwb.modules.cwb_xgboost.config import xgb_config

    catalog = tmp_path / "catalog.parquet"
    pd.DataFrame({
        "id": [1, 2],
        "job_id": [10, 11],
        "lag_idx": [1, 2],
        "trial_idx": [0, 0],
        "gps_time": [100.0, 101.0],
        "ifar": [0.0, 0.0],
        "rho": [8.0, 9.0],
        "net_cc": [0.9, 0.8],
        "coherent_energy": [100.0, 121.0],
        "coherent_energy_norm": [5.0, 6.0],
        "penalty": [1.1, 1.2],
        "likelihood": [10.0, 11.0],
        "q_veto": [4.0, 9.0],
        "q_factor": [6.0, 7.0],
        "Lveto2": [0.1, 0.2],
        "signal_energy_H1": [30.0, 40.0],
        "signal_energy_L1": [20.0, 25.0],
        "data_energy_H1": [3.0, 4.0],
        "data_energy_L1": [2.0, 5.0],
        "noise_rms_H1": [1.0, 1.1],
        "noise_rms_L1": [1.2, 1.3],
        "sim_sim_idx": [1, 2],
        "sim_vetoed_cat0": [False, False],
        "sim_vetoed_cat1": [False, False],
        "sim_vetoed_cat2": [False, False],
        "sim_across_segments": [False, False],
        "sky_error_regions": [[1, 2, 3], [4, 5, 6]],
        "large_unused_text": ["x" * 100, "y" * 100],
    }).to_parquet(catalog, index=False)

    _, ML_list, ML_caps, ML_balance, ML_options = xgb_config("blf", 2)
    projected = _xgb_required_input_columns(
        [str(catalog)],
        nifo=2,
        ML_options=ML_options,
        ML_caps=ML_caps,
        ML_balance=ML_balance,
        ML_list=ML_list,
        include_training_metadata=True,
    )

    assert {"coherent_energy", "coherent_energy_norm", "penalty", "q_veto", "q_factor"} <= set(projected)
    assert {"signal_energy_H1", "signal_energy_L1", "data_energy_H1", "data_energy_L1"} <= set(projected)
    assert {"noise_rms_H1", "noise_rms_L1", "Lveto2", "sim_sim_idx"} <= set(projected)
    assert "sky_error_regions" not in projected
    assert "large_unused_text" not in projected

    read_df = _read_and_concat([str(catalog)], "TEST", columns=projected)
    assert "coherent_energy" in read_df.columns
    assert "large_unused_text" not in read_df.columns

    processed = pd.DataFrame({
        "norm": np.array([5.0, 6.0], dtype=np.float64),
        "netcc0": np.array([0.9, 0.8], dtype=np.float64),
        "penalty": np.array([1.1, 1.2], dtype=np.float64),
        "Lveto2": np.array([0.1, 0.2], dtype=np.float64),
        "chunk": np.array([0, 0], dtype=np.int64),
        "sSNR0/likelihood": np.array([1.0, 2.0], dtype=np.float64),
        "rho0": np.array([8.0, 9.0], dtype=np.float64),
        "Qa": np.array([2.0, 3.0], dtype=np.float64),
        "Qp": np.array([1.5, 1.6], dtype=np.float64),
        "classifier": np.array([0, 1], dtype=np.int64),
        "large_unused_text": ["x", "y"],
    })
    working_columns = _training_working_columns(ML_list, ML_caps, processed, processed)
    compact = _compact_training_frame(processed, working_columns)

    assert "large_unused_text" not in compact.columns
    assert compact["classifier"].dtype == np.dtype("int8")
    assert compact["norm"].dtype == np.dtype("float32")
