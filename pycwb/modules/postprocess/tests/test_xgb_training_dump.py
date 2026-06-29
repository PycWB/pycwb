from __future__ import annotations

from pycwb.modules.cwb_xgboost.training import (
    dump_training_output,
    dump_training_settings,
)


class _FakeBooster:
    feature_names = ["rho0", "netcc0"]

    def num_boosted_rounds(self):
        return 3

    def get_score(self, importance_type="weight"):
        return {"rho0": 2.0, "netcc0": 1.0}


class _FakeClassifier:
    best_iteration = 2
    best_score = 0.91

    def get_booster(self):
        return _FakeBooster()

    def evals_result(self):
        return {
            "validation_0": {
                "logloss": [0.7, 0.4, 0.2],
                "auc": [0.6, 0.8, 0.91],
            }
        }


def test_dump_training_settings_and_output(tmp_path):
    settings_file = dump_training_settings(
        str(tmp_path / "model"),
        {"max_depth": 4, "seed": 150914},
        ["rho0", "netcc0"],
        {"rho0": 40},
        {"cuts(training)": "rho0>6.5"},
        {"rho0(define)": 1},
        metadata={"train_rows": 12, "eval_rows": 3},
    )

    output_file = dump_training_output(
        _FakeClassifier(),
        str(tmp_path / "model"),
        training_summary={"elapsed_seconds": 1.25, "model_file": "model.ubj"},
    )

    settings_text = (tmp_path / "model.cfg").read_text()
    output_text = (tmp_path / "model.out").read_text()

    assert settings_file == str(tmp_path / "model.cfg")
    assert "[metadata]" in settings_text
    assert "train_rows = 12" in settings_text
    assert "[xgb_params]" in settings_text
    assert "'rho0'" in settings_text

    assert output_file == str(tmp_path / "model.out")
    assert "[training_summary]" in output_text
    assert "[tree_statistics]" in output_text
    assert "[evals_result.validation_0.auc]" in output_text
    assert "[feature_importance_gain]" in output_text
