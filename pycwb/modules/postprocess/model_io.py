"""Explicit model-format import for legacy cWB training outputs."""

import json
from pathlib import Path

from pycwb.post_production.action_spec import action_spec


CATALOG_PREPROCESSING_VERSION = "cwb-compatible-v1"
_PREPROCESSING_ATTRIBUTE = "pycwb_catalog_preprocessing"


def _record_catalog_preprocessing(model):
    """Persist the catalog field conventions alongside the trained trees."""
    booster = model.get_booster()
    version = booster.attr(_PREPROCESSING_ATTRIBUTE)
    if version not in (None, CATALOG_PREPROCESSING_VERSION):
        raise ValueError(f"Incompatible model catalog preprocessing: {version!r}")
    booster.set_attr(**{_PREPROCESSING_ATTRIBUTE: CATALOG_PREPROCESSING_VERSION})


def _validate_catalog_preprocessing(booster, options):
    """Reject incompatible or unidentified feature semantics before scoring."""
    version = booster.attr(_PREPROCESSING_ATTRIBUTE)
    if version is None:
        version = options.get("model_preprocessing")
    if version != CATALOG_PREPROCESSING_VERSION:
        raise ValueError(
            f"Model catalog preprocessing is {version!r}; expected "
            f"{CATALOG_PREPROCESSING_VERSION!r}. Retrain older PycWB native-catalog "
            "models: norm, sSNR and detector-indexed features have changed. "
            "For an unversioned model independently verified to use the current "
            "cWB conventions, set ML_options['model_preprocessing'] = "
            f"{CATALOG_PREPROCESSING_VERSION!r} in the scoring config."
        )


@action_spec(
    inputs=["model_file"],
    outputs=["output_file"],
    description="Convert a trusted legacy cWB pickle to a portable XGBoost model",
)
def import_cwb_model(work_dir, model_file, output_file, trusted_pickle=False, **kwargs):
    """cWB writes pickle even when its CLI requires a .json filename.

    Pickle can execute code. Set trusted_pickle=True only for a model whose
    provenance you trust, e.g. your own local cwb_xgboost training output.
    Output must use the standard JSON or UBJ serialization understood by
    native postproduction scoring actions.
    """
    if not trusted_pickle:
        raise ValueError("Legacy model import requires trusted_pickle=True")
    import pickle

    import xgboost as xgb

    source = Path(work_dir) / model_file
    output = Path(work_dir) / output_file
    if output.suffix not in (".json", ".ubj") or source.resolve() == output.resolve():
        raise ValueError("Use a distinct .json or .ubj output path")
    with source.open("rb") as stream:
        model = pickle.load(stream)
    if not isinstance(model, xgb.XGBClassifier):
        raise TypeError("Expected a cWB XGBClassifier")
    _record_catalog_preprocessing(model)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(output)
    summary = {
        "model_file": str(output),
        "source_file": str(source),
        "features": model.get_booster().feature_names,
        "catalog_preprocessing": CATALOG_PREPROCESSING_VERSION,
        "best_iteration": model.best_iteration,
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(summary, indent=2)
    )
    return summary
