"""pycWB XGBoost module.

Public API
----------
.. autosummary::

    xgb_predict

Training pipelines
------------------
.. autosummary::

    train
    train_classifier
    make_train_test_split

Data helpers
------------
.. autosummary::

    load_flat_parquet
    build_features_from_parquet
    read_catalog_to_dataframe
    preprocess_events
    apply_training_cuts

Evaluation
----------
.. autosummary::

    evaluate_classifier
"""

from .cwb_xgboost import xgb_predict
from .training import train, train_classifier, make_train_test_split
from .read_data import (
    load_flat_parquet,
    build_features_from_parquet,
    read_catalog_to_dataframe,
    preprocess_events,
    apply_training_cuts,
)
from .plots import evaluate_classifier
from .config import xgb_config
from .utils import load_model, getcapname
from .utils_extended import get_balanced_tail, get_balanced_bulk, update_ML_list

__all__ = [
    # Main entry points
    "xgb_predict",
    # Training pipelines
    "train",
    "train_classifier",
    "make_train_test_split",
    # Flat-Parquet data helpers
    "load_flat_parquet",
    "build_features_from_parquet",
    # Catalog data helpers
    "read_catalog_to_dataframe",
    "preprocess_events",
    "apply_training_cuts",
    # Evaluation
    "evaluate_classifier",
    # Config
    "xgb_config",
    # Utilities
    "load_model",
    "getcapname",
    "get_balanced_tail",
    "get_balanced_bulk",
    "update_ML_list",
]
