"""
Real-time significance assignment for online triggers.

Loads a pre-trained XGBoost model and an IFAR lookup table at startup,
then maps each trigger's event features to a ranking statistic and
inverse false-alarm rate (IFAR).
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def load_significance_model(model_path: str, ifar_file: str):
    """Load the pre-trained model and IFAR table.

    Parameters
    ----------
    model_path : str
        Path to a saved XGBoost model (``.json`` or ``.ubj``).
        If empty, returns ``(None, None)`` and significance assignment
        becomes a no-op.
    ifar_file : str
        Path to the IFAR lookup table (NumPy ``.npz`` with keys
        ``stat`` and ``ifar``, or CSV with those columns).

    Returns
    -------
    tuple
        ``(model, ifar_table)`` — *model* is an ``xgboost.Booster``
        (or *None*), *ifar_table* is a dict with ``stat`` and ``ifar``
        arrays (or *None*).
    """
    model = None
    ifar_table = None

    if model_path and Path(model_path).exists():
        try:
            import xgboost as xgb
            model = xgb.Booster()
            model.load_model(model_path)
            logger.info("Loaded XGBoost model from %s", model_path)
        except Exception:
            logger.exception("Failed to load XGBoost model from %s", model_path)

    if ifar_file and Path(ifar_file).exists():
        try:
            path = Path(ifar_file)
            if path.suffix == ".npz":
                loaded = np.load(ifar_file)
                ifar_table = {"stat": loaded["stat"], "ifar": loaded["ifar"]}
            else:
                # Assume CSV with header: stat,ifar
                arr = np.loadtxt(ifar_file, delimiter=",", skiprows=1)
                ifar_table = {"stat": arr[:, 0], "ifar": arr[:, 1]}
            logger.info("Loaded IFAR table from %s (%d entries)",
                        ifar_file, len(ifar_table["stat"]))
        except Exception:
            logger.exception("Failed to load IFAR table from %s", ifar_file)

    return model, ifar_table


def assign_significance(event, model, ifar_table, feature_columns=None):
    """Assign ranking statistic and IFAR to an event in-place.

    Parameters
    ----------
    event : Event
        PyCWB event object.  ``ranking_statistic`` and ``ifar`` attributes
        are set on the object.
    model : xgboost.Booster or None
        Pre-trained classifier.  If *None*, significance is not assigned.
    ifar_table : dict or None
        Mapping from ranking statistic to IFAR.
    feature_columns : list[str] or None
        Ordered attribute names to extract from *event* as model input.
        Defaults to ``['rho', 'netcc', 'penalty', 'ecor', 'qveto', 'qfactor']``.
    """
    if model is None:
        return

    if feature_columns is None:
        feature_columns = ["rho", "netcc", "penalty", "ecor", "qveto", "qfactor"]

    features = np.array(
        [float(getattr(event, f, 0.0)) for f in feature_columns],
        dtype=np.float64,
    )

    try:
        import xgboost as xgb
        dmat = xgb.DMatrix(features.reshape(1, -1), feature_names=feature_columns)
        ranking_stat = float(model.predict(dmat)[0])
    except Exception:
        logger.exception("XGBoost prediction failed")
        ranking_stat = 0.0

    event.ranking_statistic = ranking_stat

    if ifar_table is not None:
        event.ifar = float(np.interp(
            ranking_stat,
            ifar_table["stat"],
            ifar_table["ifar"],
        ))
    else:
        event.ifar = 0.0
