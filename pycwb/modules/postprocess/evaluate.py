"""Model evaluation — workflow-compatible steps for efficiency & FAR.

Applies a trained XGBoost model to a catalog and computes:
- **Efficiency** (fraction of injections recovered above a threshold)
- **FAR** (false-alarm rate vs. ranking statistic) using live time from the
  progress file.

All scoring uses ``preprocess_events`` to produce the same derived features
(``ecor/likelihood``, ``rho0_40d0``, ``Qa``, ``Qp``, …) that the model was
trained on.

Workflow actions
----------------
``postprocess.evaluate.evaluate_efficiency``
    Load model, score a SIM catalog, compute efficiency vs. threshold.

``postprocess.evaluate.evaluate_far_rho``
    Load model, score a BKG catalog, compute FAR vs. ranking statistic
    using live time from the progress file.

``postprocess.evaluate.score_mdc_catalog``
    Score a blind MDC catalog and output triggers above an IFAR threshold.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb

from pycwb.post_production.action_spec import action_spec

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared preprocessing + scoring helper
# ---------------------------------------------------------------------------

def _preprocess_and_score(
    df: pd.DataFrame,
    nifo: int,
    search: str,
    config_file: Optional[str],
    work_dir: str,
    clf: xgb.XGBClassifier,
) -> np.ndarray:
    """Map columns, run preprocess_events, and score with a loaded model.

    Parameters
    ----------
    df : pd.DataFrame
        Raw catalog DataFrame (unmapped).
    nifo : int
        Number of interferometers.
    search : str
        Search label (``"blf"``, ``"bhf"``, …).
    config_file : str or None
        Path to xgb_config override file (relative to work_dir).
    work_dir : str
        Base directory for resolving config_file.
    clf : xgb.XGBClassifier
        Already-loaded model.

    Returns
    -------
    probs : np.ndarray
        Probability array for the positive class.
    """
    import copy
    from pycwb.modules.cwb_xgboost.config import xgb_config
    from pycwb.modules.cwb_xgboost.read_data import preprocess_events
    from pycwb.modules.cwb_xgboost.utils_extended import update_ML_list
    from pycwb.utils.module import import_function_from_file
    from .train_xgboost import _map_catalog_columns

    def _resolve(p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(work_dir, p)

    # Map column names
    df = _map_catalog_columns(df.copy(), nifo)

    # Load config (same as training)
    xgb_params, ML_list, ML_caps, ML_balance, ML_options = xgb_config(search, nifo)
    if config_file:
        ML_defcaps = copy.deepcopy(ML_caps)
        update_config_fn = import_function_from_file(_resolve(config_file), "update_config")
        update_config_fn(xgb_params, ML_list, ML_caps, ML_balance, ML_options)
        update_ML_list(ML_list, ML_defcaps, ML_caps)

    # Preprocess (creates derived features: rho0_40d0, Qa, Qp, ecor/likelihood, …)
    df = preprocess_events(df, nifo, ML_options, ML_caps)

    # Build feature matrix
    feature_names = clf.get_booster().feature_names
    X = pd.DataFrame(index=df.index)
    for f in feature_names:
        X[f] = df[f] if f in df.columns else 0.0

    return clf.predict_proba(X)[:, 1]


def _resolve_path(work_dir: str, path: str) -> str:
    """Resolve a path relative to work_dir."""
    return path if os.path.isabs(path) else os.path.join(work_dir, path)


# ---------------------------------------------------------------------------
# evaluate_efficiency
# ---------------------------------------------------------------------------

@action_spec(
    outputs=['output_file'],
    inputs=['catalog_file', 'model_file', 'config_file'],
    description='Score SIM catalog and compute efficiency vs threshold',
)
def evaluate_efficiency(
    work_dir: str,
    catalog_file: str,
    model_file: str,
    search: str = "blf",
    nifo: int = 2,
    config_file: Optional[str] = None,
    output_file: Optional[str] = None,
    threshold: float = 0.5,
    **kwargs,
) -> dict:
    """Score a SIM catalog with a trained model and compute efficiency.

    Parameters
    ----------
    work_dir : str
        Base directory.
    catalog_file : str
        Path to the SIM catalog parquet.
    model_file : str
        Path to the trained ``.ubj`` model.
    search : str
        Search label (``"blf"``, …).
    nifo : int
        Number of interferometers.
    config_file : str, optional
        Path to xgb_config override (e.g. ``xgb_config.py``), relative to work_dir.
    output_file : str, optional
        If provided, write scored probabilities to this parquet.
    threshold : float
        Classification threshold for efficiency calculation.

    Returns
    -------
    dict
        ``n_total``, ``n_recovered``, ``efficiency``, ``threshold``.
    """
    cat_path = _resolve_path(work_dir, catalog_file)
    model_path = _resolve_path(work_dir, model_file)

    clf = xgb.XGBClassifier()
    clf.load_model(model_path)
    logger.info("Loaded model with %d features", len(clf.get_booster().feature_names))

    df = pd.read_parquet(cat_path)
    n_total = len(df)

    probs = _preprocess_and_score(df, nifo, search, config_file, work_dir, clf)
    n_recovered = int((probs >= threshold).sum())
    efficiency = n_recovered / max(n_total, 1)

    logger.info(
        "Efficiency: %d / %d = %.4f  (threshold=%.2f)",
        n_recovered, n_total, efficiency, threshold,
    )

    if output_file:
        out_path = _resolve_path(work_dir, output_file)
        df_out = df.copy()
        df_out["xgb_prob"] = probs
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        df_out.to_parquet(out_path, index=False)
        logger.info("Scored catalog → %s", out_path)

    return {
        "n_total": n_total,
        "n_recovered": n_recovered,
        "efficiency": float(efficiency),
        "threshold": threshold,
    }


# ---------------------------------------------------------------------------
# evaluate_far_rho
# ---------------------------------------------------------------------------

@action_spec(
    outputs=['output_file', 'scored_catalog'],
    inputs=['catalog_file', 'model_file', 'config_file'],
    description='Score BKG catalog and compute FAR vs ranking statistic',
)
def evaluate_far_rho(
    work_dir: str,
    catalog_file: str,
    model_file: str,
    livetime: float,
    search: str = "blf",
    nifo: int = 2,
    config_file: Optional[str] = None,
    ranking_par: str = "rho",
    output_file: Optional[str] = None,
    **kwargs,
) -> dict:
    """Score a BKG catalog and compute FAR vs. ranking parameter.

    Parameters
    ----------
    work_dir : str
        Base directory.
    catalog_file : str
        Path to the BKG catalog parquet.
    model_file : str
        Path to the trained model.
    livetime : float
        Total live time in seconds for the selected BKG jobs.
    search : str
        Search label.
    nifo : int
        Number of interferometers.
    config_file : str, optional
        Path to xgb_config override.
    ranking_par : str
        Column to use as ranking statistic (default ``"rho"``).
    output_file : str, optional
        If provided, write FAR values to this JSON.

    Returns
    -------
    dict
        ``far_rho`` list of dicts with ``rho``, ``far``, ``n_events``.
    """
    cat_path = _resolve_path(work_dir, catalog_file)
    model_path = _resolve_path(work_dir, model_file)

    clf = xgb.XGBClassifier()
    clf.load_model(model_path)

    df = pd.read_parquet(cat_path)
    probs = _preprocess_and_score(df, nifo, search, config_file, work_dir, clf)

    if ranking_par not in df.columns:
        raise KeyError(f"Ranking parameter '{ranking_par}' not in catalog columns: {list(df.columns)}")
    rho_vals = df[ranking_par].values

    # Sort by ranking parameter descending, compute cumulative FAR
    order = np.argsort(-rho_vals)
    n_total = len(order)
    far_rho_data = []
    for i, idx in enumerate(order):
        n_above = i + 1
        far = n_above / max(livetime, 1.0)
        far_rho_data.append({
            "rho": float(rho_vals[idx]),
            "far": float(far),
            "n_events": n_above,
            "xgb_prob": float(probs[idx]),
        })

    logger.info(
        "FAR computed: %d events, livetime=%.0f s (%.1f days)",
        n_total, livetime, livetime / 86400.0,
    )

    if output_file:
        import json
        out_path = _resolve_path(work_dir, output_file)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(far_rho_data, f, indent=2)
        logger.info("FAR data → %s", out_path)

    # Also save scored catalog if requested
    scored_file = kwargs.get("scored_catalog")
    if scored_file:
        scored_path = _resolve_path(work_dir, scored_file)
        df_out = df.copy()
        df_out["xgb_prob"] = probs
        os.makedirs(os.path.dirname(scored_path) or ".", exist_ok=True)
        df_out.to_parquet(scored_path, index=False)
        logger.info("Scored BKG catalog → %s", scored_path)

    return {"far_rho": far_rho_data}


# ---------------------------------------------------------------------------
# MDC blind scoring
# ---------------------------------------------------------------------------

@action_spec(
    outputs=['output_csv'],
    inputs=['mdc_catalog', 'model_file', 'bkg_scored_catalog', 'progress_file', 'job_ids_file', 'config_file'],
    description='Score blind MDC catalog and output detections above IFAR threshold',
)
def score_mdc_catalog(
    work_dir: str,
    mdc_catalog: str,
    model_file: str,
    bkg_scored_catalog: str,
    progress_file: str,
    job_ids_file: str,
    search: str = "blf",
    nifo: int = 2,
    config_file: Optional[str] = None,
    ifar_threshold: str = "1yr",
    output_csv: Optional[str] = None,
    **kwargs,
) -> dict:
    """Score a blind MDC catalog and output triggers above an IFAR threshold.

    Parameters
    ----------
    mdc_catalog : str
        Path to the MDC parquet.
    model_file : str
        Trained XGBoost model.
    bkg_scored_catalog : str
        Scored BKG catalog (with ``xgb_prob``) for IFAR calibration.
    progress_file : str
        Progress parquet for live-time computation.
    job_ids_file : str
        Job IDs used for the BKG live time.
    search : str
        Search label.
    nifo : int
        Number of interferometers.
    config_file : str, optional
        Path to xgb_config override.
    ifar_threshold : str
        IFAR preset (``"1yr"``, ``"1mo"``, ``"1day"``, …) or seconds.
    output_csv : str, optional
        Save CSV to this path.

    Returns
    -------
    dict
        ``n_total``, ``n_detections``, ``prob_threshold``, ``ifar_sec``,
        ``livetime``, ``output_csv``.
    """
    import json

    mdc_path = _resolve_path(work_dir, mdc_catalog)
    model_path = _resolve_path(work_dir, model_file)
    bkg_path = _resolve_path(work_dir, bkg_scored_catalog)
    prog_path = _resolve_path(work_dir, progress_file)
    jobs_path = _resolve_path(work_dir, job_ids_file)

    # ── IFAR presets ────────────────────────────────────────────────────
    _IFAR_PRESETS = {
        "10yr": 315576000, "1yr": 31557600, "6mo": 15778800,
        "1mo": 2592000, "1wk": 604800, "1day": 86400,
    }
    ifar_sec = _IFAR_PRESETS.get(ifar_threshold,
        float(ifar_threshold) if ifar_threshold.replace(".", "").isdigit() else 31557600)

    # ── Live time ────────────────────────────────────────────────────────
    prog = pd.read_parquet(prog_path)
    with open(jobs_path) as f:
        job_ids = {int(l.strip()) for l in f if l.strip()}
    prog = prog[prog["job_id"].isin(job_ids)]
    prog_nz = prog[prog["lag_idx"] != 0]
    livetime = float(prog_nz["livetime"].sum())
    logger.info("Livetime: %.0f s = %.2f yr", livetime, livetime / 31557600)

    # ── Load model ──────────────────────────────────────────────────────
    clf = xgb.XGBClassifier()
    clf.load_model(model_path)

    # ── Load MDC catalog (zero-lag only) ────────────────────────────────
    mdc_df = pd.read_parquet(mdc_path)
    mdc_zl = mdc_df[mdc_df["lag_idx"] == 0].copy()
    logger.info("MDC: %d total, %d zero-lag", len(mdc_df), len(mdc_zl))

    # ── Score with preprocessing ────────────────────────────────────────
    probs = _preprocess_and_score(mdc_zl, nifo, search, config_file, work_dir, clf)
    mdc_zl["xgb_prob"] = probs

    # ── BKG calibration ─────────────────────────────────────────────────
    bkg_s = pd.read_parquet(bkg_path)
    bkg_probs = np.sort(bkg_s["xgb_prob"].values)[::-1]
    far_values = np.arange(1, len(bkg_probs) + 1) / max(livetime, 1.0)
    target_far = 1.0 / ifar_sec
    idx = np.searchsorted(far_values, target_far)
    prob_threshold = bkg_probs[idx] if idx < len(bkg_probs) else 1.0
    logger.info(
        "IFAR=%s (%.0f s): prob >= %.6f  (BKG above: %d / %d)",
        ifar_threshold, ifar_sec, prob_threshold, idx, len(bkg_probs),
    )

    # ── Filter detections ───────────────────────────────────────────────
    detections = mdc_zl[probs >= prob_threshold].sort_values(
        "xgb_prob", ascending=False,
    )

    # ── Compute IFAR for each detection ─────────────────────────────────
    ifars_yr = []
    for p in detections["xgb_prob"]:
        n_above = int((bkg_probs >= p).sum())
        far = n_above / max(livetime, 1.0)
        ifars_yr.append(1.0 / far / 31557600 if far > 0 else float("inf"))
    detections["ifar_yr"] = ifars_yr

    # ── Output columns ──────────────────────────────────────────────────
    # Preserve original per-IFO column values before _map_catalog_columns renames them
    _orig_cols = [
        "coherent_energy", "q_veto", "q_factor",
        "central_freq_L1", "central_freq_H1",
        "duration_L1", "duration_H1", "bandwidth_L1", "bandwidth_H1",
        "hrss_L1", "hrss_H1",
    ]
    for c in _orig_cols:
        if c in mdc_df.columns and c not in detections.columns:
            detections[c] = mdc_df.loc[detections.index, c]

    out_cols = [
        "id", "gps_time", "rho", "net_cc", "likelihood", "coherent_energy",
        "penalty", "q_veto", "q_factor",
        "central_freq_L1", "central_freq_H1",
        "duration_L1", "duration_H1", "bandwidth_L1", "bandwidth_H1",
        "ra", "dec", "sky_size", "ifar", "xgb_prob", "ifar_yr",
    ]
    out_cols = [c for c in out_cols if c in detections.columns]
    out_df = detections[out_cols]

    if output_csv:
        csv_path = _resolve_path(work_dir, output_csv)
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        out_df.to_csv(csv_path, index=False)
        logger.info("MDC detections → %s", csv_path)

    return {
        "n_total": len(mdc_zl),
        "n_detections": len(detections),
        "prob_threshold": float(prob_threshold),
        "ifar_sec": ifar_sec,
        "livetime": livetime,
        "output_csv": _resolve_path(work_dir, output_csv) if output_csv else None,
    }
