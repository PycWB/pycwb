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
from pycwb.modules.postprocess.lag_filters import (
    nonzero_lag_mask,
    try_unshifted_job_ids_from_catalog,
    unshifted_job_ids_from_progress,
    zero_lag_mask,
)

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
    _, X, _, _ = _preprocess_for_scoring(df, nifo, search, config_file, work_dir, clf)
    return clf.predict_proba(X)[:, 1]


def _preprocess_for_scoring(
    df: pd.DataFrame,
    nifo: int,
    search: str,
    config_file: Optional[str],
    work_dir: str,
    clf: xgb.XGBClassifier,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, Optional[str]]:
    """Preprocess a raw catalog into model features for scoring."""
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
        config_path = _resolve(config_file)
        update_config_fn = import_function_from_file(config_path, "update_config")
        update_config_fn(xgb_params, ML_list, ML_caps, ML_balance, ML_options)
        update_ML_list(ML_list, ML_defcaps, ML_caps)
    else:
        config_path = None

    # Preprocess (creates derived features: rho0_40d0, Qa, Qp, ecor/likelihood, …)
    df = preprocess_events(df, nifo, ML_options, ML_caps)

    # Build feature matrix
    feature_names = clf.get_booster().feature_names
    X = pd.DataFrame(index=df.index)
    for f in feature_names:
        X[f] = df[f] if f in df.columns else 0.0

    return df, X, ML_options, config_path


def _score_catalog_dataframe(
    df: pd.DataFrame,
    nifo: int,
    search: str,
    config_file: Optional[str],
    work_dir: str,
    clf: xgb.XGBClassifier,
) -> pd.DataFrame:
    """Return a catalog copy with XGBoost and user-defined ranking columns."""
    from pycwb.modules.cwb_xgboost.read_data import apply_user_ranking_statistics

    _, X, ML_options, config_path = _preprocess_for_scoring(
        df, nifo, search, config_file, work_dir, clf,
    )
    probs = clf.predict_proba(X)[:, 1]

    scored = df.copy()
    scored["xgb_prob"] = probs
    scored["MLstat"] = probs
    scored = apply_user_ranking_statistics(scored, search, config_path, ML_options)
    return scored


def _resolve_path(work_dir: str, path: str) -> str:
    """Resolve a path relative to work_dir."""
    return path if os.path.isabs(path) else os.path.join(work_dir, path)


def _scoring_read_columns(
    work_dir: str,
    catalog_file: str,
    nifo: int,
    search: str,
    config_file: Optional[str],
    extra_columns: Optional[list[str]] = None,
) -> list[str]:
    """Return projected raw columns needed for XGB scoring plus requested output columns."""
    import copy
    from pycwb.modules.cwb_xgboost.config import xgb_config
    from pycwb.modules.cwb_xgboost.utils_extended import update_ML_list
    from pycwb.utils.module import import_function_from_file
    from .train_xgboost import _xgb_required_input_columns

    cat_path = _resolve_path(work_dir, catalog_file)
    xgb_params, ML_list, ML_caps, ML_balance, ML_options = xgb_config(search, nifo)
    if config_file:
        config_path = _resolve_path(work_dir, config_file)
        ML_defcaps = copy.deepcopy(ML_caps)
        update_config_fn = import_function_from_file(config_path, "update_config")
        update_config_fn(xgb_params, ML_list, ML_caps, ML_balance, ML_options)
        update_ML_list(ML_list, ML_defcaps, ML_caps)
    return _xgb_required_input_columns(
        [cat_path],
        nifo=nifo,
        ML_options=ML_options,
        ML_caps=ML_caps,
        ML_balance=ML_balance,
        ML_list=ML_list,
        extra_columns=extra_columns,
    )


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

    scored_df = _score_catalog_dataframe(df, nifo, search, config_file, work_dir, clf)
    probs = scored_df["xgb_prob"].to_numpy()
    n_recovered = int((probs >= threshold).sum())
    efficiency = n_recovered / max(n_total, 1)

    logger.info(
        "Efficiency: %d / %d = %.4f  (threshold=%.2f)",
        n_recovered, n_total, efficiency, threshold,
    )

    if output_file:
        out_path = _resolve_path(work_dir, output_file)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        scored_df.to_parquet(out_path, index=False)
        logger.info("Scored catalog → %s", out_path)

    return {
        "n_total": n_total,
        "n_recovered": n_recovered,
        "efficiency": float(efficiency),
        "threshold": threshold,
    }


# ---------------------------------------------------------------------------
# score_catalog
# ---------------------------------------------------------------------------

@action_spec(
    outputs=['output_file'],
    inputs=['catalog_file', 'model_file', 'config_file'],
    description='Score a catalog with XGBoost and user-defined ranking statistics',
)
def score_catalog(
    work_dir: str,
    catalog_file: str,
    model_file: str,
    search: str = "blf",
    nifo: int = 2,
    config_file: Optional[str] = None,
    output_file: Optional[str] = None,
    lag_selection: str = "all",
    batch_size: int = 200_000,
    **kwargs,
) -> dict:
    """Score a catalog in parquet batches.

    ``lag_selection`` can be ``"all"``, ``"zero_lag"``, or
    ``"nonzero_lag"``.  Batch processing is mainly useful for scoring the
    zero-lag subset of a large production catalog without materializing all
    background triggers at once.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    if output_file is None:
        raise ValueError("score_catalog requires output_file")
    if lag_selection not in {"all", "zero_lag", "nonzero_lag"}:
        raise ValueError("lag_selection must be one of: all, zero_lag, nonzero_lag")

    cat_path = _resolve_path(work_dir, catalog_file)
    model_path = _resolve_path(work_dir, model_file)
    out_path = _resolve_path(work_dir, output_file)

    clf = xgb.XGBClassifier()
    clf.load_model(model_path)
    unshifted_jobs = try_unshifted_job_ids_from_catalog(cat_path)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    parquet_file = pq.ParquetFile(cat_path)
    writer: pq.ParquetWriter | None = None
    n_input = 0
    n_scored = 0

    try:
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            df = batch.to_pandas()
            n_input += len(df)
            if lag_selection == "zero_lag":
                df = df[zero_lag_mask(df, unshifted_job_ids=unshifted_jobs)]
            elif lag_selection == "nonzero_lag":
                df = df[nonzero_lag_mask(df, unshifted_job_ids=unshifted_jobs)]
            if df.empty:
                continue

            scored = _score_catalog_dataframe(
                df.reset_index(drop=True), nifo, search, config_file, work_dir, clf,
            )
            table = pa.Table.from_pandas(scored, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(out_path, table.schema)
            writer.write_table(table)
            n_scored += len(scored)
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        pd.DataFrame().to_parquet(out_path, index=False)

    logger.info("Scored %d / %d catalog rows → %s", n_scored, n_input, out_path)
    return {
        "n_input": n_input,
        "n_scored": n_scored,
        "lag_selection": lag_selection,
        "output_file": out_path,
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
    exclude_zero_lag: bool = True,
    bin_size: Optional[float] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
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
        If provided, write FAR values to this JSON.  When *bin_size* is
        set the output is a binned lookup table (``"bins"``, ``"far"``,
        ``"cum_events"``, ``"livetime"``); otherwise it is a per-event list.
    bin_size : float, optional
        If set together with *output_file*, produce a binned FAR lookup
        table instead of the default per-event list.
    vmin : float, optional
        Override minimum bin edge for binned output.  Defaults to the
        minimum *ranking_par* value in the catalog.
    vmax : float, optional
        Override maximum bin edge for binned output.  Defaults to the
        maximum *ranking_par* value in the catalog.

    Returns
    -------
    dict
        ``far_rho`` — per-event list by default, or the compact binned
        lookup table when *bin_size* is set and ``return_per_event`` is false.
        The ``binned`` key is also included when binned output is produced.
    """
    cat_path = _resolve_path(work_dir, catalog_file)
    model_path = _resolve_path(work_dir, model_file)
    trigger_unshifted_jobs = try_unshifted_job_ids_from_catalog(cat_path)

    clf = xgb.XGBClassifier()
    clf.load_model(model_path)

    scored_output_columns = [
        "id", "job_id", "lag_idx", "trial_idx", "gps_time", "ifar",
        "rho", "net_cc", "likelihood", "coherent_energy",
        "coherent_energy_norm", "central_freq", "frequency",
        "central_freq_H1", "central_freq_L1", "frequency_H1", "frequency_L1",
    ]
    read_columns = _scoring_read_columns(
        work_dir,
        catalog_file,
        nifo,
        search,
        config_file,
        extra_columns=scored_output_columns + [ranking_par],
    )
    df = pd.read_parquet(cat_path, columns=read_columns)
    if exclude_zero_lag:
        df = df[nonzero_lag_mask(df, unshifted_job_ids=trigger_unshifted_jobs)].reset_index(drop=True)
    df = _score_catalog_dataframe(df, nifo, search, config_file, work_dir, clf)
    probs = df["xgb_prob"].to_numpy()

    if ranking_par not in df.columns:
        raise KeyError(f"Ranking parameter '{ranking_par}' not in catalog columns: {list(df.columns)}")
    rho_vals = pd.to_numeric(df[ranking_par], errors="coerce").to_numpy()

    # Sort by ranking parameter descending, compute cumulative FAR
    valid = np.isfinite(rho_vals)
    if not valid.any():
        raise ValueError(f"No finite values found for ranking parameter '{ranking_par}'")
    order = np.argsort(-rho_vals[valid])
    valid_indices = np.flatnonzero(valid)[order]
    n_total = len(order)
    return_per_event = bool(kwargs.get("return_per_event", bin_size is None))
    far_rho_data = []
    if return_per_event:
        for i, idx in enumerate(valid_indices):
            n_above = i + 1
            far = n_above / max(livetime, 1.0)
            far_rho_data.append({
                "rho": float(rho_vals[idx]),
                "ranking_par": ranking_par,
                "ranking_value": float(rho_vals[idx]),
                "far": float(far),
                "n_events": n_above,
                "xgb_prob": float(probs[idx]),
            })

    logger.info(
        "FAR computed: %d events, livetime=%.0f s (%.1f days)",
        n_total, livetime, livetime / 86400.0,
    )

    # ── Binned output (optional) ────────────────────────────────────────
    binned_data = None
    if bin_size is not None:
        import json as _json
        rho_finite = rho_vals[valid]
        _vmin = float(vmin) if vmin is not None else float(rho_finite.min())
        _vmax = float(vmax) if vmax is not None else float(rho_finite.max())
        bin_edges = np.arange(_vmin, _vmax + bin_size, bin_size)
        if len(bin_edges) < 2:
            bin_edges = np.array([_vmin, _vmin + bin_size])
        hist, _ = np.histogram(rho_finite, bins=bin_edges)
        cum_hist = np.cumsum(hist[::-1])[::-1]
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        livetime_years = livetime / 86400.0 / 365.25
        binned_data = {
            "bins": bin_centers.tolist(),
            "far": (cum_hist / max(livetime_years, 1e-10)).tolist(),
            "n_events": hist.tolist(),
            "cum_events": cum_hist.tolist(),
            "ranking_par": ranking_par,
            "livetime": livetime,
            "livetime_years": livetime_years,
        }
        if output_file:
            out_path = _resolve_path(work_dir, output_file)
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            with open(out_path, "w") as f:
                _json.dump(binned_data, f, indent=2)
            logger.info("Binned FAR data → %s (bins=%d, range=[%.4g, %.4g])",
                         out_path, len(bin_centers), _vmin, _vmax)
    elif output_file:
        import json
        out_path = _resolve_path(work_dir, output_file)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(far_rho_data, f, indent=2)
        logger.info("FAR data (per-event) → %s", out_path)

    # Also save scored catalog if requested
    scored_file = kwargs.get("scored_catalog")
    if scored_file:
        scored_path = _resolve_path(work_dir, scored_file)
        os.makedirs(os.path.dirname(scored_path) or ".", exist_ok=True)
        df.to_parquet(scored_path, index=False)
        logger.info("Scored BKG catalog → %s", scored_path)

    result = {"far_rho": far_rho_data}
    if binned_data is not None:
        if not return_per_event:
            result["far_rho"] = binned_data
        result["binned"] = binned_data
    return result


# ---------------------------------------------------------------------------
# MDC blind scoring
# ---------------------------------------------------------------------------

@action_spec(
    outputs=['output_csv'],
    inputs=['mdc_catalog', 'model_file', 'bkg_scored_catalog', 'progress_file', 'job_ids_file', 'livetime', 'config_file'],
    description='Score blind MDC catalog and output detections above IFAR threshold',
)
def score_mdc_catalog(
    work_dir: str,
    mdc_catalog: str,
    model_file: str,
    bkg_scored_catalog: str,
    progress_file: Optional[str] = None,
    job_ids_file: Optional[str] = None,
    livetime: Optional[float] = None,
    search: str = "blf",
    nifo: int = 2,
    config_file: Optional[str] = None,
    ranking_par: str = "xgb_prob",
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
    progress_file : str, optional
        Progress parquet for live-time computation when ``livetime`` is not
        provided.
    job_ids_file : str, optional
        Job IDs used for legacy BKG live-time computation when ``livetime`` is
        not provided.
    livetime : float, optional
        FAR live time in seconds. Preferred for interval-based BKG splits.
    search : str
        Search label.
    nifo : int
        Number of interferometers.
    config_file : str, optional
        Path to xgb_config override.
    ranking_par : str
        Scored catalog column used for FAR calibration and MDC thresholding.
        Defaults to ``"xgb_prob"`` for backward compatibility.
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
    prog_path = _resolve_path(work_dir, progress_file) if progress_file else None
    jobs_path = _resolve_path(work_dir, job_ids_file) if job_ids_file else None

    # ── IFAR presets ────────────────────────────────────────────────────
    _IFAR_PRESETS = {
        "10yr": 315576000, "1yr": 31557600, "6mo": 15778800,
        "1mo": 2592000, "1wk": 604800, "1day": 86400,
    }
    ifar_sec = _IFAR_PRESETS.get(ifar_threshold,
        float(ifar_threshold) if ifar_threshold.replace(".", "").isdigit() else 31557600)

    # ── Live time ────────────────────────────────────────────────────────
    if livetime is None:
        if prog_path is None or jobs_path is None:
            raise ValueError("score_mdc_catalog requires either livetime or both progress_file and job_ids_file")
        prog = pd.read_parquet(prog_path)
        try:
            progress_unshifted_jobs = unshifted_job_ids_from_progress(prog_path)
        except (FileNotFoundError, ValueError, KeyError):
            progress_unshifted_jobs = None
        with open(jobs_path) as f:
            job_ids = {int(l.strip()) for l in f if l.strip()}
        prog = prog[prog["job_id"].isin(job_ids)]
        if "status" in prog.columns:
            prog = prog[prog["status"] == "completed"]
        prog_nz = prog[nonzero_lag_mask(prog, unshifted_job_ids=progress_unshifted_jobs)]
        livetime = float(prog_nz["livetime"].sum())
    else:
        livetime = float(livetime)
    logger.info("Livetime: %.0f s = %.2f yr", livetime, livetime / 31557600)

    # ── Load model ──────────────────────────────────────────────────────
    clf = xgb.XGBClassifier()
    clf.load_model(model_path)

    # ── Load MDC catalog (zero-lag only) ────────────────────────────────
    mdc_df = pd.read_parquet(mdc_path)
    mdc_unshifted_jobs = try_unshifted_job_ids_from_catalog(mdc_path)
    mdc_zl = mdc_df[zero_lag_mask(mdc_df, unshifted_job_ids=mdc_unshifted_jobs)].copy()
    logger.info("MDC: %d total, %d zero-lag", len(mdc_df), len(mdc_zl))

    # ── Score with preprocessing ────────────────────────────────────────
    mdc_zl = _score_catalog_dataframe(mdc_zl, nifo, search, config_file, work_dir, clf)
    if ranking_par not in mdc_zl.columns:
        raise KeyError(f"Ranking parameter '{ranking_par}' not in MDC catalog columns: {list(mdc_zl.columns)}")
    ranking_values = pd.to_numeric(mdc_zl[ranking_par], errors="coerce").to_numpy()

    # ── BKG calibration ─────────────────────────────────────────────────
    bkg_s = pd.read_parquet(bkg_path)
    bkg_s = bkg_s[nonzero_lag_mask(bkg_s)].reset_index(drop=True)
    if ranking_par not in bkg_s.columns:
        raise KeyError(f"Ranking parameter '{ranking_par}' not in BKG catalog columns: {list(bkg_s.columns)}")
    bkg_ranking = pd.to_numeric(bkg_s[ranking_par], errors="coerce").to_numpy()
    bkg_ranking = bkg_ranking[np.isfinite(bkg_ranking)]
    if len(bkg_ranking) == 0:
        raise ValueError(f"No finite values found for ranking parameter '{ranking_par}' in BKG catalog")
    bkg_ranking = np.sort(bkg_ranking)[::-1]
    far_values = np.arange(1, len(bkg_ranking) + 1) / max(livetime, 1.0)
    target_far = 1.0 / ifar_sec
    n_allowed = int(np.searchsorted(far_values, target_far, side="right"))
    if n_allowed <= 0:
        ranking_threshold = np.nextafter(bkg_ranking[0], np.inf)
    elif n_allowed >= len(bkg_ranking):
        ranking_threshold = bkg_ranking[-1]
    else:
        ranking_threshold = bkg_ranking[n_allowed - 1]
    logger.info(
        "IFAR=%s (%.0f s): background step %s >= %.6f  (BKG allowed: %d / %d)",
        ifar_threshold, ifar_sec, ranking_par, ranking_threshold, n_allowed, len(bkg_ranking),
    )

    # ── Compute per-MDC IFAR and filter detections ──────────────────────
    # Use per-event IFAR for inclusion.  A single ranking threshold can drop
    # events that lie in a gap between adjacent background ranks even though
    # their calibrated IFAR is still above the requested threshold.
    n_above = np.searchsorted(-bkg_ranking, -ranking_values, side="right")
    n_above = np.where(np.isfinite(ranking_values), n_above, len(bkg_ranking))
    far_per_sec = n_above / max(livetime, 1.0)
    ifars_sec = np.full(len(n_above), float("inf"), dtype=float)
    nonzero_far = n_above > 0
    ifars_sec[nonzero_far] = livetime / n_above[nonzero_far]
    ifars_yr = np.where(np.isfinite(ifars_sec), ifars_sec / 31557600, float("inf"))

    mdc_zl["bkg_events_ge_ranking"] = n_above.astype(int)
    mdc_zl["far"] = far_per_sec
    mdc_zl["ifar_sec"] = ifars_sec
    mdc_zl["ifar_yr"] = ifars_yr
    mdc_zl["ifar_years"] = ifars_yr
    mdc_zl["ifar"] = ifars_yr

    detections = mdc_zl[ifars_sec >= ifar_sec].sort_values(
        ranking_par, ascending=False,
    )

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
        "ra", "dec", "sky_size", ranking_par, "xgb_prob",
        "far", "ifar", "ifar_sec", "ifar_yr", "ifar_years",
    ]
    out_cols = list(dict.fromkeys(c for c in out_cols if c in detections.columns))
    out_df = detections[out_cols]

    if output_csv:
        csv_path = _resolve_path(work_dir, output_csv)
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        out_df.to_csv(csv_path, index=False)
        logger.info("MDC detections → %s", csv_path)

    return {
        "n_total": len(mdc_zl),
        "n_detections": len(detections),
        "ranking_par": ranking_par,
        "ranking_threshold": float(ranking_threshold),
        "prob_threshold": float(ranking_threshold) if ranking_par == "xgb_prob" else None,
        "ifar_sec": ifar_sec,
        "livetime": livetime,
        "output_csv": _resolve_path(work_dir, output_csv) if output_csv else None,
    }
