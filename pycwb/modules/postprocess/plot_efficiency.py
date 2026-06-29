"""Efficiency vs hrss plots — workflow-compatible.

Computes efficiency curves (fraction of injections recovered vs. hrss) at a
given IFAR threshold.  The IFAR threshold is determined by ranking background
events by XGBoost probability and computing the false-alarm rate.

Workflow actions
----------------
``postprocess.plot_efficiency.plot_efficiency_vs_hrss``
    Plot efficiency vs hrss at a fixed IFAR threshold and save the figure.
    Also computes the hrss at 50% efficiency.

``postprocess.plot_efficiency.compute_hrss50``
    Compute the hrss value at 50% recovery for a given IFAR.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd

from pycwb.post_production.action_spec import action_spec
from pycwb.modules.postprocess.efficiency_metrics import (
    _IFAR_PRESETS,
    _compute_efficiency_by_waveform_matched,
    _compute_efficiency_vs_hrss_by_waveform_matched,
    _fit_efficiency_curve,
    _interpolate_hrss50,
    _interpolate_hrss50_curve,
    _parse_waveform_q_frequency,
)
from pycwb.modules.postprocess.efficiency_plots import (
    _plot_efficiency_by_waveform_panels,
    _plot_efficiency_curve,
    _plot_waveform_efficiency,
)

logger = logging.getLogger(__name__)


@action_spec(
    outputs=['output_file'],
    inputs=['sim_catalog', 'bkg_catalog', 'model_file', 'config_file'],
    description='Compute hrss at 50% efficiency for a given IFAR threshold',
)
def compute_hrss50(
    work_dir: str,
    sim_catalog: str,
    bkg_catalog: str,
    livetime: float,
    ifar: str = "1yr",
    output_file: Optional[str] = None,
    **kwargs,
) -> dict:
    """Compute hrss at 50% efficiency for a given IFAR threshold.

    Parameters
    ----------
    work_dir : str
        Base directory.
    sim_catalog : str
        Path to scored SIM parquet (with ``xgb_prob`` column).
    bkg_catalog : str
        Path to BKG parquet (used to compute prob threshold from FAR).
    livetime : float
        Background live time in seconds.
    ifar : str
        IFAR threshold: ``"1yr"``, ``"6mo"``, ``"1mo"``, ``"1wk"``, ``"1day"``.
    output_file : str, optional
        Save plot to this path.
    matched_right_file : str, optional (via kwargs)
        Path to ``matched_right.parquet``.  If provided, efficiency is
        computed by counting **unique simulations** (``sim_sim_idx``) rather
        than individual triggers.  One injection → one trial.
    exclude_vetoed : bool, optional (via kwargs)
        If True and *matched_right_file* provided, exclude vetoed simulations
        (cat0|cat1|cat2|across_segments) from the denominator.

    Returns
    -------
    dict
        ``hrss50``, ``ifar_sec``, ``prob_threshold``, ``efficiency_curve``.
    """
    from .train_xgboost import _map_catalog_columns

    def _resolve(p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(work_dir, p)

    ifar_sec = _IFAR_PRESETS.get(ifar, float(ifar) if ifar.replace(".", "").isdigit() else 31557600)

    # ── Compute prob threshold from BKG ──────────────────────────────────
    bkg_df = pd.read_parquet(_resolve(bkg_catalog))
    if "xgb_prob" not in bkg_df.columns:
        raise KeyError("BKG catalog must have 'xgb_prob' column (run evaluate_far_rho first)")
    bkg_probs = np.sort(bkg_df["xgb_prob"].values)[::-1]
    n_bkg = len(bkg_probs)
    far_values = np.arange(1, n_bkg + 1) / max(livetime, 1.0)
    target_far = 1.0 / ifar_sec
    idx = np.searchsorted(far_values, target_far)
    if idx >= n_bkg:
        prob_threshold = bkg_probs[-1] + 0.01
        logger.warning(
            "Cannot reach IFAR=%s (%.0f s) with available data. Using strictest cut.",
            ifar, ifar_sec,
        )
    else:
        prob_threshold = bkg_probs[idx]
    logger.info(
        "IFAR=%s (%.0f s): prob >= %.6f  (BKG above: %d / %d)",
        ifar, ifar_sec, prob_threshold, idx, n_bkg,
    )

    # ── Check if we should use matched_right for unique-sim efficiency ────
    matched_right_file = kwargs.get("matched_right_file")
    exclude_vetoed = kwargs.get("exclude_vetoed", False)

    if matched_right_file:
        # ── Unique-simulation-based efficiency ──────────────────────────
        mr_path = _resolve(matched_right_file)
        mr_df = pd.read_parquet(mr_path)
        logger.info(
            "matched_right: %d rows, %d unique sim_sim_idx, %d recovered (id not null)",
            len(mr_df), mr_df["sim_sim_idx"].nunique(), mr_df["id"].notna().sum(),
        )

        # Score recovered sims with XGBoost
        import xgboost as xgb
        from .evaluate import _preprocess_and_score
        model_file = kwargs.get("model_file")
        if not model_file:
            # If no model_file provided, use prob_threshold on existing scored data
            # Fall back to trigger-based method
            logger.warning("matched_right_file requires model_file; falling back to trigger-based")
        else:
            search = kwargs.get("search", "blf")
            nifo = kwargs.get("nifo", 2)
            config_file = kwargs.get("config_file")
            clf = xgb.XGBClassifier()
            clf.load_model(_resolve(model_file))
            # Score only recovered sims (those with trigger data)
            recovered = mr_df[mr_df["id"].notna()].copy()
            if len(recovered) > 0:
                recovered["xgb_prob"] = _preprocess_and_score(
                    recovered, nifo, search, config_file, work_dir, clf,
                )
                # Map probs back to all rows
                mr_df["xgb_prob"] = 0.0
                mr_df.loc[recovered.index, "xgb_prob"] = recovered["xgb_prob"]
            else:
                mr_df["xgb_prob"] = 0.0

            # Veto filter for denominator (optional)
            if exclude_vetoed:
                veto_mask = (
                    mr_df["sim_vetoed_cat0"].fillna(False).astype(bool)
                    | mr_df["sim_vetoed_cat1"].fillna(False).astype(bool)
                    | mr_df["sim_vetoed_cat2"].fillna(False).astype(bool)
                    | mr_df["sim_across_segments"].fillna(False).astype(bool)
                )
                mr_clean = mr_df[~veto_mask]
                logger.info(
                    "Vetoed sims excluded: %d → %d denominator",
                    len(mr_df), len(mr_clean),
                )
            else:
                mr_clean = mr_df

            # Build efficiency curve: group by sim_hrss
            mr_clean = mr_clean.copy()
            mr_clean["sim_hrss_float"] = pd.to_numeric(mr_clean["sim_hrss"], errors="coerce")
            mr_valid = mr_clean.dropna(subset=["sim_hrss_float"])
            hrss_vals = sorted(mr_valid["sim_hrss_float"].unique())

            eff_curve = []
            for h in hrss_vals:
                mask = mr_valid["sim_hrss_float"] == h
                n_total = mask.sum()  # unique sims at this hrss
                n_rec = int((mr_valid.loc[mask, "xgb_prob"] >= prob_threshold).sum())
                eff = n_rec / max(n_total, 1)
                eff_curve.append({
                    "hrss": float(h), "efficiency": float(eff),
                    "n_total": int(n_total), "n_recovered": int(n_rec),
                })

            hrss50 = _interpolate_hrss50(eff_curve)
            logger.info("hrss50 (unique-sim) = %.2e", hrss50)

            if output_file:
                out_path = _resolve(output_file)
                os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
                _plot_efficiency_curve(eff_curve, hrss50, ifar, ifar_sec, prob_threshold, out_path)

            return {
                "hrss50": float(hrss50) if hrss50 is not None else None,
                "ifar_sec": ifar_sec,
                "prob_threshold": float(prob_threshold),
                "efficiency_curve": eff_curve,
                "method": "unique_simulation",
                "exclude_vetoed": exclude_vetoed,
            }

    # ── Fallback: trigger-based efficiency (original behaviour) ─────────
    sim_path = _resolve(sim_catalog)
    sim_df = pd.read_parquet(sim_path)
    if "xgb_prob" not in sim_df.columns:
        raise KeyError("SIM catalog must have 'xgb_prob' column (run evaluate_efficiency first)")

    sim_df["hrss_inj"] = sim_df["injection"].apply(
        lambda x: float(x.get("hrss", np.nan)) if isinstance(x, dict) else np.nan
    )
    sim_df = sim_df.dropna(subset=["hrss_inj"])
    hrss_vals = sorted(sim_df["hrss_inj"].unique())
    logger.info("SIM: %d events, %d unique hrss values", len(sim_df), len(hrss_vals))

    eff_curve = []
    for h in hrss_vals:
        mask = sim_df["hrss_inj"] == h
        n_total = mask.sum()
        n_rec = int((sim_df.loc[mask, "xgb_prob"] >= prob_threshold).sum())
        eff = n_rec / max(n_total, 1)
        eff_curve.append({"hrss": float(h), "efficiency": float(eff), "n_total": int(n_total), "n_recovered": int(n_rec)})

    hrss50 = _interpolate_hrss50(eff_curve)
    logger.info("hrss50 (trigger-based) = %.2e", hrss50)

    if output_file:
        out_path = _resolve(output_file)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        _plot_efficiency_curve(eff_curve, hrss50, ifar, ifar_sec, prob_threshold, out_path)

    return {
        "hrss50": float(hrss50) if hrss50 is not None else None,
        "ifar_sec": ifar_sec,
        "prob_threshold": float(prob_threshold),
        "efficiency_curve": eff_curve,
        "method": "trigger_based",
    }


# ---------------------------------------------------------------------------
# plot_efficiency_vs_hrss (alias with plot output)
# ---------------------------------------------------------------------------

@action_spec(
    outputs=['output_file'],
    inputs=['sim_catalog', 'bkg_catalog', 'model_file', 'config_file'],
    description='Plot efficiency vs hrss at a given IFAR threshold',
)
def plot_efficiency_vs_hrss(
    work_dir: str,
    sim_catalog: str,
    bkg_catalog: str,
    livetime: float,
    ifar: str = "1yr",
    output_file: str = "efficiency_vs_hrss.png",
    **kwargs,
) -> dict:
    """Plot efficiency vs hrss at a given IFAR threshold.

    See :func:`compute_hrss50` for parameter details.
    """
    return compute_hrss50(
        work_dir=work_dir,
        sim_catalog=sim_catalog,
        bkg_catalog=bkg_catalog,
        livetime=livetime,
        ifar=ifar,
        output_file=output_file,
    )


# ---------------------------------------------------------------------------
# Waveform-level efficiency using matched_right.parquet
# ---------------------------------------------------------------------------

@action_spec(
    outputs=['output_file'],
    inputs=['sim_catalog', 'matched_file', 'bkg_catalog', 'model_file', 'config_file'],
    description='Compute per-waveform efficiency using matched_right cross-match',
)
def compute_efficiency_by_waveform(
    work_dir: str,
    sim_catalog: str,
    matched_file: str,
    bkg_catalog: str,
    livetime: float,
    model_file: str,
    search: str = "blf",
    nifo: int = 2,
    config_file: Optional[str] = None,
    ifar: str = "1mo",
    output_file: Optional[str] = None,
    **kwargs,
) -> dict:
    """Score full SIM catalog, cross-match with matched_right, compute per-waveform efficiency.

    Uses ``matched_right.parquet`` (from ``pycwb match-simulations``) to
    identify which injections were detected by the cWB pipeline.  The
    XGBoost IFAR threshold provides an additional cut.

    Parameters
    ----------
    work_dir : str
        Base directory.
    sim_catalog : str
        Path to the FULL SIM STDINJs catalog parquet (no splitting).
    matched_file : str
        Path to ``matched_right.parquet``.
    bkg_catalog : str
        Path to scored BKG catalog (with ``xgb_prob`` column).
    livetime : float
        Background live time in seconds.
    model_file : str
        Path to trained XGBoost model.
    ifar : str
        IFAR threshold: ``"1yr"``, ``"1mo"``, ``"1wk"``, ``"1day"``.
    output_file : str, optional
        Save per-waveform efficiency bar chart to this path.

    Returns
    -------
    dict
        ``efficiency_by_waveform`` list, ``prob_threshold``, ``ifar_sec``.
    """
    from .train_xgboost import _map_catalog_columns
    import xgboost as xgb

    def _resolve(p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(work_dir, p)

    sim_path = _resolve(sim_catalog)
    matched_path = _resolve(matched_file)
    bkg_path = _resolve(bkg_catalog)
    model_path = _resolve(model_file)

    ifar_sec = _IFAR_PRESETS.get(ifar, float(ifar) if ifar.replace(".", "").isdigit() else 31557600)

    # ── Check for unique-simulation mode ────────────────────────────────
    use_unique_sim = kwargs.get("use_unique_sim", False)
    exclude_vetoed = kwargs.get("exclude_vetoed", False)

    if use_unique_sim:
        # ── Unique-simulation-based per-waveform efficiency ─────────────
        return _compute_efficiency_by_waveform_matched(
            work_dir, matched_file, bkg_catalog, livetime, model_file,
            search, nifo, config_file, ifar, ifar_sec, output_file,
            exclude_vetoed,
        )

    # ── Original trigger-based method ────────────────────────────────────
    # Load model
    clf = xgb.XGBClassifier()
    clf.load_model(model_path)
    feature_names = clf.get_booster().feature_names
    logger.info("Model: %d features", len(feature_names))

    # ── Load & score full SIM catalog ────────────────────────────────────
    sim_df = pd.read_parquet(sim_path)
    from .evaluate import _preprocess_and_score
    sim_df["xgb_prob"] = _preprocess_and_score(sim_df, nifo, search, config_file, work_dir, clf)
    logger.info("Scored SIM: %d rows", len(sim_df))

    # Extract injection identity
    sim_df["inj_name"] = sim_df["injection"].apply(
        lambda x: x.get("name", "") if isinstance(x, dict) else ""
    )
    sim_df["inj_hrss"] = sim_df["injection"].apply(
        lambda x: float(x.get("hrss", np.nan)) if isinstance(x, dict) else np.nan
    )
    sim_df["inj_gps"] = sim_df["injection"].apply(
        lambda x: float(x.get("gps_time", np.nan)) if isinstance(x, dict) else np.nan
    )

    # ── Load matched_right, build match key set ──────────────────────────
    mr = pd.read_parquet(matched_path)
    mr_key = set()
    for _, row in mr.iterrows():
        mr_key.add((row["sim_name"], round(row["sim_hrss"], 12), round(row["sim_gps_time"], 6)))
    logger.info("Matched right: %d rows, %d unique (name,hrss,gps)", len(mr), len(mr_key))

    # Mark matched
    sim_df["matched"] = sim_df.apply(
        lambda r: (r["inj_name"], round(r["inj_hrss"], 12), round(r["inj_gps"], 6)) in mr_key,
        axis=1,
    )

    # ── Compute prob threshold from BKG ──────────────────────────────────
    bkg_df = pd.read_parquet(bkg_path)
    if "xgb_prob" not in bkg_df.columns:
        raise KeyError("BKG catalog must have 'xgb_prob' column")
    bkg_probs = np.sort(bkg_df["xgb_prob"].values)[::-1]
    far_values = np.arange(1, len(bkg_probs) + 1) / max(livetime, 1.0)
    target_far = 1.0 / ifar_sec
    idx = np.searchsorted(far_values, target_far)
    if idx >= len(bkg_probs):
        prob_threshold = bkg_probs[-1] + 0.01
    else:
        prob_threshold = bkg_probs[idx]
    logger.info("IFAR=%s: prob >= %.6f", ifar, prob_threshold)

    # ── Detection: matched AND above prob threshold ──────────────────────
    sim_df["detected"] = sim_df["matched"] & (sim_df["xgb_prob"] >= prob_threshold)

    # ── Per-waveform efficiency ──────────────────────────────────────────
    results = []
    waveform_names = sorted(s for s in sim_df["inj_name"].unique() if s)
    for name in waveform_names:
        sub = sim_df[sim_df["inj_name"] == name]
        n_total = len(sub)
        n_matched = int(sub["matched"].sum())
        n_detected = int(sub["detected"].sum())
        eff_matched = n_matched / max(n_total, 1)
        eff_detected = n_detected / max(n_total, 1)
        hrss_vals = sorted(sub["inj_hrss"].dropna().unique())
        results.append({
            "waveform": name,
            "n_total": n_total,
            "n_matched": n_matched,
            "n_detected": n_detected,
            "eff_matched": float(eff_matched),
            "eff_detected": float(eff_detected),
            "hrss_values": [float(h) for h in hrss_vals],
        })
        logger.info("  %-25s: detected=%d/%d (%.3f), matched=%d (%.3f)",
                     name, n_detected, n_total, eff_detected, n_matched, eff_matched)

    # ── Plot ─────────────────────────────────────────────────────────────
    if output_file:
        out_path = _resolve(output_file)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        _plot_waveform_efficiency(results, prob_threshold, ifar, ifar_sec, out_path)

    return {
        "efficiency_by_waveform": results,
        "prob_threshold": float(prob_threshold),
        "ifar_sec": ifar_sec,
    }


# ---------------------------------------------------------------------------
# Efficiency vs hrss by waveform (per-waveform curves)
# ---------------------------------------------------------------------------

@action_spec(
    outputs=['output_file', 'fit_parameters_file'],
    inputs=['sim_catalog', 'matched_file', 'bkg_catalog', 'model_file', 'config_file'],
    description='Efficiency vs hrss curves for each waveform, grouped by Q-factor',
)
def compute_efficiency_vs_hrss_by_waveform(
    work_dir: str,
    sim_catalog: str,
    matched_file: str,
    bkg_catalog: str,
    livetime: float,
    model_file: str,
    search: str = "blf",
    nifo: int = 2,
    config_file: Optional[str] = None,
    ifar: str = "1mo",
    output_file: Optional[str] = None,
    **kwargs,
) -> dict:
    """Efficiency vs hrss curves for each waveform, grouped by Q-factor.

    Produces a multi-panel plot with efficiency vs hrss for all waveforms,
    organized by Q-factor (Q3, Q9, Q100) in separate subplots.

    Parameters
    ----------
    work_dir : str
        Base directory.
    sim_catalog : str
        Path to the FULL SIM STDINJs catalog parquet.
    matched_file : str
        Path to ``matched_right.parquet``.
    bkg_catalog : str
        Path to scored BKG catalog.
    livetime : float
        Background live time in seconds.
    model_file : str
        Trained XGBoost model path.
    ifar : str
        IFAR threshold (``"1yr"``, ``"1mo"``, ``"1wk"``, ``"1day"``).
    output_file : str, optional
        Save plot to this path.

    Returns
    -------
    dict
        ``curves`` list of per-waveform efficiency data,
        ``prob_threshold``, ``ifar_sec``.
    """
    use_unique_sim = kwargs.get("use_unique_sim", False)
    if use_unique_sim:
        return _compute_efficiency_vs_hrss_by_waveform_matched(
            work_dir=work_dir,
            matched_file=matched_file,
            bkg_catalog=bkg_catalog,
            livetime=livetime,
            model_file=model_file,
            search=search,
            nifo=nifo,
            config_file=config_file,
            ifar_label=ifar,
            output_file=output_file,
            exclude_vetoed=kwargs.get("exclude_vetoed", False),
            fit_parameters_file=kwargs.get("fit_parameters_file"),
        )

    from .train_xgboost import _map_catalog_columns
    import xgboost as xgb

    def _resolve(p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(work_dir, p)

    sim_path = _resolve(sim_catalog)
    matched_path = _resolve(matched_file)
    bkg_path = _resolve(bkg_catalog)
    model_path = _resolve(model_file)

    ifar_sec = _IFAR_PRESETS.get(ifar, float(ifar) if ifar.replace(".", "").isdigit() else 31557600)

    # ── Load model ───────────────────────────────────────────────────────
    clf = xgb.XGBClassifier()
    clf.load_model(model_path)
    feature_names = clf.get_booster().feature_names

    # ── Score full SIM ───────────────────────────────────────────────────
    sim_df = pd.read_parquet(sim_path)
    from .evaluate import _preprocess_and_score
    sim_df["xgb_prob"] = _preprocess_and_score(sim_df, nifo, search, config_file, work_dir, clf)

    sim_df["inj_name"] = sim_df["injection"].apply(lambda x: x.get("name", "") if isinstance(x, dict) else "")
    sim_df["inj_hrss"] = sim_df["injection"].apply(lambda x: float(x.get("hrss", np.nan)) if isinstance(x, dict) else np.nan)
    sim_df["inj_gps"] = sim_df["injection"].apply(lambda x: float(x.get("gps_time", np.nan)) if isinstance(x, dict) else np.nan)

    # ── Matched ──────────────────────────────────────────────────────────
    mr = pd.read_parquet(matched_path)
    mr_key = set()
    for _, row in mr.iterrows():
        mr_key.add((row["sim_name"], round(row["sim_hrss"], 12), round(row["sim_gps_time"], 6)))
    sim_df["matched"] = sim_df.apply(
        lambda r: (r["inj_name"], round(r["inj_hrss"], 12), round(r["inj_gps"], 6)) in mr_key, axis=1,
    )

    # ── Prob threshold ───────────────────────────────────────────────────
    bkg_df = pd.read_parquet(bkg_path)
    bkg_probs = np.sort(bkg_df["xgb_prob"].values)[::-1]
    far_values = np.arange(1, len(bkg_probs) + 1) / max(livetime, 1.0)
    target_far = 1.0 / ifar_sec
    idx = np.searchsorted(far_values, target_far)
    prob_thr = bkg_probs[idx] if idx < len(bkg_probs) else 1.0
    logger.info("IFAR=%s: prob >= %.6f", ifar, prob_thr)

    sim_df["detected"] = sim_df["matched"] & (sim_df["xgb_prob"] >= prob_thr)

    # ── Per-waveform efficiency vs hrss ──────────────────────────────────
    curves = []
    for name in sorted(s for s in sim_df["inj_name"].unique() if s):
        sub = sim_df[sim_df["inj_name"] == name]
        # Extract Q and freq
        q_match = pd.Series([name]).str.extract(r"_Q(\d+)_", expand=False).iloc[0]
        f_match = pd.Series([name]).str.extract(r"_(\d+)Hz", expand=False).iloc[0]
        q_val = int(q_match) if q_match else 0
        f_val = int(f_match) if f_match else 0

        hrs_vals = sorted(sub["inj_hrss"].dropna().unique())
        eff_data = []
        for h in hrs_vals:
            s = sub[sub["inj_hrss"] == h]
            n_total = len(s)
            n_det = int(s["detected"].sum())
            eff_data.append({
                "hrss": float(h),
                "efficiency": float(n_det / max(n_total, 1)),
                "n_total": n_total,
                "n_detected": n_det,
            })
        curves.append({
            "waveform": name,
            "Q": q_val,
            "frequency": f_val,
            "data": eff_data,
        })

    # ── Plot ─────────────────────────────────────────────────────────────
    if output_file:
        out_path = _resolve(output_file)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        _plot_efficiency_by_waveform_panels(curves, prob_thr, ifar, ifar_sec, out_path)

    return {
        "curves": curves,
        "prob_threshold": float(prob_thr),
        "ifar_sec": ifar_sec,
    }


# ---------------------------------------------------------------------------
# Per-waveform hrss50 CSV report (multiple IFARs)
# ---------------------------------------------------------------------------

@action_spec(
    outputs=['output_csv'],
    inputs=['sim_catalog', 'matched_file', 'bkg_catalog', 'model_file', 'config_file'],
    description='Compute hrss50 for each waveform at multiple IFARs, save CSV',
)
def compute_hrss50_by_waveform_csv(
    work_dir: str,
    sim_catalog: str,
    matched_file: str,
    bkg_catalog: str,
    livetime: float,
    model_file: str,
    search: str = "blf",
    nifo: int = 2,
    config_file: Optional[str] = None,
    ifars: str = "1mo,1yr,10yr",
    output_csv: Optional[str] = None,
    **kwargs,
) -> dict:
    """Compute hrss50 for each waveform at multiple IFAR thresholds, save CSV.

    Parameters
    ----------
    ifars : str
        Comma-separated IFAR labels, e.g. ``"1mo,1yr,10yr"``.
    output_csv : str, optional
        Save CSV to this path.
    """
    from .train_xgboost import _map_catalog_columns
    import xgboost as xgb

    def _resolve(p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(work_dir, p)

    sim_path = _resolve(sim_catalog)
    matched_path = _resolve(matched_file)
    bkg_path = _resolve(bkg_catalog)
    model_path = _resolve(model_file)

    # Load model
    clf = xgb.XGBClassifier()
    clf.load_model(model_path)
    from pycwb.modules.statistics.sigmoid_fit import estimate_hrss, fit, logNfit

    # Score SIM
    sim_df = pd.read_parquet(sim_path)
    from .evaluate import _preprocess_and_score
    sim_df["xgb_prob"] = _preprocess_and_score(sim_df, nifo, search, config_file, work_dir, clf)
    sim_df["inj_name"] = sim_df["injection"].apply(lambda x: x.get("name", "") if isinstance(x, dict) else "")
    sim_df["inj_hrss"] = sim_df["injection"].apply(lambda x: float(x.get("hrss", np.nan)) if isinstance(x, dict) else np.nan)
    sim_df["inj_gps"] = sim_df["injection"].apply(lambda x: float(x.get("gps_time", np.nan)) if isinstance(x, dict) else np.nan)

    # Matched
    mr = pd.read_parquet(matched_path)
    mr_key = set()
    for _, row in mr.iterrows():
        mr_key.add((row["sim_name"], round(row["sim_hrss"], 12), round(row["sim_gps_time"], 6)))
    sim_df["matched"] = sim_df.apply(
        lambda r: (r["inj_name"], round(r["inj_hrss"], 12), round(r["inj_gps"], 6)) in mr_key, axis=1,
    )

    # BKG probs
    bkg_df = pd.read_parquet(bkg_path)
    bkg_probs = np.sort(bkg_df["xgb_prob"].values)[::-1]
    far_values = np.arange(1, len(bkg_probs) + 1) / max(livetime, 1.0)

    ifar_list = [s.strip() for s in ifars.split(",")]
    waveforms = sorted(s for s in sim_df["inj_name"].unique() if s)

    rows = []
    for ifar_label in ifar_list:
        ifar_sec = _IFAR_PRESETS.get(ifar_label, float(ifar_label) if ifar_label.replace(".", "").isdigit() else 31557600)
        target_far = 1.0 / ifar_sec
        idx = np.searchsorted(far_values, target_far)
        prob_thr = bkg_probs[idx] if idx < len(bkg_probs) else 1.0

        for name in waveforms:
            sub = sim_df[sim_df["inj_name"] == name]
            hrs_vals = sorted(sub["inj_hrss"].dropna().unique())
            effs = []
            eff_data = []
            for h in hrs_vals:
                s = sub[sub["inj_hrss"] == h]
                n_det = int((s["matched"] & (s["xgb_prob"] >= prob_thr)).sum())
                eff = n_det / max(len(s), 1)
                effs.append(eff)
                eff_data.append({
                    "hrss": float(h),
                    "efficiency": float(eff),
                    "n_total": int(len(s)),
                    "n_detected": int(n_det),
                })
            fit_result = _fit_efficiency_curve(name, eff_data, fit, estimate_hrss, logNfit)
            rows.append({
                "ifar": ifar_label,
                "ifar_sec": ifar_sec,
                "prob_threshold": float(prob_thr),
                "waveform": name,
                "fit_status": fit_result.get("status"),
                "hrss10": fit_result.get("hrss10"),
                "hrss50": fit_result.get("hrss50"),
                "hrss90": fit_result.get("hrss90"),
                "hrssEr": fit_result.get("hrssEr"),
                "chi2": fit_result.get("chi2"),
                "sigma": fit_result.get("sigma"),
                "betam": fit_result.get("betam"),
                "betap": fit_result.get("betap"),
                "flag": fit_result.get("flag"),
            })

    df_out = pd.DataFrame(rows)
    if output_csv:
        csv_path = _resolve(output_csv)
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        df_out.to_csv(csv_path, index=False)
        logger.info("hrss50 CSV → %s", csv_path)

    return {"hrss50_csv": df_out.to_dict(orient="records")}
