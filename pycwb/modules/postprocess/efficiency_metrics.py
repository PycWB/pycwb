"""Efficiency metrics: hrss interpolation, sigmoid fits, and unique-simulation
efficiency computation.

This module owns the numerical efficiency calculations used by the public
efficiency actions in :mod:`pycwb.modules.postprocess.plot_efficiency`.  It
depends on :mod:`pycwb.modules.postprocess.efficiency_plots` only to render the
figures produced by the unique-simulation compute helpers.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd

from pycwb.modules.postprocess.efficiency_plots import (
    _plot_efficiency_by_waveform_panels,
    _plot_waveform_efficiency,
)

logger = logging.getLogger(__name__)

# IFAR presets (seconds)
_IFAR_PRESETS = {
    "10yr": 315576000,
    "1yr": 31557600,
    "6mo": 15778800,
    "1mo": 2592000,
    "1wk": 604800,
    "1day": 86400,
}


def _interpolate_hrss50(eff_curve: list[dict]) -> Optional[float]:
    """Find hrss where efficiency crosses 50% by linear interpolation."""
    hrs = np.array([d["hrss"] for d in eff_curve])
    effs = np.array([d["efficiency"] for d in eff_curve])

    # Check if 50% is within range
    if effs.min() > 0.5:
        logger.warning("Min efficiency %.3f > 0.5 — hrss50 is below data range", effs.min())
        return float(hrs[0])  # return lowest hrss as lower bound
    if effs.max() < 0.5:
        logger.warning("Max efficiency %.3f < 0.5 — hrss50 is above data range", effs.max())
        return float(hrs[-1])

    # Find crossing point
    for i in range(len(effs) - 1):
        if (effs[i] >= 0.5 and effs[i + 1] <= 0.5) or (effs[i] <= 0.5 and effs[i + 1] >= 0.5):
            # Linear interpolation in log space
            log_h1, log_h2 = np.log10(hrs[i]), np.log10(hrs[i + 1])
            e1, e2 = effs[i], effs[i + 1]
            if abs(e2 - e1) < 1e-12:
                log_h50 = log_h1
            else:
                log_h50 = log_h1 + (0.5 - e1) / (e2 - e1) * (log_h2 - log_h1)
            return float(10 ** log_h50)

    return None


def _fit_efficiency_curve(name, eff_data, fit_func, estimate_func, logn_func) -> dict:
    hrs = np.array([d["hrss"] for d in eff_data], dtype=float)
    effs = np.array([d["efficiency"] for d in eff_data], dtype=float)
    valid = np.isfinite(hrs) & np.isfinite(effs) & (hrs > 0)
    hrs = hrs[valid]
    effs = effs[valid]
    if len(hrs) < 3:
        return {"status": "skipped", "reason": "fewer than 3 hrss points"}

    order = np.argsort(hrs)
    hrs = hrs[order]
    effs = effs[order]
    try:
        chi2, hrss50, hrssEr, sigma, betam, betap, flag = fit_func(np.log10(hrs), effs)
        xlim = (float(np.log10(hrs.min())), float(np.log10(hrs.max())))
        hrss10 = estimate_func((hrss50, sigma, betam, betap, flag), xlim, 0.1)
        hrss90 = estimate_func((hrss50, sigma, betam, betap, flag), xlim, 0.9)
        fit_x = np.linspace(xlim[0], xlim[1], 300)
        fit_y = logn_func(fit_x, np.log10(hrss50), sigma, betam, betap, flag)
        return {
            "status": "ok",
            "chi2": float(chi2),
            "hrss10": float(hrss10) if np.isfinite(hrss10) else np.nan,
            "hrss50": float(hrss50),
            "hrss90": float(hrss90) if np.isfinite(hrss90) else np.nan,
            "hrssEr": float(hrssEr),
            "sigma": float(sigma),
            "betam": float(betam),
            "betap": float(betap),
            "flag": int(flag),
            "fit_x": (10 ** fit_x).tolist(),
            "fit_y": fit_y.tolist(),
        }
    except Exception as exc:
        logger.warning("Sigmoid fit failed for %s: %s", name, exc)
        return {"status": "failed", "reason": str(exc)}


def _parse_waveform_q_frequency(name: str) -> tuple[int, int]:
    q_match = pd.Series([name]).str.extract(r"_Q(\d+)_", expand=False).iloc[0]
    f_match = pd.Series([name]).str.extract(r"_(\d+)Hz", expand=False).iloc[0]
    q_val = int(q_match) if q_match else 0
    f_val = int(f_match) if f_match else 0
    return q_val, f_val


def _interpolate_hrss50_curve(hrs: np.ndarray, effs: np.ndarray) -> Optional[float]:
    """Interpolate hrss at 50% efficiency from per-waveform data."""
    if effs.min() > 0.5:
        return float(hrs[0])
    if effs.max() < 0.5:
        return float(hrs[-1])
    for i in range(len(effs) - 1):
        if (effs[i] >= 0.5 and effs[i + 1] <= 0.5) or (effs[i] <= 0.5 and effs[i + 1] >= 0.5):
            log_h1, log_h2 = np.log10(hrs[i]), np.log10(hrs[i + 1])
            e1, e2 = effs[i], effs[i + 1]
            if abs(e2 - e1) < 1e-12:
                return float(10 ** log_h1)
            log_h50 = log_h1 + (0.5 - e1) / (e2 - e1) * (log_h2 - log_h1)
            return float(10 ** log_h50)
    return None


def _compute_efficiency_vs_hrss_by_waveform_matched(
    work_dir: str,
    matched_file: str,
    bkg_catalog: str,
    livetime: float,
    model_file: str,
    search: str,
    nifo: int,
    config_file: Optional[str],
    ifar_label: str,
    output_file: Optional[str],
    exclude_vetoed: bool = False,
    fit_parameters_file: Optional[str] = None,
) -> dict:
    """Efficiency-vs-hrss curves using one matched_right row per simulation."""
    import xgboost as xgb
    from pycwb.modules.statistics.sigmoid_fit import estimate_hrss, fit, logNfit

    def _resolve(p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(work_dir, p)

    ifar_sec = _IFAR_PRESETS.get(
        ifar_label,
        float(ifar_label) if ifar_label.replace(".", "").isdigit() else 31557600,
    )

    mr = pd.read_parquet(_resolve(matched_file))
    if exclude_vetoed:
        veto_mask = (
            mr["sim_vetoed_cat0"].fillna(False).astype(bool)
            | mr["sim_vetoed_cat1"].fillna(False).astype(bool)
            | mr["sim_vetoed_cat2"].fillna(False).astype(bool)
            | mr["sim_across_segments"].fillna(False).astype(bool)
        )
        mr = mr[~veto_mask].reset_index(drop=True)

    clf = xgb.XGBClassifier()
    clf.load_model(_resolve(model_file))
    from .evaluate import _preprocess_and_score

    recovered = mr[mr["id"].notna()].copy()
    mr["xgb_prob"] = 0.0
    if len(recovered) > 0:
        recovered["xgb_prob"] = _preprocess_and_score(
            recovered, nifo, search, config_file, work_dir, clf,
        )
        mr.loc[recovered.index, "xgb_prob"] = recovered["xgb_prob"]

    bkg_df = pd.read_parquet(_resolve(bkg_catalog))
    bkg_probs = np.sort(bkg_df["xgb_prob"].values)[::-1]
    far_values = np.arange(1, len(bkg_probs) + 1) / max(livetime, 1.0)
    target_far = 1.0 / ifar_sec
    idx = np.searchsorted(far_values, target_far)
    prob_threshold = bkg_probs[idx] if idx < len(bkg_probs) else 1.0
    mr["detected"] = mr["id"].notna() & (mr["xgb_prob"] >= prob_threshold)

    curves = []
    fit_rows = []
    waveform_names = sorted(
        s for s in mr["sim_name"].dropna().unique()
        if not (isinstance(s, float) and np.isnan(s))
    )
    for name in waveform_names:
        sub = mr[mr["sim_name"] == name].copy()
        q_val, f_val = _parse_waveform_q_frequency(name)
        hrss_vals = sorted(pd.to_numeric(sub["sim_hrss"], errors="coerce").dropna().unique())
        eff_data = []
        for h in hrss_vals:
            s = sub[pd.to_numeric(sub["sim_hrss"], errors="coerce") == h]
            n_total = s["sim_sim_idx"].nunique()
            n_det = int(s["detected"].sum())
            eff_data.append({
                "hrss": float(h),
                "efficiency": float(n_det / max(n_total, 1)),
                "n_total": int(n_total),
                "n_detected": int(n_det),
            })

        fit_result = _fit_efficiency_curve(name, eff_data, fit, estimate_hrss, logNfit)
        curves.append({
            "waveform": name,
            "Q": q_val,
            "frequency": f_val,
            "data": eff_data,
            "fit": fit_result,
        })
        fit_rows.append({
            "waveform": name,
            "ifar": ifar_label,
            "ifar_sec": ifar_sec,
            "prob_threshold": float(prob_threshold),
            **{k: v for k, v in fit_result.items() if k != "fit_x" and k != "fit_y"},
        })

    if fit_parameters_file:
        fit_path = _resolve(fit_parameters_file)
        os.makedirs(os.path.dirname(fit_path) or ".", exist_ok=True)
        pd.DataFrame(fit_rows).to_csv(fit_path, index=False)
        logger.info("Waveform sigmoid fit parameters → %s", fit_path)

    if output_file:
        out_path = _resolve(output_file)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        _plot_efficiency_by_waveform_panels(
            curves, prob_threshold, ifar_label, ifar_sec, out_path,
            show_sigmoid_fit=True,
        )

    return {
        "curves": curves,
        "fit_parameters": fit_rows,
        "prob_threshold": float(prob_threshold),
        "ifar_sec": ifar_sec,
        "method": "unique_simulation_sigmoid_fit",
        "exclude_vetoed": exclude_vetoed,
    }


def _compute_efficiency_by_waveform_matched(
    work_dir: str,
    matched_file: str,
    bkg_catalog: str,
    livetime: float,
    model_file: str,
    search: str,
    nifo: int,
    config_file: Optional[str],
    ifar_label: str,
    ifar_sec: float,
    output_file: Optional[str],
    exclude_vetoed: bool = False,
) -> dict:
    """Per-waveform efficiency counting unique simulations from matched_right.parquet.

    Each row in matched_right.parquet is one unique simulation (by sim_sim_idx).
    Recovered sims have non-null trigger columns (id, rho, …); missed sims have
    null trigger columns.  We score the recovered sims with XGBoost and count
    unique sim_sim_idx for numerator and denominator.
    """
    import xgboost as xgb

    def _resolve(p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(work_dir, p)

    mr_path = _resolve(matched_file)
    bkg_path = _resolve(bkg_catalog)
    model_path = _resolve(model_file)

    # ── Load matched_right ───────────────────────────────────────────────
    mr = pd.read_parquet(mr_path)
    n_total_sims = mr["sim_sim_idx"].nunique()
    n_recovered_cwb = mr["id"].notna().sum()
    logger.info(
        "matched_right: %d rows, %d unique sims, %d recovered by cWB",
        len(mr), n_total_sims, n_recovered_cwb,
    )

    # ── Load model & score recovered sims ────────────────────────────────
    clf = xgb.XGBClassifier()
    clf.load_model(model_path)

    from .evaluate import _preprocess_and_score
    recovered = mr[mr["id"].notna()].copy()
    if len(recovered) > 0:
        recovered["xgb_prob"] = _preprocess_and_score(
            recovered, nifo, search, config_file, work_dir, clf,
        )
        mr["xgb_prob"] = 0.0
        mr.loc[recovered.index, "xgb_prob"] = recovered["xgb_prob"]
    else:
        mr["xgb_prob"] = 0.0

    # ── Compute prob threshold from BKG ──────────────────────────────────
    bkg_df = pd.read_parquet(bkg_path)
    bkg_probs = np.sort(bkg_df["xgb_prob"].values)[::-1]
    far_values = np.arange(1, len(bkg_probs) + 1) / max(livetime, 1.0)
    target_far = 1.0 / ifar_sec
    idx = np.searchsorted(far_values, target_far)
    prob_threshold = bkg_probs[idx] if idx < len(bkg_probs) else 1.0
    logger.info("IFAR=%s: prob >= %.6f", ifar_label, prob_threshold)

    # ── Detection: cWB-recovered AND XGBoost above threshold ─────────────
    mr["detected"] = mr["id"].notna() & (mr["xgb_prob"] >= prob_threshold)

    # ── Veto filter for denominator (optional) ───────────────────────────
    if exclude_vetoed:
        veto_mask = (
            mr["sim_vetoed_cat0"].fillna(False).astype(bool)
            | mr["sim_vetoed_cat1"].fillna(False).astype(bool)
            | mr["sim_vetoed_cat2"].fillna(False).astype(bool)
            | mr["sim_across_segments"].fillna(False).astype(bool)
        )
        mr_denom = mr[~veto_mask]
        logger.info("Vetoed sims excluded: %d → %d denominator", len(mr), len(mr_denom))
    else:
        mr_denom = mr

    # ── Per-waveform efficiency ──────────────────────────────────────────
    results = []
    waveform_names = sorted(s for s in mr["sim_name"].unique() if s and not (isinstance(s, float) and np.isnan(s)))
    for name in waveform_names:
        sub = mr_denom[mr_denom["sim_name"] == name]
        n_total = sub["sim_sim_idx"].nunique()
        n_detected = int(sub["detected"].sum())
        n_recovered = int(sub["id"].notna().sum())
        eff_detected = n_detected / max(n_total, 1)
        eff_recovered = n_recovered / max(n_total, 1)
        hrss_vals = sorted(pd.to_numeric(sub["sim_hrss"], errors="coerce").dropna().unique())
        results.append({
            "waveform": name,
            "n_total": n_total,
            "n_recovered": n_recovered,
            "n_detected": n_detected,
            "eff_recovered": float(eff_recovered),
            "eff_detected": float(eff_detected),
            "hrss_values": [float(h) for h in hrss_vals],
        })
        logger.info("  %-25s: detected=%d/%d (%.3f), recovered=%d (%.3f)",
                     name, n_detected, n_total, eff_detected, n_recovered, eff_recovered)

    # ── Plot ─────────────────────────────────────────────────────────────
    if output_file:
        out_path = _resolve(output_file)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        _plot_waveform_efficiency(results, prob_threshold, ifar_label, ifar_sec, out_path)

    return {
        "efficiency_by_waveform": results,
        "prob_threshold": float(prob_threshold),
        "ifar_sec": ifar_sec,
        "method": "unique_simulation",
        "exclude_vetoed": exclude_vetoed,
    }
