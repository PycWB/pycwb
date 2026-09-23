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
import re
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


def _parse_ifar_seconds(label):
    # Preserve the historical presets (notably the 30-day "1mo").
    units = {"s": 1., "day": 86400., "wk": 604800., "mo": 2592000., "yr": 31557600.}
    match = re.fullmatch(r"(\d+(?:\.\d+)?)(s|day|wk|mo|yr)", str(label))
    value = (_IFAR_PRESETS[label] if label in _IFAR_PRESETS else
             float(match[1]) * units[match[2]] if match else float(label))
    if not np.isfinite(value) or value <= 0:
        raise ValueError("IFAR must be a known preset or finite positive seconds")
    return float(value)


def _interpolate_hrss50(eff_curve: list[dict]) -> Optional[float]:
    """Return a measured crossing; an unbracketed crossing is not an estimate."""
    return _interpolate_hrss50_curve(
        np.array([d["hrss"] for d in eff_curve], dtype=float),
        np.array([d["efficiency"] for d in eff_curve], dtype=float),
    )


def _fit_efficiency_curve(name, eff_data, fit_func, estimate_func, logn_func) -> dict:
    hrs = np.array([d["hrss"] for d in eff_data], dtype=float)
    effs = np.array([d["efficiency"] for d in eff_data], dtype=float)
    valid = np.isfinite(hrs) & np.isfinite(effs) & (hrs > 0)
    hrs = hrs[valid]
    effs = effs[valid]
    if len(hrs) and effs.max() < .5:
        return {"status": "above_sampled_range", "hrss50": None, "bound": float(hrs.max())}
    if len(hrs) and effs.min() > .5:
        return {"status": "below_sampled_range", "hrss50": None, "bound": float(hrs.min())}
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


def _validate_fixed_hrss_population(frame: pd.DataFrame) -> None:
    """Reject seed amplitudes from target-SNR populations in amplitude reports."""
    for column in ("sim_target_snr", "sim_targeted_snr"):
        if column in frame and pd.to_numeric(frame[column], errors="coerce").fillna(0).ne(0).any():
            raise ValueError(
                "Target-SNR populations are not supported by hrss efficiency reports: "
                "sim_hrss is the unscaled input amplitude. Use a fixed-hrss population."
            )
    if "sim_snr_scale" in frame:
        scale = pd.to_numeric(frame["sim_snr_scale"], errors="coerce")
        if (scale.notna() & scale.ne(1)).any():
            raise ValueError("SNR-scaled injections require a fixed-hrss population for hrss efficiency reports")


def _parse_waveform_q_frequency(name: str) -> tuple[float, int]:
    # Both native descriptive names and the original cWB SG/SGE names.
    compact = re.fullmatch(r"SGE?(\d+)Q(\d+(?:[d.]\d+)?)", name)
    if compact:
        return float(compact[2].replace("d", ".")), int(compact[1])
    q = re.search(r"_Q(\d+(?:[d.]\d+)?)(?:_|$)", name)
    f = re.search(r"_(\d+)Hz", name)
    return float(q[1].replace("d", ".")) if q else 0, int(f[1]) if f else 0


def _interpolate_hrss50_curve(hrs: np.ndarray, effs: np.ndarray) -> Optional[float]:
    """Interpolate a bracketed crossing in log amplitude; never return a bound as a point."""
    hrs, effs = np.asarray(hrs, dtype=float), np.asarray(effs, dtype=float)
    valid = np.isfinite(hrs) & (hrs > 0) & np.isfinite(effs)
    hrs, effs = hrs[valid], effs[valid]
    if not len(hrs):
        return None
    order = np.argsort(hrs)
    hrs, effs = hrs[order], effs[order]
    if effs.min() > .5 or effs.max() < .5:
        return None
    for i in range(len(effs)):
        if effs[i] == .5:
            return float(hrs[i])
        if i + 1 < len(effs) and (effs[i] - .5) * (effs[i + 1] - .5) < 0:
            fraction = (.5 - effs[i]) / (effs[i + 1] - effs[i])
            return float(np.exp(np.log(hrs[i]) + fraction * np.log(hrs[i + 1] / hrs[i])))
    return None


def _validate_unique_simulations(matched: pd.DataFrame) -> None:
    """Require the right-joined, unique-selected catalog; never silently double count."""
    ids = matched["sim_sim_idx"]
    if ids.isna().any() or ids.duplicated().any():
        raise ValueError("Efficiency requires one unique row per sim_sim_idx; "
                         "apply unique simulation matching and use its right join first")


def _empirical_probability_detection(scores, background, livetime, ifar_sec):
    """Use inclusive empirical tail counts, including ties and gaps between ranks.

    Zero background exceedances have empirical FAR zero (as in score_mdc_catalog),
    not a measured infinite exposure. A finite-background upper limit is a separate
    inference. The returned threshold is only a display representation of this cut.
    """
    if not np.isfinite(livetime) or livetime <= 0 or not np.isfinite(ifar_sec) or ifar_sec <= 0:
        raise ValueError("livetime and ifar_sec must be finite and positive")
    background = np.asarray(background, dtype=float)
    background = np.sort(background[np.isfinite(background)])
    if not len(background):
        raise ValueError("No finite background scores available for IFAR calibration")
    scores = np.asarray(scores, dtype=float)
    counts = len(background) - np.searchsorted(background, scores, side="left")
    detected = np.isfinite(scores) & (counts / livetime <= 1. / ifar_sec)
    # At the highest disallowed observed score the inclusive tail is too large.
    tail = len(background) - np.searchsorted(background, background, side="left")
    disallowed = background[tail / livetime > 1. / ifar_sec]
    threshold = np.nextafter(disallowed[-1], np.inf) if len(disallowed) else -np.inf
    return detected, float(threshold)


def _matched_ranking_scores(matched, work_dir, ranking_par="xgb_prob", scored_file=None,
                            model_file=None, nifo=2, search="blf", config_file=None):
    """Align scores by event ID; prediction-cut failures and misses remain NaN.

    Reusing a scored catalog keeps ranking hooks and prediction cuts identical
    to the background workflow. Extra (non-unique) trigger IDs are harmless.
    """
    def resolve(path):
        return path if os.path.isabs(path) else os.path.join(work_dir, path)
    if scored_file:
        scores = pd.read_parquet(resolve(scored_file), columns=["id", ranking_par])
    elif model_file:
        import xgboost as xgb
        from .evaluate import _score_catalog_dataframe
        recovered = matched[matched["id"].notna()].copy()
        if recovered.empty:
            return pd.Series(np.nan, index=matched.index)
        clf = xgb.XGBClassifier()
        clf.load_model(resolve(model_file))
        scores = _score_catalog_dataframe(recovered, nifo, search, config_file, work_dir, clf)
    else:
        if ranking_par not in matched:
            raise ValueError(f"Provide scored_file, model_file, or matched column {ranking_par}")
        return pd.to_numeric(matched[ranking_par], errors="coerce").where(matched.id.notna())
    if ranking_par not in scores:
        raise ValueError(f"Missing ranking column {ranking_par}")
    if scores.id.isna().any() or scores.id.duplicated().any():
        raise ValueError("Scored events must have unique, nonnull IDs")
    values = pd.to_numeric(scores.set_index("id")[ranking_par], errors="coerce")
    return matched["id"].map(values).where(matched.id.notna())


def _ranking_metadata(ranking_par, threshold):
    return {"ranking_par": ranking_par, "ranking_threshold": float(threshold),
            "prob_threshold": float(threshold) if ranking_par == "xgb_prob" else None,
            "ifar_convention": "inclusive_empirical_tail"}


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
    ranking_par: str = "xgb_prob",
    scored_file: Optional[str] = None,
) -> dict:
    """Efficiency-vs-hrss curves using one matched_right row per simulation."""
    from pycwb.modules.statistics.sigmoid_fit import estimate_hrss, fit, logNfit

    def _resolve(p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(work_dir, p)

    ifar_sec = _parse_ifar_seconds(ifar_label)

    mr = pd.read_parquet(_resolve(matched_file))
    _validate_unique_simulations(mr)
    _validate_fixed_hrss_population(mr)
    if exclude_vetoed:
        veto_mask = (
            mr["sim_vetoed_cat0"].fillna(False).astype(bool)
            | mr["sim_vetoed_cat1"].fillna(False).astype(bool)
            | mr["sim_vetoed_cat2"].fillna(False).astype(bool)
            | mr["sim_across_segments"].fillna(False).astype(bool)
        )
        mr = mr[~veto_mask].reset_index(drop=True)

    scores = _matched_ranking_scores(mr, work_dir, ranking_par, scored_file,
                                     model_file, nifo, search, config_file)
    bkg_df = pd.read_parquet(_resolve(bkg_catalog), columns=[ranking_par])
    detected, prob_threshold = _empirical_probability_detection(
        scores, bkg_df[ranking_par], livetime, ifar_sec,
    )
    mr["detected"] = mr["id"].notna() & detected

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
            **_ranking_metadata(ranking_par, prob_threshold),
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
            show_sigmoid_fit=True, ranking_par=ranking_par,
        )

    return {
        "curves": curves,
        "fit_parameters": fit_rows,
        **_ranking_metadata(ranking_par, prob_threshold),
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
    ranking_par: str = "xgb_prob",
    scored_file: Optional[str] = None,
) -> dict:
    """Per-waveform efficiency counting unique simulations from matched_right.parquet.

    Each row in matched_right.parquet is one unique simulation (by sim_sim_idx).
    Recovered sims have non-null trigger columns (id, rho, …); missed sims have
    null trigger columns.  We score the recovered sims with XGBoost and count
    unique sim_sim_idx for numerator and denominator.
    """
    def _resolve(p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(work_dir, p)

    mr_path = _resolve(matched_file)
    bkg_path = _resolve(bkg_catalog)

    # ── Load matched_right ───────────────────────────────────────────────
    mr = pd.read_parquet(mr_path)
    _validate_unique_simulations(mr)
    _validate_fixed_hrss_population(mr)
    n_total_sims = mr["sim_sim_idx"].nunique()
    n_recovered_cwb = mr["id"].notna().sum()
    logger.info(
        "matched_right: %d rows, %d unique sims, %d recovered by cWB",
        len(mr), n_total_sims, n_recovered_cwb,
    )

    scores = _matched_ranking_scores(mr, work_dir, ranking_par, scored_file,
                                     model_file, nifo, search, config_file)
    bkg_df = pd.read_parquet(bkg_path, columns=[ranking_par])
    detected, prob_threshold = _empirical_probability_detection(
        scores, bkg_df[ranking_par], livetime, ifar_sec,
    )
    logger.info("IFAR=%s: %s >= %.6f", ifar_label, ranking_par, prob_threshold)

    # ── Detection: cWB-recovered AND XGBoost above threshold ─────────────
    mr["detected"] = mr["id"].notna() & detected

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
        _plot_waveform_efficiency(results, prob_threshold, ifar_label, ifar_sec, out_path,
                                  ranking_par=ranking_par)

    return {
        "efficiency_by_waveform": results,
        **_ranking_metadata(ranking_par, prob_threshold),
        "ifar_sec": ifar_sec,
        "method": "unique_simulation",
        "exclude_vetoed": exclude_vetoed,
    }
