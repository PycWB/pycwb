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

import json
import logging
import os
from typing import Optional

import numpy as np
import pandas as pd

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
# Helpers
# ---------------------------------------------------------------------------

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


def _plot_efficiency_curve(
    eff_curve: list[dict],
    hrss50: Optional[float],
    ifar_label: str,
    ifar_sec: float,
    prob_threshold: float,
    output_path: str,
) -> None:
    """Save an efficiency vs hrss plot."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping plot")
        return

    hrs = np.array([d["hrss"] for d in eff_curve])
    effs = np.array([d["efficiency"] for d in eff_curve]) * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogx(hrs, effs, "o-", color="steelblue", markersize=6, linewidth=2)
    ax.axhline(50, color="gray", linestyle="--", alpha=0.5, label="50% efficiency")
    if hrss50 is not None:
        ax.axvline(hrss50, color="crimson", linestyle="--", alpha=0.7,
                   label=f"hrss50 = {hrss50:.2e}")
    ax.set_xlabel("hrss")
    ax.set_ylabel("Efficiency [%]")
    ax.set_title(f"Efficiency vs hrss  —  IFAR = {ifar_label} ({ifar_sec:.0f} s)\n"
                 f"prob threshold = {prob_threshold:.4f}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    logger.info("Plot saved → %s", output_path)


# ---------------------------------------------------------------------------
# Waveform-level efficiency using matched_right.parquet
# ---------------------------------------------------------------------------

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


def _plot_waveform_efficiency(
    results: list[dict],
    prob_threshold: float,
    ifar_label: str,
    ifar_sec: float,
    output_path: str,
) -> None:
    """Bar chart of per-waveform efficiency."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping plot")
        return

    names = [r["waveform"] for r in results]
    effs = [r["eff_detected"] * 100 for r in results]

    # Sort by efficiency
    order = np.argsort(effs)
    names = [names[i] for i in order]
    effs = [effs[i] for i in order]

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.RdYlGn([e / 100 for e in effs])
    bars = ax.barh(range(len(names)), effs, color=colors, edgecolor="gray", linewidth=0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Efficiency [%]")
    ax.set_title(f"Efficiency by Waveform  —  IFAR = {ifar_label} ({ifar_sec:.0f} s)\n"
                 f"prob threshold = {prob_threshold:.4f}")
    ax.axvline(50, color="gray", linestyle="--", alpha=0.5, label="50%")
    ax.set_xlim(0, 105)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3, axis="x")

    # Add value labels
    for i, (e, r) in enumerate(zip(effs, [results[j] for j in order])):
        ax.text(e + 1, i, f"{e:.0f}% ({r['n_detected']}/{r['n_total']})",
                va="center", fontsize=7)

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    logger.info("Waveform efficiency plot → %s", output_path)


# ---------------------------------------------------------------------------
# Efficiency vs hrss by waveform (per-waveform curves)
# ---------------------------------------------------------------------------

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


def _plot_efficiency_by_waveform_panels(
    curves: list[dict],
    prob_threshold: float,
    ifar_label: str,
    ifar_sec: float,
    output_path: str,
) -> None:
    """Multi-panel plot: efficiency vs hrss, one panel per Q-factor."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping plot")
        return

    # Group by Q
    q_groups = {3: [], 9: [], 100: []}
    for c in curves:
        q = c["Q"]
        if q in q_groups:
            q_groups[q].append(c)

    existing_qs = [q for q in [3, 9, 100] if q_groups[q]]
    n_panels = len(existing_qs)
    if n_panels == 0:
        return

    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5), squeeze=False)
    axes = axes[0]

    # Color map per frequency
    all_freqs = sorted(set(c["frequency"] for c in curves))
    cmap = plt.cm.tab10
    freq_colors = {f: cmap(i % 10) for i, f in enumerate(all_freqs)}

    for pi, q_val in enumerate(existing_qs):
        ax = axes[pi]
        group = sorted(q_groups[q_val], key=lambda c: c["frequency"])

        for c in group:
            hrs = [d["hrss"] for d in c["data"]]
            effs = [d["efficiency"] * 100 for d in c["data"]]
            color = freq_colors[c["frequency"]]
            ax.semilogx(hrs, effs, "o-", color=color, markersize=4, linewidth=1.5,
                        label=f"{c['frequency']}Hz", alpha=0.8)

        ax.axhline(50, color="gray", linestyle="--", alpha=0.4)
        ax.set_xlabel("hrss")
        ax.set_ylabel("Efficiency [%]")
        ax.set_title(f"Q = {q_val}")
        ax.set_ylim(-5, 105)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="lower right")

    fig.suptitle(f"Efficiency vs hrss by Waveform  —  IFAR = {ifar_label} ({ifar_sec:.0f} s)\n"
                 f"prob threshold = {prob_threshold:.4f}",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Efficiency vs hrss by waveform → %s", output_path)


# ---------------------------------------------------------------------------
# Per-waveform hrss50 CSV report (multiple IFARs)
# ---------------------------------------------------------------------------

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
            hrs_arr = np.array(hrs_vals)
            effs = []
            for h in hrs_vals:
                s = sub[sub["inj_hrss"] == h]
                n_det = int((s["matched"] & (s["xgb_prob"] >= prob_thr)).sum())
                effs.append(n_det / max(len(s), 1))
            effs = np.array(effs)
            hrss50 = _interpolate_hrss50_curve(hrs_arr, effs)
            rows.append({
                "ifar": ifar_label,
                "ifar_sec": ifar_sec,
                "prob_threshold": float(prob_thr),
                "waveform": name,
                "hrss50": float(hrss50) if hrss50 is not None else None,
            })

    df_out = pd.DataFrame(rows)
    if output_csv:
        csv_path = _resolve(output_csv)
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        df_out.to_csv(csv_path, index=False)
        logger.info("hrss50 CSV → %s", csv_path)

    return {"hrss50_csv": df_out.to_dict(orient="records")}


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


# ---------------------------------------------------------------------------
# Unique-simulation-based per-waveform efficiency (uses matched_right directly)
# ---------------------------------------------------------------------------

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
