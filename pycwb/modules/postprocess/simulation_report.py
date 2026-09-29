"""Report actions for unique-injection efficiency with explicit ranking cuts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pycwb.modules.postprocess.efficiency_metrics import (
    _interpolate_hrss50_curve,
    _validate_unique_simulations,
    _validate_fixed_hrss_population,
)
from pycwb.post_production.action_spec import action_spec


@action_spec(
    inputs=["matched_file", "scored_file"],
    outputs=["output_dir"],
    description="Report injection denominators, efficiency curves and measured hrss50",
)
def simulation_efficiency(
    work_dir,
    matched_file,
    output_dir,
    ranking_par="rho_alt",
    threshold=0.0,
    comparison=">",
    scored_file=None,
    amplitude_rtol=0.0,
    **kwargs,
):
    """Count unique injected signals, including misses, at an explicit rank cut.

    The matched-right table supplies the denominator. Optional scored_file
    supplies one score per recovered event ID; absent events remain misses.
    amplitude_rtol=0.1 reproduces cWB simulation=1's first-matching amplitude
    grouping, within each waveform. The default groups exact amplitudes.
    This action accepts any ranking (raw rho, XGBoost probability, or rhor).
    It reports a measured hrss50 crossing or a bound, without extrapolation.
    """
    if comparison not in (">", ">=") or not np.isfinite(threshold):
        raise ValueError("Use a finite threshold and > or >= comparison")
    if not np.isfinite(amplitude_rtol) or not 0 <= amplitude_rtol < 1:
        raise ValueError("amplitude_rtol must be in [0,1)")
    frame = pd.read_parquet(Path(work_dir) / matched_file)
    _validate_unique_simulations(frame)
    _validate_fixed_hrss_population(frame)
    required = {"sim_sim_idx", "sim_name", "sim_hrss", "id"}
    if required - set(frame):
        raise ValueError(f"Missing simulation columns: {required - set(frame)}")
    if scored_file:
        scores = pd.read_parquet(
            Path(work_dir) / scored_file, columns=["id", ranking_par]
        )
        if scores.id.isna().any() or scores.id.duplicated().any():
            raise ValueError("Scored events must have unique, nonnull IDs")
        if not scores.id.isin(frame.id.dropna()).all():
            raise ValueError(
                "Scored events are not members of the matched simulation table"
            )
        frame = frame.drop(columns=[ranking_par], errors="ignore").merge(
            scores, on="id", how="left", validate="many_to_one"
        )
    if ranking_par not in frame:
        raise ValueError(f"Missing ranking column {ranking_par}")
    amplitudes = frame.sim_hrss.to_numpy(dtype=float)
    if not np.all(np.isfinite(amplitudes) & (amplitudes > 0)):
        raise ValueError("Every injection needs finite positive sim_hrss")
    if frame.sim_name.isna().any():
        raise ValueError("Every injection needs a waveform name")
    values = pd.to_numeric(frame[ranking_par], errors="coerce")
    passes = values.gt(threshold) if comparison == ">" else values.ge(threshold)
    frame["detected"] = frame.id.notna() & np.isfinite(values) & passes
    rows = []
    summaries = []
    for name, group in frame.groupby("sim_name", sort=True):
        centers = []
        members = []
        for index, row in group.iterrows():
            h = float(row.sim_hrss)
            slot = next(
                (
                    i
                    for i, c in enumerate(centers)
                    if c == h or abs(c - h) < amplitude_rtol * h
                ),
                None,
            )
            if slot is None:
                slot = len(centers)
                centers.append(h)
                members.append([])
            members[slot].append(index)
        local = []
        for h, indices in zip(centers, members):
            subset = frame.loc[indices]
            n = len(subset)
            k = int(subset.detected.sum())
            p = k / n
            z = 1.959963984540054
            denom = 1 + z * z / n
            center = (p + z * z / (2 * n)) / denom
            half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
            item = {
                "waveform": name,
                "hrss": h,
                "n_injected": n,
                "n_recovered": int(subset.id.notna().sum()),
                "n_detected": k,
                "efficiency": p,
                "wilson_low": center - half,
                "wilson_high": center + half,
            }
            rows.append(item)
            local.append(item)
        local = sorted(local, key=lambda x: x["hrss"])
        hs = np.array([r["hrss"] for r in local])
        eff = np.array([r["efficiency"] for r in local])
        crossing = _interpolate_hrss50_curve(hs, eff)
        status = (
            "measured"
            if crossing is not None
            else (
                "above_sampled_range"
                if max(eff) < 0.5
                else "below_sampled_range"
                if min(eff) > 0.5
                else "not_bracketed"
            )
        )
        summaries.append(
            {
                "waveform": name,
                "n_injected": len(group),
                "n_detected": int(group.detected.sum()),
                "hrss50": crossing,
                "status": status,
                "bound": float(max(hs))
                if status == "above_sampled_range"
                else float(min(hs))
                if status == "below_sampled_range"
                else None,
            }
        )
    out = Path(work_dir) / output_dir
    out.mkdir(parents=True, exist_ok=True)
    curve = pd.DataFrame(
        rows,
        columns=[
            "waveform",
            "hrss",
            "n_injected",
            "n_recovered",
            "n_detected",
            "efficiency",
            "wilson_low",
            "wilson_high",
        ],
    )
    curve.to_csv(out / "efficiency.csv", index=False)
    frame[
        ["sim_sim_idx", "sim_name", "sim_hrss", "id", ranking_par, "detected"]
    ].to_parquet(out / "decisions.parquet", index=False)
    summary = {
        "ranking_par": ranking_par,
        "threshold": threshold,
        "comparison": comparison,
        "n_injected": len(frame),
        "n_recovered": int(frame.id.notna().sum()),
        "n_detected": int(frame.detected.sum()),
        "amplitude_rtol": amplitude_rtol,
        "waveforms": summaries,
        "hrss50_method": "log-amplitude interpolation; no extrapolation",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    if len(curve):
        import matplotlib.pyplot as plt

        ncols = 3
        nrows = (len(summaries) + 2) // 3
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 2.7 * nrows), squeeze=False)
        for ax, (name, group) in zip(axes.flat, curve.groupby("waveform", sort=True)):
            group = group.sort_values("hrss")
            p = group.efficiency.to_numpy()
            ax.errorbar(
                group.hrss,
                p,
                yerr=np.array([p - group.wilson_low, group.wilson_high - p]).clip(0),
                fmt="o-",
                ms=3,
            )
            ax.set_xscale("log")
            ax.set_ylim(-0.04, 1.04)
            ax.set_title(name, fontsize=9)
            ax.set_xlabel("Injected hrss")
            ax.set_ylabel("Efficiency")
            ax.grid(alpha=0.2)
        for ax in axes.flat[len(summaries) :]:
            ax.set_visible(False)
        fig.tight_layout()
        fig.savefig(out / "efficiency.png", dpi=110)
        plt.close(fig)
    return {
        **summary,
        "summary_file": str(out / "summary.json"),
        "curve_file": str(out / "efficiency.csv"),
        "decisions_file": str(out / "decisions.parquet"),
        "plot_file": str(out / "efficiency.png"),
    }
