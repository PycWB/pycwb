"""Matplotlib rendering helpers for efficiency-vs-hrss products.

These functions only render figures from already-computed efficiency data.
They do not read catalogs or score models; the public efficiency actions in
:mod:`pycwb.modules.postprocess.plot_efficiency` own all data preparation and
file-path decisions and call into these helpers to produce the figures.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


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


def _plot_efficiency_by_waveform_panels(
    curves: list[dict],
    prob_threshold: float,
    ifar_label: str,
    ifar_sec: float,
    output_path: str,
    show_sigmoid_fit: bool = False,
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
            hrs = np.array([d["hrss"] for d in c["data"]])
            effs = np.array([d["efficiency"] for d in c["data"]])
            # Binomial standard error: sqrt(p*(1-p)/N), in %
            n_totals = np.array([max(d.get("n_total", 1), 1) for d in c["data"]])
            eff_errs = np.sqrt(effs * (1.0 - effs) / n_totals) * 100.0
            effs_pct = effs * 100.0
            color = freq_colors[c["frequency"]]
            ax.errorbar(hrs, effs_pct, yerr=eff_errs, fmt="o", color=color,
                        markersize=4, capsize=3, label=f"{c['frequency']}Hz",
                        alpha=0.8)
            fit_data = c.get("fit") or {}
            if show_sigmoid_fit and fit_data.get("status") == "ok":
                ax.semilogx(
                    fit_data["fit_x"],
                    np.array(fit_data["fit_y"]) * 100,
                    "-",
                    color=color,
                    linewidth=1.0,
                    alpha=0.45,
                )
                hrss50 = fit_data.get("hrss50")
                if hrss50:
                    ax.axvline(hrss50, color=color, linestyle=":", linewidth=0.8, alpha=0.35)

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
