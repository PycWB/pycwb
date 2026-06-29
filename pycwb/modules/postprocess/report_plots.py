"""Plot helpers for postprocess report artifacts."""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def plot_far_rho(data: dict, out_dir: str) -> None:
    """Plot false-alarm rate vs ranking parameter."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    bins = np.array(data["bins"])
    far = np.array(data["far"])
    rp = data["ranking_par"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.step(bins, far, where="mid", color="steelblue", linewidth=1.5)
    ax.set_xlabel(rp)
    ax.set_ylabel("FAR [yr⁻¹]")
    ax.set_yscale("log")
    ax.set_title(f"False Alarm Rate vs {rp}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "far_rho.png"), dpi=120)
    plt.close(fig)


def plot_n_events(data: dict, out_dir: str) -> None:
    """Plot cumulative event count vs ranking parameter."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    bins = np.array(data["bins"])
    n_events = np.array(data["cum_events"])
    rp = data["ranking_par"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.step(bins, n_events, where="mid", color="crimson", linewidth=1.5)
    ax.set_xlabel(rp)
    ax.set_ylabel("Cumulative events (≥ threshold)")
    ax.set_yscale("log")
    ax.set_title(f"Cumulative Events vs {rp}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "far_rho_n_events.png"), dpi=120)
    plt.close(fig)


def load_public_alert_candidates(
    zero_lag_df: pd.DataFrame,
    public_alerts_file: str,
    gps_time_window: float,
) -> dict[int, str]:
    """Match two-column public alerts to zero-lag triggers by GPS time."""
    if "gps_time" not in zero_lag_df.columns:
        raise KeyError("zero-lag catalog must contain gps_time to mark public alerts")
    if not os.path.exists(public_alerts_file):
        raise FileNotFoundError(f"public_alerts_file not found: {public_alerts_file}")

    trigger_gps = pd.to_numeric(zero_lag_df["gps_time"], errors="coerce").reset_index(drop=True)
    known_candidates: dict[int, str] = {}

    with open(public_alerts_file) as f:
        for line in f:
            fields = line.split()
            if len(fields) < 2 or fields[0].startswith("#"):
                continue
            try:
                alert_gps = float(fields[1])
            except ValueError:
                continue

            time_delta = (trigger_gps - alert_gps).abs()
            if not time_delta.notna().any():
                continue
            trigger_index = int(time_delta.idxmin())
            if time_delta.iloc[trigger_index] <= gps_time_window:
                candidate_id = fields[0]
                if trigger_index in known_candidates:
                    known_candidates[trigger_index] += f", {candidate_id}"
                else:
                    known_candidates[trigger_index] = candidate_id

    logger.info("Public alert candidates matched from %s: %d", public_alerts_file, len(known_candidates))
    return known_candidates


def plot_zero_lag(
    df: pd.DataFrame,
    ranking_par: str,
    livetime_years: float,
    out_dir: str,
    known_candidates: Optional[dict[int, str]] = None,
    output_prefix: str = "zero_lag",
    plot_label: str = "Zero-lag",
    far_rho_data: Optional[dict] = None,
) -> None:
    """Plot zero-lag trigger FAR scatter and significance histogram."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    if len(df) == 0:
        logger.warning("No %s triggers to plot", plot_label)
        return

    rho = df[ranking_par].values
    sig = df["significance"].values if "significance" in df.columns else np.zeros(len(df))
    ifar = df["ifar_years"].values if "ifar_years" in df.columns else np.zeros(len(df))
    far_attached = df["far_attached"].values if "far_attached" in df.columns else np.zeros(len(df))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.scatter(rho, far_attached, c=sig, cmap="viridis", s=30, alpha=0.8, edgecolors="gray", linewidth=0.3)
    ax1.set_xlabel(ranking_par)
    ax1.set_ylabel("FAR [yr⁻¹]")
    ax1.set_yscale("log")
    ax1.set_title(f"{plot_label} triggers (livetime={livetime_years:.1f} yr)")
    ax1.grid(True, alpha=0.3)
    plt.colorbar(ax1.collections[0], ax=ax1, label="significance")

    if far_rho_data is not None:
        bins = np.asarray(far_rho_data["bins"], dtype=float)
        bkg_far = np.asarray(far_rho_data["far"], dtype=float)
        ax1.step(bins, bkg_far, where="mid", color="crimson", linewidth=1.5,
                 label="Background FAR", zorder=10)
        ax1.legend(loc="upper right")

    ax2.hist(sig, bins=30, color="steelblue", edgecolor="white", alpha=0.8)
    ax2.axvline(3, color="crimson", linestyle="--", alpha=0.7, label="3σ")
    ax2.axvline(5, color="darkred", linestyle="--", alpha=0.7, label="5σ")
    ax2.set_xlabel("Significance [-log₁₀(p)]")
    ax2.set_ylabel("Count")
    ax2.set_title("Significance Distribution")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{output_prefix}_report.png"), dpi=120)
    plt.close(fig)

    plot_zero_lag_poisson(
        ifar,
        livetime_years,
        out_dir,
        known_candidates=known_candidates,
        output_prefix=output_prefix,
        plot_label=plot_label,
    )


def plot_zero_lag_poisson(
    ifar: np.ndarray,
    livetime_years: float,
    out_dir: str,
    known_candidates: Optional[dict[int, str]] = None,
    output_prefix: str = "zero_lag",
    plot_label: str = "Zero-lag",
) -> None:
    """Plot zero-lag cumulative events vs IFAR with Poisson confidence bands."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    ifar = np.asarray(ifar, dtype=float)
    ifar_ceiling = max(livetime_years * 1e6, 1e10)
    valid_event_indices = np.where(
        np.isfinite(ifar) & (ifar > 0) & (ifar <= ifar_ceiling)
    )[0]
    if len(valid_event_indices) == 0:
        return

    order = valid_event_indices[np.argsort(-ifar[valid_event_indices])]
    sorted_ifar = ifar[order]

    n_events = len(sorted_ifar)
    foreground_ifar = sorted_ifar[::-1]
    foreground_count = np.arange(n_events, 0, -1, dtype=float)

    ranks = {int(event_index): rank + 1 for rank, event_index in enumerate(order)}
    markers = []
    for event_index, candidate_id in (known_candidates or {}).items():
        if event_index in ranks and np.isfinite(ifar[event_index]) and ifar[event_index] > 0:
            markers.append((float(ifar[event_index]), float(ranks[event_index]), candidate_id))

    ifar_max = foreground_ifar.max()
    ifar_min = max(foreground_ifar.min(), 1e-10)

    n_bins = 800
    ifar_min_bg = max(ifar_min * 0.3, 1e-10)
    ifar_max_bg = ifar_max * 1e4
    ifar_bg = np.logspace(np.log10(ifar_min_bg), np.log10(ifar_max_bg), n_bins)
    mu = livetime_years / ifar_bg

    FAP0 = 1.0 - 0.682689
    FAP1 = 1.0 - 0.954499
    FAP2 = 1.0 - 0.997300
    sigma_labels = ["3σ", "2σ", "1σ"]
    alphas = [0.25, 0.4, 0.6]

    try:
        from scipy.stats import poisson

        percentiles = np.array([FAP2 / 2, FAP1 / 2, FAP0 / 2,
                                1 - FAP0 / 2, 1 - FAP1 / 2, 1 - FAP2 / 2])
        conf = np.array([poisson.ppf(percentiles, m) for m in mu])

        fig, ax = plt.subplots(figsize=(10, 6))

        for k in range(3):
            lower = conf[:, k]
            upper = conf[:, 5 - k]
            ax.fill_between(ifar_bg, lower, upper,
                            color="gray", alpha=alphas[k],
                            label=sigma_labels[k], linewidth=0)

        ax.plot(ifar_bg, mu, color="black", linewidth=0.8, linestyle="--",
                label="Expected BKG")
        ax.step(foreground_ifar, foreground_count, where="pre", color="red", linewidth=1.5,
                label="Foreground")
        if len(markers) > 0:
            marker_ifar = [marker[0] for marker in markers]
            marker_rank = [marker[1] for marker in markers]
            ax.scatter(
                marker_ifar,
                marker_rank,
                s=34,
                color="blue",
                edgecolors="navy",
                linewidths=0.6,
                label="Known candidates",
                zorder=6,
            )
            for marker_ifar, marker_rank, candidate_id in markers:
                ax.annotate(
                    candidate_id,
                    (marker_ifar, marker_rank),
                    xytext=(6, 4),
                    textcoords="offset points",
                    fontsize=8,
                    color="black",
                    zorder=7,
                )

        if n_events > 0:
            loudest_ifar = foreground_ifar[-1]
            loudest_nevt = foreground_count[-1]
            ax.plot(loudest_ifar, loudest_nevt, marker="o", color="red",
                    markersize=6, markeredgecolor="darkred", markeredgewidth=1)

        ax.set_xlabel("IFAR [yr]")
        ax.set_ylabel("Cumulative Number of Events")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(ifar_min_bg, ifar_max_bg)
        y_max = max(mu[0], foreground_count[0])
        ax.set_ylim(0.1, y_max * 1.5)
        ax.set_title(f"cWB {plot_label} Events vs IFAR\nlivetime = {livetime_years:.4f} yr")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3, which="both")

        fig.tight_layout()
        poisson_path = os.path.join(out_dir, f"{output_prefix}_poisson.png")
        fig.savefig(poisson_path, dpi=150)
        plt.close(fig)
        logger.info("%s Poisson plot → %s", plot_label, poisson_path)

    except Exception as exc:
        logger.warning("Poisson confidence intervals failed: %s", exc)

