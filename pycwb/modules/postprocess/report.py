"""Post-production reports — workflow-compatible FAR & zero-lag plots.

Works directly with filtered parquet catalogs (pandas DataFrames) and
progress files, avoiding Catalog-schema dependencies.

Workflow actions
----------------
``postprocess.report.far_rho_plot``
    Compute FAR vs ranking parameter from a BKG catalog and plot.

``postprocess.report.zero_lag_report``
    Compute zero-lag significance and plot triggers with FAR attached.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# far_rho_plot
# ---------------------------------------------------------------------------

def far_rho_plot(
    work_dir: str,
    catalog_file: str,
    progress_file: str,
    job_ids_file: str,
    ranking_par: str = "rho",
    exclude_zero_lag: bool = True,
    bin_size: float = 0.1,
    output_dir: str = "public",
    **kwargs,
) -> dict:
    """Compute FAR vs ranking parameter and save plots.

    Parameters
    ----------
    work_dir : str
        Base directory.
    catalog_file : str
        Path to BKG catalog parquet.
    progress_file : str
        Path to progress parquet.
    job_ids_file : str
        Path to job list file (determines which jobs' live time to use).
    ranking_par : str
        Column name for ranking (default ``"rho"``).
    exclude_zero_lag : bool
        Exclude ``lag_idx == 0`` from live time.
    bin_size : float
        Histogram bin size for ranking parameter.
    output_dir : str
        Directory for output plots (relative to *work_dir*).

    Returns
    -------
    dict
        ``far_rho`` data with keys ``bins``, ``far``, ``n_events``, ``livetime_years``.
    """
    def _resolve(p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(work_dir, p)

    cat_path = _resolve(catalog_file)
    prog_path = _resolve(progress_file)
    jobs_path = _resolve(job_ids_file)
    out_dir = _resolve(output_dir)

    # ── Load triggers ────────────────────────────────────────────────────
    df = pd.read_parquet(cat_path)
    if ranking_par not in df.columns:
        raise KeyError(f"Ranking parameter '{ranking_par}' not in catalog. Columns: {list(df.columns)}")
    values = df[ranking_par].dropna().values
    logger.info("Triggers: %d total, %d with valid %s", len(df), len(values), ranking_par)

    # ── Live time ────────────────────────────────────────────────────────
    with open(jobs_path) as f:
        job_ids = {int(l.strip()) for l in f if l.strip()}
    prog = pd.read_parquet(prog_path)
    prog = prog[prog["job_id"].isin(job_ids)]
    if exclude_zero_lag and "lag_idx" in prog.columns:
        prog = prog[prog["lag_idx"] != 0]
    livetime = float(prog["livetime"].sum())
    livetime_years = livetime / 86400.0 / 365.25
    logger.info("Live time: %.0f s = %.2f yr", livetime, livetime_years)

    # ── Histogram & FAR ──────────────────────────────────────────────────
    vmin, vmax = float(values.min()), float(values.max())
    bins = np.arange(vmin, vmax + bin_size, bin_size)
    hist, _ = np.histogram(values, bins=bins)
    # Cumulative from high to low
    cum_hist = np.cumsum(hist[::-1])[::-1]
    far = cum_hist / max(livetime_years, 1e-10)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    data = {
        "bins": bin_centers.tolist(),
        "far": far.tolist(),
        "n_events": hist.tolist(),
        "cum_events": cum_hist.tolist(),
        "ranking_par": ranking_par,
        "livetime": livetime,
        "livetime_years": livetime_years,
    }

    # ── Plots ────────────────────────────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)
    _plot_far_rho(data, out_dir)
    _plot_n_events(data, out_dir)

    # Save JSON
    json_path = os.path.join(out_dir, "far_rho.json")
    # Convert numpy types for JSON
    json_data = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in data.items()}
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    logger.info("FAR data → %s", json_path)

    return {"far_rho": data}


# ---------------------------------------------------------------------------
# zero_lag_report
# ---------------------------------------------------------------------------

def zero_lag_report(
    work_dir: str,
    catalog_file: str,
    progress_file: str,
    job_ids_file: str,
    far_rho_data: Optional[dict] = None,
    ranking_par: str = "rho",
    output_dir: str = "public",
    **kwargs,
) -> dict:
    """Plot zero-lag triggers with FAR values and Poisson significance.

    Parameters
    ----------
    work_dir : str
        Base directory.
    catalog_file : str
        Path to zero-lag BKG catalog parquet.
    progress_file : str
        Path to progress parquet.
    job_ids_file : str
        Path to job list for zero-lag slice.
    far_rho_data : dict, optional
        FAR data from :func:`far_rho_plot`.  If not provided, reads from
        ``kwargs["far_rho"]`` or ``{output_dir}/far_rho.json``.
    ranking_par : str
        Ranking column name.
    output_dir : str
        Output directory.

    Returns
    -------
    dict
        ``zero_lag`` trigger info with FAR attached.
    """
    def _resolve(p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(work_dir, p)

    cat_path = _resolve(catalog_file)
    prog_path = _resolve(progress_file)
    jobs_path = _resolve(job_ids_file)
    out_dir = _resolve(output_dir)

    # ── Resolve FAR data ─────────────────────────────────────────────────
    if far_rho_data is None:
        far_rho_data = kwargs.get("far_rho")
    # Unwrap if double-wrapped (output_alias stores {far_rho: {...}})
    if isinstance(far_rho_data, dict) and "far_rho" in far_rho_data and "bins" not in far_rho_data:
        far_rho_data = far_rho_data["far_rho"]
    if far_rho_data is None:
        json_path = os.path.join(out_dir, "far_rho.json")
        if os.path.exists(json_path):
            with open(json_path) as f:
                far_rho_data = json.load(f)
    if far_rho_data is None:
        raise ValueError("far_rho_data not provided and far_rho.json not found")

    # ── Load zero-lag triggers ───────────────────────────────────────────
    df = pd.read_parquet(cat_path)
    # Filter to zero-lag only
    if "lag_idx" in df.columns:
        df = df[df["lag_idx"] == 0].reset_index(drop=True)
        logger.info("Zero-lag triggers (lag_idx==0): %d", len(df))
    if ranking_par not in df.columns:
        raise KeyError(f"Ranking parameter '{ranking_par}' not found")
    logger.info("Zero-lag triggers: %d", len(df))

    # ── Live time ────────────────────────────────────────────────────────
    with open(jobs_path) as f:
        job_ids = {int(l.strip()) for l in f if l.strip()}
    prog = pd.read_parquet(prog_path)
    prog = prog[prog["job_id"].isin(job_ids)]
    if "lag_idx" in prog.columns:
        prog = prog[prog["lag_idx"] == 0]  # zero-lag only
    zl_livetime = float(prog["livetime"].sum())
    zl_livetime_years = zl_livetime / 86400.0 / 365.25
    logger.info("Zero-lag live time: %.0f s = %.4f yr", zl_livetime, zl_livetime_years)

    # ── Attach FAR to each trigger ───────────────────────────────────────
    bins = np.array(far_rho_data["bins"])
    far = np.array(far_rho_data["far"])
    # Use zero-lag live time for Poisson expectation, not the far_rho (non-zero-lag) live time

    rho_vals = df[ranking_par].values
    attached_far = np.zeros(len(df))
    for i, r in enumerate(rho_vals):
        idx = np.searchsorted(bins, r, side="right") - 1
        if 0 <= idx < len(far):
            attached_far[i] = far[idx]
        else:
            attached_far[i] = far[-1] if idx >= len(far) else far[0]

    df["far_attached"] = attached_far
    df["ifar_years"] = 1.0 / np.maximum(attached_far, 1e-30)

    # ── Poisson significance ─────────────────────────────────────────────
    # Expected number of BKG events above each rho threshold
    # For each trigger, expected = FAR * zl_livetime (in years)
    expected = attached_far * zl_livetime / 86400 / 365.25
    from scipy.stats import poisson
    p_values = 1.0 - poisson.cdf(0, expected)
    df["p_value"] = p_values
    df["significance"] = -np.log10(np.maximum(p_values, 1e-300))

    # ── Plot ─────────────────────────────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)
    _plot_zero_lag(df, ranking_par, zl_livetime_years, out_dir)

    # Save CSV
    csv_path = os.path.join(out_dir, "zero_lag_triggers.csv")
    cols = [c for c in ["id", "job_id", "lag_idx", ranking_par, "ifar", "far_attached",
                         "ifar_years", "significance", "p_value", "gps_time",
                         "net_cc", "likelihood", "coherent_energy"]
            if c in df.columns]
    df[cols].to_csv(csv_path, index=False)
    logger.info("Zero-lag table → %s", csv_path)

    return {
        "zero_lag_n": len(df),
        "livetime_years": float(zl_livetime_years),
        "max_significance": float(df["significance"].max()) if len(df) > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _plot_far_rho(data: dict, out_dir: str) -> None:
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


def _plot_n_events(data: dict, out_dir: str) -> None:
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


def _plot_zero_lag(df: pd.DataFrame, ranking_par: str, livetime_years: float, out_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    if len(df) == 0:
        logger.warning("No zero-lag triggers to plot")
        return

    rho = df[ranking_par].values
    sig = df["significance"].values if "significance" in df.columns else np.zeros(len(df))
    ifar = df["ifar_years"].values if "ifar_years" in df.columns else np.zeros(len(df))

    # ── Figure 1: IFAR scatter + significance histogram ──────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.scatter(rho, ifar, c=sig, cmap="viridis", s=30, alpha=0.8, edgecolors="gray", linewidth=0.3)
    ax1.set_xlabel(ranking_par)
    ax1.set_ylabel("IFAR [yr]")
    ax1.set_yscale("log")
    ax1.set_title(f"Zero-lag triggers (livetime={livetime_years:.1f} yr)")
    ax1.grid(True, alpha=0.3)
    cbar = plt.colorbar(ax1.collections[0], ax=ax1, label="significance")

    ax2.hist(sig, bins=30, color="steelblue", edgecolor="white", alpha=0.8)
    ax2.axvline(3, color="crimson", linestyle="--", alpha=0.7, label="3σ")
    ax2.axvline(5, color="darkred", linestyle="--", alpha=0.7, label="5σ")
    ax2.set_xlabel("Significance [-log₁₀(p)]")
    ax2.set_ylabel("Count")
    ax2.set_title("Significance Distribution")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "zero_lag_report.png"), dpi=120)
    plt.close(fig)

    # ── Figure 2: Cumulative events vs IFAR with Poisson confidence ──────
    _plot_zero_lag_poisson(df, ifar, livetime_years, out_dir)


def _plot_zero_lag_poisson(
    df: pd.DataFrame,
    ifar: np.ndarray,
    livetime_years: float,
    out_dir: str,
) -> None:
    """Cumulative events vs IFAR with Poisson confidence bands.

    Follows the approach in ``Make_PP_IFAR.C``:
    - Sorted foreground (zero-lag) events plotted as stepwise cumulative count.
    - Expected background = livetime_years / IFAR (a 1/x line in log-log).
    - Poisson confidence belts computed from the expected background using
      continuos Poisson percentiles (FAP0=1σ, FAP1=2σ, FAP2=3σ).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    if len(df) == 0:
        return

    # ── Sort by IFAR descending (most significant first) ─────────────────
    order = np.argsort(-ifar)
    sorted_ifar = ifar[order]

    # Stepwise cumulative count (matching Make_PP_IFAR step style)
    n_events = len(sorted_ifar)
    zl_ifar = np.empty(2 * n_events - 1)
    zl_nevt = np.empty(2 * n_events - 1)
    k = 0
    for i in range(n_events - 1, -1, -1):
        zl_ifar[k] = sorted_ifar[i]
        zl_nevt[k] = i + 1
        k += 1
        if k >= len(zl_nevt) - 1:
            break
        # Add step point
        zl_ifar[k] = zl_ifar[k - 1]
        zl_nevt[k] = zl_nevt[k - 1] - 1
        k += 1

    ifar_max = zl_ifar.max()
    ifar_min = max(zl_ifar.min(), 1e-10)

    # ── Log-spaced IFAR bins for background & belts ──────────────────────
    # Extend far beyond foreground to fill the entire lower-right plot area.
    # Use a fixed upper limit based on decades beyond ifar_max.
    n_bins = 800
    ifar_min_bg = max(ifar_min * 0.3, 1e-10)
    # Extend 4 decades beyond ifar_max to cover full plot
    ifar_max_bg = ifar_max * 1e4
    ifar_bg = np.logspace(np.log10(ifar_min_bg), np.log10(ifar_max_bg), n_bins)
    mu = livetime_years / ifar_bg  # expected background events

    # ── Poisson confidence intervals ─────────────────────────────────────
    # FAP values matching Make_PP_IFAR: 1σ, 2σ, 3σ
    # C++ order: {FAP2/2, FAP1/2, FAP0/2, 1-FAP0/2, 1-FAP1/2, 1-FAP2/2}
    # Returns:  [lower_3σ, lower_2σ, lower_1σ, upper_1σ, upper_2σ, upper_3σ]
    FAP0 = 1.0 - 0.682689  # 1σ
    FAP1 = 1.0 - 0.954499  # 2σ
    FAP2 = 1.0 - 0.997300  # 3σ
    sigma_labels = ["3σ", "2σ", "1σ"]
    alphas = [0.25, 0.4, 0.6]   # 3σ lightest (drawn first), 1σ darkest (drawn last)

    try:
        from scipy.stats import poisson

        # Compute Poisson percentiles directly (avoids continues_poisson CDF bug for small mu)
        percentiles = np.array([FAP2 / 2, FAP1 / 2, FAP0 / 2,
                                1 - FAP0 / 2, 1 - FAP1 / 2, 1 - FAP2 / 2])
        conf = np.array([poisson.ppf(percentiles, m) for m in mu])

        # ── Plot ─────────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 6))

        # Poisson belts: pairs are (conf[:,k], conf[:,5-k]) for k=0,1,2
        # k=0 → (lower_3σ, upper_3σ), k=1 → (lower_2σ, upper_2σ), k=2 → (lower_1σ, upper_1σ)
        for k in range(3):
            lower = conf[:, k]
            upper = conf[:, 5 - k]
            ax.fill_between(ifar_bg, lower, upper,
                            color="gray", alpha=alphas[k],
                            label=sigma_labels[k], linewidth=0)

        # Expected background line
        ax.plot(ifar_bg, mu, color="black", linewidth=0.8, linestyle="--",
                label="Expected BKG")

        # Foreground (zero-lag) step plot
        ax.step(zl_ifar, zl_nevt, where="post", color="red", linewidth=1.5,
                label="Foreground")

        # Loudest event marker
        if n_events > 0:
            loudest_ifar = zl_ifar[-1]
            loudest_nevt = zl_nevt[-1]
            ax.plot(loudest_ifar, loudest_nevt, marker="o", color="red",
                    markersize=6, markeredgecolor="darkred", markeredgewidth=1)

        ax.set_xlabel("IFAR [yr]")
        ax.set_ylabel("Cumulative Number of Events")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(ifar_min_bg, ifar_max_bg)
        ax.set_ylim(0.1, max(mu[0], zl_nevt[0]) * 1.5)
        ax.set_title(f"cWB Zero-lag Events vs IFAR\nlivetime = {livetime_years:.4f} yr")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3, which="both")

        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "zero_lag_poisson.png"), dpi=150)
        plt.close(fig)
        logger.info("Zero-lag Poisson plot → %s", os.path.join(out_dir, "zero_lag_poisson.png"))

    except Exception as e:
        logger.warning("Poisson confidence intervals failed: %s", e)
