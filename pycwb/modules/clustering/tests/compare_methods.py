#!/usr/bin/env python3
"""compare_methods.py — Compare clustering method outputs side by side.

Loads the catalog.parquet produced by each method under runs/<method>/catalog/
and prints a table of key detection statistics, plus optional matplotlib plots.

Usage
-----
::

    # from tests/clustering/
    python compare_methods.py

    # custom method list or paths
    python compare_methods.py --methods baseline weighted_graph --plot
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RUNS_DIR = Path(__file__).parent / "runs"

# Key scalar columns to compare (parquet_column, display_label)
SCALAR_COLS = [
    ("rho",                "rho (primary SNR)"),
    ("rho_alt",            "rho_alt (secondary SNR)"),
    ("likelihood",         "Likelihood"),
    ("coherent_energy",    "Coherent energy"),
    ("packet_norm",        "Norm"),
    ("penalty",            "Penalty"),
    ("net_cc",             "net_cc (network CC)"),
    ("sky_cc",             "sky_cc (sky CC)"),
    ("subnet_cc",          "subnet_cc"),
    ("q_veto",             "q_veto (Qa)"),
    ("q_factor",           "q_factor (Qp)"),
    ("n_pixels_total",     "n_pixels_total (cluster size)"),
    ("gps_time",           "gps_time"),
    ("phi",                "phi (sky, deg)"),
    ("theta",              "theta (sky, deg)"),
    ("ra",                 "RA (deg)"),
    ("dec",                "Dec (deg)"),
]

# GPS matching tolerance in seconds
GPS_TOL = 0.5

# Injection type identification by approximate GPS time
# (from user_parameters_clustering_phase3.yaml)
INJECTION_TYPES = {
    "CBC":  1126259162.4,
    "WNB":  1126259362.4,
    "SGE":  1126259562.4,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_catalog(method: str) -> pd.DataFrame:
    path = RUNS_DIR / method / "catalog" / "catalog.parquet"
    if not path.exists():
        sys.exit(f"ERROR: catalog not found: {path}\n"
                 f"       Run: pycwb run runs/{method}/user_parameters.yaml --force-overwrite")
    df = pd.read_parquet(path)
    df["_method"] = method
    return df


def identify_injection_type(gps: float) -> str:
    best, best_dt = "unknown", float("inf")
    for label, t_inj in INJECTION_TYPES.items():
        dt = abs(gps - t_inj)
        if dt < best_dt:
            best, best_dt = label, dt
    if best_dt > GPS_TOL * 10:
        return "unknown"
    return best


def match_events(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Cross-match all events by GPS time (within GPS_TOL) and return a merged table."""
    methods = list(dfs.keys())
    # collect all unique events from the first method; match others
    ref_key = methods[0]
    ref = dfs[ref_key].copy()
    ref["_inj_type"] = ref["gps_time"].apply(identify_injection_type)

    rows = []
    for _, ev_ref in ref.iterrows():
        row = {"_inj_type": ev_ref["_inj_type"]}
        for col, _ in SCALAR_COLS:
            row[f"{ref_key}__{col}"] = ev_ref.get(col, np.nan)
        for method in methods[1:]:
            cand = dfs[method]
            dt = (cand["gps_time"] - ev_ref["gps_time"]).abs()
            match = cand[dt <= GPS_TOL]
            if len(match) == 0:
                for col, _ in SCALAR_COLS:
                    row[f"{method}__{col}"] = np.nan
            else:
                ev_m = match.iloc[0]
                for col, _ in SCALAR_COLS:
                    row[f"{method}__{col}"] = ev_m.get(col, np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def print_comparison(matched: pd.DataFrame, methods: list[str]) -> None:
    ref = methods[0]
    others = methods[1:]

    for _, row in matched.iterrows():
        inj_type = row["_inj_type"]
        print(f"\n{'='*68}")
        print(f"  Event: {inj_type}   (gps_time ≈ {row[f'{ref}__gps_time']:.2f})")
        print(f"{'='*68}")
        print(f"  {'Statistic':<30}  {'baseline':>12}", end="")
        for m in others:
            print(f"  {m:>18}", end="")
        print(f"  {'delta':>12}")
        print(f"  {'-'*30}  {'-'*12}", end="")
        for _ in others:
            print(f"  {'-'*18}", end="")
        print(f"  {'-'*12}")

        for col, label in SCALAR_COLS:
            if col == "gps_time":
                continue  # already printed in header
            ref_val = row.get(f"{ref}__{col}", np.nan)
            line = f"  {label:<30}  {ref_val:>12.4g}"
            for m in others:
                m_val = row.get(f"{m}__{col}", np.nan)
                line += f"  {m_val:>18.4g}"
            # delta: last method vs ref
            last_val = row.get(f"{others[-1]}__{col}", np.nan)
            if np.isfinite(ref_val) and np.isfinite(last_val) and ref_val != 0:
                delta = (last_val - ref_val) / abs(ref_val) * 100
                line += f"  {delta:>+11.4g}%"
            else:
                line += f"  {'N/A':>12}"
            print(line)


def print_summary(matched: pd.DataFrame, methods: list[str]) -> None:
    ref = methods[0]
    print(f"\n{'='*68}")
    print("  SUMMARY: Mean |relative difference| from baseline")
    print(f"{'='*68}")
    print(f"  {'Statistic':<30}", end="")
    for m in methods[1:]:
        print(f"  {m:>18}", end="")
    print()

    for col, label in SCALAR_COLS:
        if col in ("gps_time", "phi", "theta", "ra", "dec"):
            continue
        ref_vals = np.array([
            row.get(f"{ref}__{col}", np.nan) for _, row in matched.iterrows()
        ], dtype=float)
        line = f"  {label:<30}"
        for m in methods[1:]:
            m_vals = np.array([
                row.get(f"{m}__{col}", np.nan) for _, row in matched.iterrows()
            ], dtype=float)
            mask = np.isfinite(ref_vals) & np.isfinite(m_vals) & (ref_vals != 0)
            if mask.sum() == 0:
                line += f"  {'N/A':>18}"
            else:
                mean_rel = np.mean(np.abs((m_vals[mask] - ref_vals[mask]) / ref_vals[mask])) * 100
                line += f"  {mean_rel:>17.4g}%"
        print(line)


def make_plots(matched: pd.DataFrame, methods: list[str], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    plot_cols = [
        ("rho",             "rho"),
        ("likelihood",      "Likelihood"),
        ("coherent_energy", "Coherent energy"),
        ("net_cc",          "net_cc"),
        ("n_pixels_total",  "n_pixels_total"),
    ]
    ref = methods[0]
    others = methods[1:]
    inj_labels = matched["_inj_type"].tolist()

    x = np.arange(len(matched))
    width = 0.8 / len(methods)
    fig, axes = plt.subplots(len(plot_cols), 1, figsize=(max(8, 3 * len(matched)), 3.5 * len(plot_cols)))
    if len(plot_cols) == 1:
        axes = [axes]

    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]

    for ax, (col, label) in zip(axes, plot_cols):
        for k, m in enumerate(methods):
            vals = [row.get(f"{m}__{col}", np.nan) for _, row in matched.iterrows()]
            bars = ax.bar(x + k * width, vals, width=width,
                          label=m, color=colors[k % len(colors)], alpha=0.8)
        ax.set_ylabel(label)
        ax.set_xticks(x + width * (len(methods) - 1) / 2)
        ax.set_xticklabels(inj_labels)
        ax.legend(loc="upper right")
        ax.set_title(f"{label} by injection type")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"\nPlot saved to: {out_path}")


def make_pdf(matched: pd.DataFrame, methods: list[str], out_path: Path) -> None:
    """Generate a multi-page PDF comparing all clustering methods."""
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.backends.backend_pdf import PdfPages

    ref = methods[0]
    others = methods[1:]
    inj_labels = matched["_inj_type"].tolist()
    n_ev = len(matched)
    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]
    method_colors = {m: colors[i % len(colors)] for i, m in enumerate(methods)}

    # ── metric groups ────────────────────────────────────────────────────────
    snr_cols    = [("rho", "rho"), ("rho_alt", "rho_alt"),
                   ("likelihood", "Likelihood"), ("coherent_energy", "Coherent energy")]
    shape_cols  = [("n_pixels_total", "n_pixels"), ("norm", "Norm"),
                   ("penalty", "Penalty")]
    cc_cols     = [("net_cc", "net_cc"), ("sky_cc", "sky_cc"),
                   ("subnet_cc", "subnet_cc")]
    veto_cols   = [("q_veto", "q_veto (Qa)"), ("q_factor", "q_factor (Qp)")]

    # summary: cols for heatmap (exclude sky/GPS)
    heatmap_cols = [c for c, _ in SCALAR_COLS if c not in ("gps_time", "phi", "theta", "ra", "dec")]

    def _bar_group(ax, col, label, ylabel=None):
        x = np.arange(n_ev)
        width = 0.8 / len(methods)
        for k, m in enumerate(methods):
            vals = [row.get(f"{m}__{col}", np.nan) for _, row in matched.iterrows()]
            ax.bar(x + k * width, vals, width=width, label=m,
                   color=method_colors[m], alpha=0.85)
        ax.set_ylabel(ylabel or label)
        ax.set_xticks(x + width * (len(methods) - 1) / 2)
        ax.set_xticklabels(inj_labels)
        ax.set_title(label)
        ax.legend(fontsize=7, ncol=len(methods))

    def _rel_diff_group(ax, col, label):
        """Relative difference (%) vs baseline, one line per non-baseline method."""
        x = np.arange(n_ev)
        ref_vals = np.array([row.get(f"{ref}__{col}", np.nan) for _, row in matched.iterrows()], float)
        for m in others:
            m_vals = np.array([row.get(f"{m}__{col}", np.nan) for _, row in matched.iterrows()], float)
            with np.errstate(invalid="ignore", divide="ignore"):
                rel = np.where(ref_vals != 0, (m_vals - ref_vals) / np.abs(ref_vals) * 100, np.nan)
            ax.plot(x, rel, "o-", label=m, color=method_colors[m])
        ax.axhline(0, color="gray", lw=0.8, ls="--")
        ax.set_ylabel("Δ from baseline (%)")
        ax.set_xticks(x)
        ax.set_xticklabels(inj_labels)
        ax.set_title(f"{label} — rel. Δ vs {ref}")
        ax.legend(fontsize=7)

    with PdfPages(out_path) as pdf:

        # ── Page 1: title / overview ─────────────────────────────────────────
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.5, 0.65, "Clustering Method Comparison", ha="center",
                 fontsize=26, fontweight="bold")
        fig.text(0.5, 0.58, f"Methods: {', '.join(methods)}", ha="center", fontsize=14)
        fig.text(0.5, 0.52, f"Injections: {', '.join(inj_labels)}", ha="center", fontsize=12)
        fig.text(0.5, 0.46, f"Reference: {ref}", ha="center", fontsize=12, color="gray")
        summary_lines = []
        for col in ("rho", "likelihood", "n_pixels_total"):
            for m in others:
                ref_v = np.array([row.get(f"{ref}__{col}", np.nan) for _, row in matched.iterrows()], float)
                m_v   = np.array([row.get(f"{m}__{col}", np.nan) for _, row in matched.iterrows()], float)
                mask = np.isfinite(ref_v) & np.isfinite(m_v) & (ref_v != 0)
                if mask.sum():
                    mean_rel = np.mean(np.abs((m_v[mask] - ref_v[mask]) / ref_v[mask])) * 100
                    summary_lines.append(f"  {m} vs {ref} | {col}: mean |Δ| = {mean_rel:.2f}%")
        fig.text(0.15, 0.38, "\n".join(summary_lines), fontsize=10, family="monospace",
                 va="top")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ── Page 2: SNR metrics — absolute values ────────────────────────────
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        fig.suptitle("SNR Metrics (absolute)", fontsize=14, fontweight="bold")
        for ax, (col, label) in zip(axes.flat, snr_cols):
            _bar_group(ax, col, label)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ── Page 3: SNR metrics — relative difference ────────────────────────
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        fig.suptitle("SNR Metrics — relative Δ vs baseline (%)", fontsize=14, fontweight="bold")
        for ax, (col, label) in zip(axes.flat, snr_cols):
            _rel_diff_group(ax, col, label)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ── Page 4: Cluster shape & CC metrics ───────────────────────────────
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        fig.suptitle("Cluster Shape & Coherence Metrics", fontsize=14, fontweight="bold")
        for ax, (col, label) in zip(axes[0], shape_cols):
            _bar_group(ax, col, label)
        for ax, (col, label) in zip(axes[1], cc_cols):
            _bar_group(ax, col, label)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ── Page 5: Veto metrics & pixel size relative diff ──────────────────
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle("Veto Metrics & Cluster Size (Δ)", fontsize=14, fontweight="bold")
        for ax, (col, label) in zip(axes[:2], veto_cols):
            _bar_group(ax, col, label)
        _rel_diff_group(axes[2], "n_pixels_total", "n_pixels_total")
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ── Page 6: Sky localization (RA / Dec scatter) ───────────────────────
        fig, axes = plt.subplots(1, n_ev, figsize=(6 * n_ev, 5))
        if n_ev == 1:
            axes = [axes]
        fig.suptitle("Sky Localization (RA, Dec) per injection", fontsize=14, fontweight="bold")
        for ax, (_, row) in zip(axes, matched.iterrows()):
            inj = row["_inj_type"]
            for m in methods:
                ra  = row.get(f"{m}__ra",  np.nan)
                dec = row.get(f"{m}__dec", np.nan)
                ax.scatter([ra], [dec], label=m, color=method_colors[m], s=120,
                           zorder=3, edgecolors="k", linewidths=0.5)
            ax.set_xlabel("RA (deg)")
            ax.set_ylabel("Dec (deg)")
            ax.set_title(f"{inj}")
            ax.legend(fontsize=8)
            ax.grid(True, ls="--", alpha=0.4)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ── Page 7: Summary heatmap — mean |Δ| from baseline ─────────────────
        heat_data = []
        heat_row_labels = []
        for m in others:
            row_vals = []
            for col in heatmap_cols:
                ref_v = np.array([r.get(f"{ref}__{col}", np.nan) for _, r in matched.iterrows()], float)
                m_v   = np.array([r.get(f"{m}__{col}",  np.nan) for _, r in matched.iterrows()], float)
                mask = np.isfinite(ref_v) & np.isfinite(m_v) & (ref_v != 0)
                if mask.sum():
                    row_vals.append(np.mean(np.abs((m_v[mask] - ref_v[mask]) / ref_v[mask])) * 100)
                else:
                    row_vals.append(np.nan)
            heat_data.append(row_vals)
            heat_row_labels.append(m)

        heat_arr = np.array(heat_data, float)
        col_labels = [dict(SCALAR_COLS)[c] for c in heatmap_cols]

        fig, ax = plt.subplots(figsize=(max(12, len(heatmap_cols) * 1.1), max(4, len(others) * 1.0)))
        fig.suptitle("Summary: mean |relative Δ| from baseline (%)", fontsize=13, fontweight="bold")
        # clip colormap at 20% so the rho_alt outlier doesn't wash out everything
        im = ax.imshow(heat_arr, aspect="auto", cmap="YlOrRd", vmin=0, vmax=20)
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=40, ha="right", fontsize=9)
        ax.set_yticks(range(len(heat_row_labels)))
        ax.set_yticklabels(heat_row_labels)
        for i in range(len(heat_row_labels)):
            for j in range(len(col_labels)):
                v = heat_arr[i, j]
                txt = f"{v:.1f}%" if np.isfinite(v) else "N/A"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                        color="black" if v < 10 else "white")
        fig.colorbar(im, ax=ax, label="mean |Δ| (%)", shrink=0.8)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    print(f"\nPDF report saved to: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare clustering method catalogs side by side."
    )
    parser.add_argument(
        "--methods", nargs="+",
        default=["baseline", "weighted_graph", "dbscan", "hdbscan", "optics",
                 "mra_weighted_graph", "mra_hdbscan"],
        help="Method directory names under runs/ (default: all methods)",
    )
    parser.add_argument(
        "--tol", type=float, default=GPS_TOL,
        help=f"GPS matching tolerance in seconds (default: {GPS_TOL})",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Generate comparison bar plots (saved as compare_methods.png)",
    )
    parser.add_argument(
        "--pdf", action="store_true",
        help="Generate a multi-page PDF report (saved as compare_methods.pdf)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global GPS_TOL
    GPS_TOL = args.tol

    print(f"Loading catalogs for methods: {args.methods}")
    dfs = {m: load_catalog(m) for m in args.methods}
    for m, df in dfs.items():
        print(f"  {m}: {len(df)} events")

    matched = match_events(dfs)
    print_comparison(matched, args.methods)
    print_summary(matched, args.methods)

    if args.plot:
        out_plot = Path(__file__).parent / "compare_methods.png"
        try:
            make_plots(matched, args.methods, out_plot)
        except ImportError:
            print("WARNING: matplotlib not available — skipping plots")

    if args.pdf:
        out_pdf = Path(__file__).parent / "compare_methods.pdf"
        try:
            make_pdf(matched, args.methods, out_pdf)
        except ImportError as e:
            print(f"WARNING: {e} — skipping PDF")


if __name__ == "__main__":
    main()
