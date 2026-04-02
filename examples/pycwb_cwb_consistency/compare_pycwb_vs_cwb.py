#!/usr/bin/env python3
"""compare_pycwb_vs_cwb.py — Consistency check between pycWB and cWB run outputs.

Matches triggers from a pycWB Parquet catalog and a cWB ROOT waveburst tree
by GPS time, then produces a PDF report comparing:

  - Core XGBoost features  (rho0, norm, netcc0, penalty, Qa, Qp,
                             sSNR0/likelihood, sSNR1/likelihood, …)
  - Sky-location parameters (phi, theta, ra, dec, sky error area erA[1])
  - Injection-matched fields (if SIM run)

GPS matching tolerance is configurable (default 5 ms).  When injection GPS
times are available in both files the matching is done on injection time first,
then verified with the reconstructed time.

Usage
-----
::

    python compare_pycwb_vs_cwb.py \\
        --parquet  catalog/catalog.M1.parquet \\
        --root     catalog/wave_O4b0_BCK_C00_LH_BurstLF_SIM0_run1.M1.root \\
        --ifo L1 H1 \\
        --tol 0.05 \\
        --out  report_consistency.pdf \\
        --csv  matched_triggers.csv

All arguments have defaults pointing to the test dataset in
``tests/postprod/O4_K21b0_C00_BurstLF_LH_SIM_MDC_short0_test5/``.
"""
from __future__ import annotations

import argparse
import logging
import math
import re
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf_backend
from matplotlib.gridspec import GridSpec
from scipy import stats

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default paths (relative to repo root) — override via CLI
# ---------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parents[2]
_TEST_DIR = _REPO / "tests/postprod/O4_K21b0_C00_BurstLF_LH_SIM_MDC_short0_test5/catalog"
_DEFAULT_PARQUET = str(_TEST_DIR / "catalog.M1.parquet")
_DEFAULT_ROOT    = str(_TEST_DIR / "wave_O4b0_BCK_C00_LH_BurstLF_SIM0_run1.M1.root")
_DEFAULT_OUT     = "report_consistency.pdf"
_DEFAULT_CSV     = "matched_triggers.csv"

# ---------------------------------------------------------------------------
# Branch / column mapping  ROOT → pycWB parquet
# ---------------------------------------------------------------------------
# Each entry: (root_branch, root_index_or_None, parquet_column, label)
# root_index=None means the branch is already a scalar in ROOT.
SCALAR_MAP = [
    # (root_branch, root_idx, parquet_col,          display_label)
    ("likelihood", None, "likelihood",          "Likelihood"),
    ("ecor",       None, "coherent_energy",     "Coherent energy (ecor)"),
    ("norm",       None, "packet_norm",         "Norm"),
    ("penalty",    None, "penalty",             "Penalty"),
    ("rho",        0,    "rho",                 "rho[0]  (primary SNR)"),
    ("rho",        1,    "rho_alt",             "rho[1]  (secondary SNR)"),
    ("netcc",      0,    "net_cc",              "netcc[0]  (network CC)"),
    ("netcc",      1,    "sky_cc",              "netcc[1]  (sky CC)"),
    ("netcc",      2,    "subnet_cc",           "netcc[2]  (subnet CC)"),
    ("Qveto",      0,    "q_veto",              "Qveto[0]  (Qa)"),
    ("Qveto",      1,    "q_factor",            "Qveto[1]  (Qp)"),
    # sky
    ("phi",        0,    "phi",                 "phi[0]  (sky phi, deg)"),
    ("theta",      0,    "theta",               "theta[0]  (sky theta, deg)"),
    ("phi",        2,    "ra",                  "phi[2]  (RA, deg)"),
    ("theta",      2,    "dec",                 "theta[2]  (Dec, deg)"),
    # sky error (erA[1] = 50% credible region in deg²)
    ("erA",        1,    "sky_err_50",          "erA[1]  (50% sky area, deg²)"),
]

# Per-IFO signal energy (sSNR) — built dynamically from ifo_list
# ROOT: sSNR[ifo_idx], parquet: signal_energy_{IFO}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_col(arr, idx, default=np.nan):
    """Extract element *idx* from a jagged/object numpy array column."""
    if arr is None:
        return np.full(1, default)
    out = np.empty(len(arr), dtype=np.float64)
    for i, row in enumerate(arr):
        try:
            v = row[idx]
            out[i] = float(v.item() if hasattr(v, "item") else v)
        except (IndexError, TypeError):
            out[i] = default
    return out


def compute_rho0(ecor, penalty):
    """XGBoost rho0 = sqrt(ecor / (1 + penalty*(max(1,penalty)-1)))."""
    pen = np.asarray(penalty, dtype=np.float64)
    ec  = np.asarray(ecor,    dtype=np.float64)
    denom = 1.0 + pen * (np.maximum(1.0, pen) - 1.0)
    return np.sqrt(np.where(denom > 0, ec / denom, 0.0))


def angular_separation_deg(phi1, theta1, phi2, theta2):
    """Great-circle angular separation in degrees between two cWB sky coords.

    cWB convention: theta = co-latitude [0°,180°], phi = longitude [0°,360°].
    Both inputs in degrees.
    """
    ph1 = np.radians(phi1);  th1 = np.radians(theta1)
    ph2 = np.radians(phi2);  th2 = np.radians(theta2)
    cos_sep = (np.sin(th1) * np.sin(th2) * np.cos(ph1 - ph2)
               + np.cos(th1) * np.cos(th2))
    return np.degrees(np.arccos(np.clip(cos_sep, -1, 1)))


# ---------------------------------------------------------------------------
# Load ROOT → flat DataFrame
# ---------------------------------------------------------------------------

def load_root(root_file: str, ifo_list: list[str], tree_name: str = "waveburst") -> pd.DataFrame:
    """Read the waveburst TTree and return a flat DataFrame."""
    requested = [
        "likelihood", "ecor", "norm", "penalty",
        "rho", "netcc", "Qveto", "Lveto",
        "phi", "theta", "erA",
        "sSNR", "noise",
        "time", "gps",
        "chirp", "duration", "bandwidth", "frequency", "low", "high",
    ]
    log.info("Reading ROOT file: %s", root_file)
    with uproot.open(f"{root_file}:{tree_name}") as tree:
        available = set(tree.keys())
        to_read   = [b for b in requested if b in available]
        missing   = [b for b in requested if b not in available]
        if missing:
            log.warning("ROOT branches not found (skipped): %s", missing)
        n = tree.num_entries
        log.info("  %d events in tree", n)
        raw = tree.arrays(to_read, library="np")

    rows: dict[str, np.ndarray] = {}

    def _scalar(key):
        return raw[key].astype(np.float64) if key in raw else np.full(n, np.nan)

    def _col(key, idx):
        return _safe_col(raw[key], idx) if key in raw else np.full(n, np.nan)

    # GPS time from time[0]  (first IFO trigger time = reference time)
    rows["gps_time_cwb"] = _col("time", 0)

    # Scalars
    rows["likelihood_cwb"] = _scalar("likelihood")
    rows["ecor_cwb"]       = _scalar("ecor")
    rows["norm_cwb"]       = _scalar("norm")
    rows["penalty_cwb"]    = _scalar("penalty")

    # Per-array branches
    for branch, idx, col, _lbl in SCALAR_MAP:
        if branch in raw:
            rows[col + "_cwb"] = _col(branch, idx)
        else:
            rows[col + "_cwb"] = np.full(n, np.nan)

    # Derived: rho0
    rows["rho0_cwb"] = compute_rho0(rows["ecor_cwb"], rows["penalty_cwb"])

    # Lveto[2] (XGBoost feature for blf/bhf/bld)
    if "Lveto" in raw:
        rows["Lveto2_cwb"] = _col("Lveto", 2)
    else:
        rows["Lveto2_cwb"] = np.full(n, np.nan)

    # Per-IFO: sSNR, noise, frequency, bandwidth, duration
    for i, ifo in enumerate(ifo_list):
        rows[f"sSNR_{ifo}_cwb"]       = _col("sSNR",      i) if "sSNR"      in raw else np.full(n, np.nan)
        rows[f"noise_{ifo}_cwb"]      = _col("noise",     i) if "noise"     in raw else np.full(n, np.nan)
        rows[f"freq_{ifo}_cwb"]       = _col("frequency", i) if "frequency" in raw else np.full(n, np.nan)
        rows[f"freq_low_{ifo}_cwb"]   = _col("low",       i) if "low"       in raw else np.full(n, np.nan)
        rows[f"freq_high_{ifo}_cwb"]  = _col("high",      i) if "high"      in raw else np.full(n, np.nan)
        rows[f"bandwidth_{ifo}_cwb"]  = _col("bandwidth", i) if "bandwidth" in raw else np.full(n, np.nan)
        rows[f"duration_{ifo}_cwb"]   = _col("duration",  i) if "duration"  in raw else np.full(n, np.nan)

    # sSNR0/likelihood  (XGBoost feature)
    rows["sSNR0_over_lik_cwb"] = rows[f"sSNR_{ifo_list[0]}_cwb"] / np.where(
        rows["likelihood_cwb"] != 0, rows["likelihood_cwb"], np.nan)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Load Parquet → flat DataFrame
# ---------------------------------------------------------------------------

def load_parquet(parquet_file: str, ifo_list: list[str]) -> pd.DataFrame:
    """Read a pycWB Parquet catalog and return a flat DataFrame."""
    log.info("Reading Parquet file: %s", parquet_file)
    table = pq.read_table(parquet_file)
    df = table.to_pandas()
    log.info("  %d triggers in catalog", len(df))

    nifo = len(ifo_list)
    rows: dict[str, np.ndarray] = {}

    def _get(col):
        return df[col].to_numpy(dtype=np.float64) if col in df.columns else np.full(len(df), np.nan)

    rows["gps_time_pyc"]  = _get("gps_time")
    rows["likelihood_pyc"] = _get("likelihood")
    rows["ecor_pyc"]       = _get("coherent_energy")
    rows["norm_pyc"]       = _get("packet_norm")
    rows["penalty_pyc"]    = _get("penalty")

    # rho
    rows["rho_pyc"]     = _get("rho")
    rows["rho_alt_pyc"] = _get("rho_alt")

    # netcc
    rows["net_cc_pyc"]     = _get("net_cc")
    rows["sky_cc_pyc"]     = _get("sky_cc")
    rows["subnet_cc_pyc"]  = _get("subnet_cc")

    # Qveto
    rows["q_veto_pyc"]   = _get("q_veto")
    rows["q_factor_pyc"] = _get("q_factor")

    # sky
    rows["phi_pyc"]   = _get("phi")
    rows["theta_pyc"] = _get("theta")
    rows["ra_pyc"]    = _get("ra")
    rows["dec_pyc"]   = _get("dec")

    # sky error area — erA is not yet computed by pycWB (all zeros),
    # so mark as NaN to signal "not available" rather than comparing zeros.
    if "sky_error_regions" in df.columns:
        def _era1(x):
            try:
                lst = list(x)
                v = float(lst[1]) if len(lst) > 1 else np.nan
                return np.nan if v == 0.0 else v  # 0 means not computed
            except Exception:
                return np.nan
        rows["sky_err_50_pyc"] = np.array([_era1(v) for v in df["sky_error_regions"]], dtype=np.float64)
    else:
        rows["sky_err_50_pyc"] = np.full(len(df), np.nan)

    # Derived: rho0
    rows["rho0_pyc"] = compute_rho0(rows["ecor_pyc"], rows["penalty_pyc"])

    # Per-IFO sSNR, noise, frequency, bandwidth, duration
    for ifo in ifo_list:
        rows[f"sSNR_{ifo}_pyc"]      = _get(f"signal_energy_{ifo}")
        rows[f"noise_{ifo}_pyc"]     = _get(f"noise_rms_{ifo}")
        rows[f"freq_{ifo}_pyc"]      = _get(f"central_freq_{ifo}")
        rows[f"freq_low_{ifo}_pyc"]  = _get(f"freq_low_{ifo}")
        rows[f"freq_high_{ifo}_pyc"] = _get(f"freq_high_{ifo}")
        rows[f"bandwidth_{ifo}_pyc"] = _get(f"bandwidth_{ifo}")
        rows[f"duration_{ifo}_pyc"]  = _get(f"duration_{ifo}")

    # sSNR0/likelihood
    rows["sSNR0_over_lik_pyc"] = rows[f"sSNR_{ifo_list[0]}_pyc"] / np.where(
        rows["likelihood_pyc"] != 0, rows["likelihood_pyc"], np.nan)

    # Injection GPS time (for better matching in SIM runs)
    if "injection" in df.columns:
        try:
            inj = pd.json_normalize(df["injection"].dropna())
            if "gps_time" in inj.columns:
                inj_t = np.full(len(df), np.nan)
                inj_t[df["injection"].notna()] = inj["gps_time"].to_numpy(dtype=np.float64)
                rows["inj_gps_time_pyc"] = inj_t
            else:
                rows["inj_gps_time_pyc"] = np.full(len(df), np.nan)
        except Exception:
            rows["inj_gps_time_pyc"] = np.full(len(df), np.nan)
    else:
        rows["inj_gps_time_pyc"] = np.full(len(df), np.nan)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# GPS-time matching
# ---------------------------------------------------------------------------

def match_triggers(df_cwb: pd.DataFrame, df_pyc: pd.DataFrame,
                   tol: float = 0.05
                   ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Match events by reconstructed GPS time within *tol* seconds.

    Returns
    -------
    merged : pd.DataFrame
        Matched pairs (cWB columns + pycWB columns side-by-side).
    unmatched_cwb : pd.DataFrame
        cWB triggers that had no pycWB counterpart within *tol*.
    unmatched_pyc : pd.DataFrame
        pycWB triggers that had no cWB counterpart within *tol*.
    """
    log.info("Matching triggers: %d cWB  vs  %d pycWB  (tol=%.3f s)",
             len(df_cwb), len(df_pyc), tol)

    t_cwb = df_cwb["gps_time_cwb"].to_numpy()
    t_pyc = df_pyc["gps_time_pyc"].to_numpy()

    # Sort both by GPS time for efficient nearest-neighbour lookup
    sort_cwb = np.argsort(t_cwb)
    sort_pyc = np.argsort(t_pyc)
    ts_cwb   = t_cwb[sort_cwb]
    ts_pyc   = t_pyc[sort_pyc]

    matched_cwb: list[int] = []
    matched_pyc: list[int] = []
    used_pyc: set[int] = set()

    for orig_i, i in enumerate(sort_cwb):
        t = ts_cwb[orig_i]
        # binary search for closest pycwb time
        pos = np.searchsorted(ts_pyc, t)
        best_j = None
        best_dt = np.inf
        for j in [pos - 1, pos]:
            if 0 <= j < len(ts_pyc):
                dt = abs(ts_pyc[j] - t)
                orig_j = sort_pyc[j]
                if dt < best_dt and orig_j not in used_pyc:
                    best_dt = dt
                    best_j  = orig_j
        if best_j is not None and best_dt <= tol:
            matched_cwb.append(i)
            matched_pyc.append(best_j)
            used_pyc.add(best_j)

    all_cwb = set(range(len(df_cwb)))
    all_pyc = set(range(len(df_pyc)))
    unm_cwb = sorted(all_cwb - set(matched_cwb))
    unm_pyc = sorted(all_pyc - set(matched_pyc))

    log.info("  Matched: %d  |  cWB-only: %d  |  pycWB-only: %d",
             len(matched_cwb), len(unm_cwb), len(unm_pyc))

    left  = df_cwb.iloc[matched_cwb].reset_index(drop=True)
    right = df_pyc.iloc[matched_pyc].reset_index(drop=True)
    merged = pd.concat([left, right], axis=1)
    return merged, df_cwb.iloc[unm_cwb].reset_index(drop=True), df_pyc.iloc[unm_pyc].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _scatter_panel(ax, x, y, label_x, label_y, title, color="steelblue"):
    """1:1 scatter with Pearson r and RMS annotations."""
    mask = np.isfinite(x) & np.isfinite(y)
    xm, ym = x[mask], y[mask]
    if len(xm) < 2:
        ax.text(0.5, 0.5, "insufficient data", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title(title)
        return
    ax.scatter(xm, ym, s=6, alpha=0.4, color=color, rasterized=True)
    vmin = min(xm.min(), ym.min()); vmax = max(xm.max(), ym.max())
    ax.plot([vmin, vmax], [vmin, vmax], "k--", lw=0.8, label="1:1")
    r, pval = stats.pearsonr(xm, ym)
    rms = np.sqrt(np.mean((xm - ym) ** 2))
    ax.set_xlabel(label_x, fontsize=8)
    ax.set_ylabel(label_y, fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.text(0.03, 0.97,
            f"N={len(xm)}\nr={r:.4f}\nRMS={rms:.4g}",
            ha="left", va="top", transform=ax.transAxes, fontsize=7,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    ax.legend(fontsize=7)


def _residual_panel(ax, x, y, label, color="steelblue"):
    """Histogram of (pycWB − cWB) residuals."""
    diff = y - x
    mask = np.isfinite(diff)
    dm = diff[mask]
    if len(dm) < 2:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes)
        return
    ax.hist(dm, bins=50, color=color, alpha=0.7, edgecolor="none")
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.axvline(np.mean(dm), color="red", lw=1.0, ls="-", label=f"mean={np.mean(dm):.3g}")
    ax.set_xlabel(f"pycWB − cWB  [{label}]", fontsize=8)
    ax.set_ylabel("Counts", fontsize=8)
    ax.set_title(f"Residuals: {label}", fontsize=9)
    ax.legend(fontsize=7)


def _distribution_pages(pdf_pages, unmatched_cwb: pd.DataFrame,
                         unmatched_pyc: pd.DataFrame,
                         matched_cwb_vals: dict[str, np.ndarray],
                         matched_pyc_vals: dict[str, np.ndarray],
                         ifo_list: list[str]) -> None:
    """Append pages showing parameter distributions for matched vs. unmatched triggers.

    Each page covers one parameter group.  For each feature three overlaid
    normalised histograms are plotted:

    * **matched cWB**   (green, dashed)   — reference sample
    * **unmatched cWB** (red, solid)      — cWB triggers absent in pycWB
    * **unmatched pycWB** (blue, solid)   — pycWB triggers absent in cWB
    """

    def _get_cwb(df, branch, idx=None, default=np.nan):
        """Extract a column from the cWB unmatched DataFrame."""
        col = (branch if idx is None else branch) + "_cwb"
        if col in df.columns:
            return df[col].to_numpy(dtype=np.float64)
        return np.full(len(df), default)

    def _get_pyc(df, col_suffix, default=np.nan):
        col = col_suffix + "_pyc"
        if col in df.columns:
            return df[col].to_numpy(dtype=np.float64)
        return np.full(len(df), default)

    # ── parameter specs: (label, cwb_col_suffix, pyc_col_suffix) ─────────────
    # cwb_col_suffix / pyc_col_suffix refer to the keys used in the flat DFs
    DIST_PARAMS = [
        ("rho0  (XGB)",           "rho0",            "rho0"),
        ("rho[0]",                "rho",              "rho"),
        ("Likelihood",            "likelihood",       "likelihood"),
        ("Coherent energy",       "ecor",             "ecor"),
        ("Norm",                  "norm",             "norm"),
        ("Penalty",               "penalty",          "penalty"),
        ("netcc[0]",              "net_cc",           "net_cc"),
        ("netcc[1]  sky CC",      "sky_cc",           "sky_cc"),
        ("Qa  (Qveto[0])",        "q_veto",           "q_veto"),
        ("Qp  (Qveto[1])",        "q_factor",         "q_factor"),
        ("phi[0]  (deg)",         "phi",              "phi"),
        ("theta[0]  (deg)",       "theta",            "theta"),
        ("RA  (deg)",             "ra",               "ra"),
        ("Dec  (deg)",            "dec",              "dec"),
        ("sSNR[0]/likelihood",    "sSNR0_over_lik",   "sSNR0_over_lik"),
    ]
    for i, ifo in enumerate(ifo_list):
        DIST_PARAMS.append((f"sSNR[{i}]  ({ifo})",
                            f"sSNR_{ifo}", f"sSNR_{ifo}"))
    # frequency / bandwidth / duration per IFO
    for i, ifo in enumerate(ifo_list):
        DIST_PARAMS.extend([
            (f"Freq[{i}] ({ifo}, Hz)",       f"freq_{ifo}",      f"freq_{ifo}"),
            (f"Freq low[{i}] ({ifo}, Hz)",   f"freq_low_{ifo}",   f"freq_low_{ifo}"),
            (f"Freq high[{i}] ({ifo}, Hz)",  f"freq_high_{ifo}",  f"freq_high_{ifo}"),
            (f"Bandwidth[{i}] ({ifo}, Hz)",  f"bandwidth_{ifo}",  f"bandwidth_{ifo}"),
            (f"Duration[{i}] ({ifo}, s)",    f"duration_{ifo}",   f"duration_{ifo}"),
        ])

    ncols = 3
    nrows = 3  # 9 panels per page
    params_per_page = ncols * nrows

    pages = [DIST_PARAMS[i:i + params_per_page]
             for i in range(0, len(DIST_PARAMS), params_per_page)]

    colors = {
        "matched_cwb":   ("#2ca02c", "--", 0.55, "matched cWB"),
        "unmatched_cwb": ("#d62728", "-",  0.65, "unmatched cWB"),
        "unmatched_pyc": ("#1f77b4", "-",  0.65, "unmatched pycWB"),
    }

    for page_idx, page_params in enumerate(pages):
        fig, axes = plt.subplots(nrows, ncols,
                                  figsize=(5 * ncols, 4 * nrows),
                                  squeeze=False)
        fig.suptitle(
            f"Unmatched-trigger parameter distributions  (page {page_idx + 1}/{len(pages)})",
            fontsize=13,
        )

        for slot, (label, cwb_suf, pyc_suf) in enumerate(page_params):
            r, c = divmod(slot, ncols)
            ax = axes[r, c]

            # data arrays
            mc  = matched_cwb_vals.get(cwb_suf + "_cwb",
                  np.full(1, np.nan))   # matched cWB
            uc  = _get_cwb(unmatched_cwb, cwb_suf)  # unmatched cWB
            up  = _get_pyc(unmatched_pyc, pyc_suf)  # unmatched pycWB

            all_vals = np.concatenate([
                mc[np.isfinite(mc)],
                uc[np.isfinite(uc)],
                up[np.isfinite(up)],
            ])
            if len(all_vals) < 2:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes)
                ax.set_title(label, fontsize=9)
                continue

            lo, hi = np.nanpercentile(all_vals, 0.5), np.nanpercentile(all_vals, 99.5)
            if lo == hi:
                lo, hi = all_vals.min() - 0.5, all_vals.max() + 0.5
            bins = np.linspace(lo, hi, 51)

            for arr, (color, ls, alpha, handle_label) in zip(
                [mc, uc, up], colors.values()
            ):
                finite = arr[np.isfinite(arr)]
                finite = finite[(finite >= lo) & (finite <= hi)]
                if len(finite) == 0:
                    continue
                counts, edges = np.histogram(finite, bins=bins)
                norm = counts / counts.sum() if counts.sum() > 0 else counts
                centres = 0.5 * (edges[:-1] + edges[1:])
                ax.step(centres, norm, where="mid",
                        color=color, ls=ls, alpha=alpha,
                        label=f"{handle_label} (N={len(finite)})")

            ax.set_xlabel(label, fontsize=8)
            ax.set_ylabel("Fraction", fontsize=8)
            ax.set_title(label, fontsize=9)
            ax.legend(fontsize=6, loc="upper right")

        # hide unused slots
        for slot in range(len(page_params), params_per_page):
            r, c = divmod(slot, ncols)
            axes[r, c].set_visible(False)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        pdf_pages.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    # ── 2-D scatter: rho0 vs netcc[0] coloured by class ──────────────────────
    fig, axes2d = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("rho0 vs netcc[0]  —  Matched / Unmatched", fontsize=12)

    for ax2, (vals_rho0, vals_netcc, title) in zip(
        axes2d,
        [
            # cWB perspective
            (
                {
                    "matched":   matched_cwb_vals.get("rho0_cwb", np.array([])),
                    "unmatched": unmatched_cwb["rho0_cwb"].to_numpy(dtype=np.float64)
                                 if "rho0_cwb" in unmatched_cwb.columns
                                 else np.array([]),
                },
                {
                    "matched":   matched_cwb_vals.get("net_cc_cwb", np.array([])),
                    "unmatched": unmatched_cwb["net_cc_cwb"].to_numpy(dtype=np.float64)
                                 if "net_cc_cwb" in unmatched_cwb.columns
                                 else np.array([]),
                },
                "cWB  (green=matched, red=unmatched)",
            ),
            # pycWB perspective
            (
                {
                    "matched":   matched_pyc_vals.get("rho0_pyc", np.array([])),
                    "unmatched": unmatched_pyc["rho0_pyc"].to_numpy(dtype=np.float64)
                                 if "rho0_pyc" in unmatched_pyc.columns
                                 else np.array([]),
                },
                {
                    "matched":   matched_pyc_vals.get("net_cc_pyc", np.array([])),
                    "unmatched": unmatched_pyc["net_cc_pyc"].to_numpy(dtype=np.float64)
                                 if "net_cc_pyc" in unmatched_pyc.columns
                                 else np.array([]),
                },
                "pycWB  (green=matched, blue=unmatched)",
            ),
        ],
    ):
        rho_m  = vals_rho0["matched"];   cc_m  = vals_netcc["matched"]
        rho_u  = vals_rho0["unmatched"]; cc_u  = vals_netcc["unmatched"]
        mask_m = np.isfinite(rho_m) & np.isfinite(cc_m)
        mask_u = np.isfinite(rho_u) & np.isfinite(cc_u)
        if mask_m.any():
            ax2.scatter(rho_m[mask_m], cc_m[mask_m], s=5, alpha=0.3,
                        color="#2ca02c", label=f"matched (N={mask_m.sum()})",
                        rasterized=True)
        if mask_u.any():
            uc_color = "#d62728" if "cWB" in title else "#1f77b4"
            ax2.scatter(rho_u[mask_u], cc_u[mask_u], s=8, alpha=0.6,
                        color=uc_color,
                        label=f"unmatched (N={mask_u.sum()})",
                        rasterized=True)
        ax2.set_xlabel("rho0", fontsize=9)
        ax2.set_ylabel("netcc[0]", fontsize=9)
        ax2.set_title(title, fontsize=9)
        ax2.legend(fontsize=7)

    plt.tight_layout()
    pdf_pages.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _stats_row(x, y):
    diff = y - x
    mask = np.isfinite(diff)
    dm = diff[mask]
    if len(dm) < 2:
        return {"N": len(dm), "mean_diff": np.nan, "std_diff": np.nan,
                "rms_diff": np.nan, "max_abs_diff": np.nan, "pearson_r": np.nan}
    r = stats.pearsonr(x[mask], y[mask])[0] if len(dm) > 1 else np.nan
    return {
        "N":            len(dm),
        "mean_diff":    float(np.mean(dm)),
        "std_diff":     float(np.std(dm)),
        "rms_diff":     float(np.sqrt(np.mean(dm**2))),
        "max_abs_diff": float(np.max(np.abs(dm))),
        "pearson_r":    float(r),
    }


# ---------------------------------------------------------------------------
# Reference-events matching  (--ref_events)
# ---------------------------------------------------------------------------

# Column names for headerless CSVs with a known 9-column layout
_REF_COL_NAMES = [
    "gps_start_ref", "gps_end_ref", "hrss_ref", "t_central_ref",
    "freq_low_ref", "freq_high_ref", "amplitude_ref", "col7_ref", "col8_ref",
]


def load_ref_events(ref_csv: str) -> pd.DataFrame:
    """Load reference events CSV.  First column is GPS start, second is GPS end.

    Auto-detects whether a header row is present.  If the file has no header
    (first field parses as a float) the columns are named using
    ``_REF_COL_NAMES`` for the first 9 columns; any extras get ``colN_ref``.
    The resulting DataFrame always has ``gps_start_ref`` and ``gps_end_ref``
    as its first two columns.
    """
    with open(ref_csv) as fh:
        first = fh.readline().split(",")[0].strip()
    has_header = True
    try:
        float(first)
        has_header = False
    except ValueError:
        pass

    df = pd.read_csv(ref_csv, header=0 if has_header else None)
    cols = list(df.columns)
    if not has_header:
        rename = {}
        for i, c in enumerate(cols):
            rename[c] = _REF_COL_NAMES[i] if i < len(_REF_COL_NAMES) else f"col{i}_ref"
        df = df.rename(columns=rename)
    else:
        # Ensure first two columns are named gps_start_ref / gps_end_ref
        df = df.rename(columns={cols[0]: "gps_start_ref"})
        if len(cols) > 1:
            df = df.rename(columns={cols[1]: "gps_end_ref"})
    log.info("Reference events loaded: %d rows from %s", len(df), ref_csv)
    return df


def match_ref_to_pyc(df_ref: pd.DataFrame, df_pyc: pd.DataFrame, tolerance: float = 0.0) -> pd.DataFrame:
    """Match reference events to pycWB triggers by time-range containment.

    A pycWB trigger is considered **found** for a reference event when its
    GPS time falls within the reference event's window (±tolerance):
        gps_start_ref - tolerance <= gps_time_pyc <= gps_end_ref + tolerance

    If multiple pycWB triggers fall inside the same reference window the one
    closest to the window centre is chosen.

    Parameters
    ----------
    tolerance : float
        Extra time (seconds) added on both sides of the reference window.

    Adds columns to a copy of *df_ref*:
    - ``ref_status``    : ``"found"`` or ``"missing_in_pycwb"``
    - All ``*_pyc`` columns of the best-matching pycWB trigger (NaN if missing)
    """
    t_start = df_ref["gps_start_ref"].to_numpy(dtype=np.float64)
    t_end   = df_ref["gps_end_ref"].to_numpy(dtype=np.float64)
    t_pyc   = df_pyc["gps_time_pyc"].to_numpy(dtype=np.float64)

    matched_pyc_idx: list[int | None] = [None] * len(t_start)
    used_pyc: set[int] = set()

    # Process reference events sorted by window size (smallest first) so that
    # tightly-constrained injections claim their trigger before wider windows,
    # then fall back to tolerance for remaining unmatched events.
    order = np.argsort(t_end - t_start)           # smallest window first
    for i in order:
        t0, t1 = t_start[i], t_end[i]
        centre = 0.5 * (t0 + t1)
        candidates = [
            (abs(t_pyc[j] - centre), j)
            for j in range(len(t_pyc))
            if t0 - tolerance <= t_pyc[j] <= t1 + tolerance and j not in used_pyc
        ]
        if candidates:
            _, best_j = min(candidates)
            matched_pyc_idx[i] = best_j
            used_pyc.add(best_j)

    df_out = df_ref.copy()
    df_out["ref_status"] = [
        "found" if idx is not None else "missing_in_pycwb"
        for idx in matched_pyc_idx
    ]

    # Attach matched pycwb columns (NaN for unmatched rows)
    pyc_cols = [c for c in df_pyc.columns]
    for col in pyc_cols:
        vals = np.full(len(df_ref), np.nan)
        for i, idx in enumerate(matched_pyc_idx):
            if idx is not None:
                v = df_pyc.iloc[idx][col]
                try:
                    vals[i] = float(v)
                except (TypeError, ValueError):
                    pass
        df_out[col] = vals

    n_found   = sum(1 for x in matched_pyc_idx if x is not None)
    n_missing = len(matched_pyc_idx) - n_found
    log.info("Reference events: found=%d  missing_in_pycwb=%d", n_found, n_missing)
    return df_out


def _ref_events_pages(pdf_pages, df_ref: pd.DataFrame,
                      ifo_list: list[str]) -> None:
    """Append PDF pages summarising found vs missing reference events."""
    if "ref_status" not in df_ref.columns:
        return

    status_col = df_ref["ref_status"]
    counts     = status_col.value_counts()
    COLORS = {"found": "#2ca02c", "missing_in_pycwb": "#d62728"}

    # ── Page A: bar chart ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(counts.index,
                  counts.values,
                  color=[COLORS.get(s, "gray") for s in counts.index])
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(val), ha="center", va="bottom", fontsize=11)
    ax.set_xlabel("Status", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("Reference events — pycWB detection status", fontsize=11)
    total = len(df_ref)
    n_found = counts.get("found", 0)
    ax.text(0.98, 0.97,
            f"Total: {total}\nFound: {n_found} ({100*n_found/max(total,1):.1f} %)",
            transform=ax.transAxes, ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round", fc="#f5f5f5", ec="gray"))
    plt.tight_layout()
    pdf_pages.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # ── Page B: parameter distributions from ref CSV (found vs missing) ────────
    # Collect numeric ref-side columns (exclude matched pycwb columns)
    ref_numeric = [
        c for c in df_ref.columns
        if c.endswith("_ref") and pd.api.types.is_numeric_dtype(df_ref[c])
    ]
    # Also include matched pycwb trigger parameters for found events
    pyc_show = [
        ("gps_time_pyc",      "Matched GPS time (pycWB)"),
        ("likelihood_pyc",    "Likelihood (pycWB)"),
        ("rho0_pyc",          "rho0 (pycWB)"),
        ("net_cc_pyc",        "netcc[0] (pycWB)"),
        ("q_veto_pyc",        "Qa (pycWB)"),
        ("penalty_pyc",       "Penalty (pycWB)"),
    ]
    for ifo in ifo_list:
        pyc_show.append((f"sSNR_{ifo}_pyc", f"sSNR {ifo} (pycWB)"))

    all_statuses = sorted(counts.index)

    # ── ref-column distributions ──────────────────────────────────────────────
    if ref_numeric:
        ncols = 4
        nrows = math.ceil(len(ref_numeric) / ncols)
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(5 * ncols, 4 * nrows), squeeze=False)
        fig.suptitle("Reference-event parameters: found vs missing", fontsize=12)
        for slot, col in enumerate(ref_numeric):
            r, c = divmod(slot, ncols)
            ax = axes[r][c]
            all_vals = df_ref[col].to_numpy(dtype=np.float64)
            finite = all_vals[np.isfinite(all_vals)]
            if len(finite) < 2:
                ax.set_title(col, fontsize=9)
                continue
            lo = np.nanpercentile(finite, 0.5)
            hi = np.nanpercentile(finite, 99.5)
            if lo == hi:
                lo, hi = finite.min() - 0.5, finite.max() + 0.5
            bins = np.linspace(lo, hi, 40)
            for st in all_statuses:
                mask = (status_col == st).to_numpy()
                vals = df_ref.loc[mask, col].to_numpy(dtype=np.float64)
                vals = vals[np.isfinite(vals)]
                vals = vals[(vals >= lo) & (vals <= hi)]
                if len(vals) == 0:
                    continue
                ch, edges = np.histogram(vals, bins=bins)
                nh = ch / ch.sum() if ch.sum() > 0 else ch
                centres = 0.5 * (edges[:-1] + edges[1:])
                ax.step(centres, nh, where="mid",
                        color=COLORS.get(st, "gray"), alpha=0.75,
                        label=f"{st} (N={len(vals)})")
            ax.set_xlabel(col, fontsize=8)
            ax.set_ylabel("Fraction", fontsize=8)
            ax.set_title(col, fontsize=9)
            ax.legend(fontsize=6, loc="upper right")
        for slot in range(len(ref_numeric), nrows * ncols):
            r, c = divmod(slot, ncols)
            axes[r][c].set_visible(False)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        pdf_pages.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    # ── matched pycwb-column distributions (found entries only) vs missing ────
    avail_pyc = [(col, lbl) for col, lbl in pyc_show if col in df_ref.columns]
    if avail_pyc:
        ncols = 4
        nrows = math.ceil(len(avail_pyc) / ncols)
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(5 * ncols, 4 * nrows), squeeze=False)
        fig.suptitle("Matched pycWB parameters for found reference events", fontsize=12)
        found_mask = (status_col == "found").to_numpy()
        for slot, (col, lbl) in enumerate(avail_pyc):
            r, c = divmod(slot, ncols)
            ax = axes[r][c]
            vals = df_ref.loc[found_mask, col].to_numpy(dtype=np.float64)
            finite = vals[np.isfinite(vals)]
            if len(finite) < 2:
                ax.set_title(lbl, fontsize=9); continue
            ax.hist(finite, bins=40, color=COLORS["found"], alpha=0.75, edgecolor="none")
            ax.set_xlabel(lbl, fontsize=8)
            ax.set_ylabel("Count", fontsize=8)
            ax.set_title(lbl, fontsize=9)
        for slot in range(len(avail_pyc), nrows * ncols):
            r, c = divmod(slot, ncols)
            axes[r][c].set_visible(False)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        pdf_pages.savefig(fig, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Log-based classification of unmatched cWB triggers
# ---------------------------------------------------------------------------

_CLUSTER_LINE_RE = re.compile(
    r"likelihood\s+(accepted|rejected)\s+cluster\s+\d+.*?"
    r"pixels,\s+from\s+([\d.]+)\s+-\s+([\d.]+)\s+s"
)
_ANALYZE_WIN_RE = re.compile(r"Analyze window:\s+\[([0-9.]+),\s*([0-9.]+)\]")
_PADDED_WIN_RE  = re.compile(r"Padded window:\s+\[([0-9.]+),\s*([0-9.]+)\]")


def _parse_log_windows(log_file: str) -> tuple[float, float, float] | None:
    """Return (analyze_start, analyze_end, padded_start) from first ~50 lines."""
    a_start = a_end = p_start = None
    with open(log_file) as f:
        for i, line in enumerate(f):
            if i > 50:
                break
            m = _ANALYZE_WIN_RE.search(line)
            if m:
                a_start, a_end = float(m.group(1)), float(m.group(2))
            m = _PADDED_WIN_RE.search(line)
            if m:
                p_start = float(m.group(1))
            if a_start is not None and p_start is not None:
                break
    if a_start is None or p_start is None:
        return None
    return a_start, a_end, p_start


def _load_log_cluster_info(log_path: str) -> list[dict]:
    """Load analyze/padded windows and cluster lines from log file(s)."""
    p = Path(log_path)
    log_files = sorted(p.glob("job_*.log")) if p.is_dir() else [p] if p.is_file() else []
    if not log_files:
        log.warning("No log files found at: %s", log_path)
        return []
    infos = []
    for lf in log_files:
        windows = _parse_log_windows(str(lf))
        if windows is None:
            log.warning("Could not parse windows from %s", lf)
            continue
        a_start, a_end, p_start = windows
        clusters = []
        with open(lf) as f:
            for line in f:
                m = _CLUSTER_LINE_RE.search(line)
                if m:
                    clusters.append({
                        "outcome": m.group(1),
                        "t_start": float(m.group(2)),
                        "t_end":   float(m.group(3)),
                    })
        infos.append({
            "analyze_start": a_start,
            "analyze_end":   a_end,
            "padded_start":  p_start,
            "clusters":      clusters,
            "file":          str(lf),
        })
        log.info("Log %s: window [%.1f, %.1f], %d clusters",
                 lf.name, a_start, a_end, len(clusters))
    return infos


def classify_unmatched_cwb_from_logs(unmatched_cwb: pd.DataFrame,
                                      log_path: str) -> pd.DataFrame:
    """Add 'pycwb_status' column to unmatched_cwb.

    Status values
    -------------
    rejected_likelihood : GPS time falls within a pycWB likelihood-rejected cluster.
    not_found           : Inside the analyze window but no matching cluster in log.
    outside_window      : GPS time outside all log analyze windows.
    """
    infos = _load_log_cluster_info(log_path)
    if not infos:
        df = unmatched_cwb.copy()
        df["pycwb_status"] = "log_not_found"
        return df

    statuses = []
    for _, row in unmatched_cwb.iterrows():
        gps = row["gps_time_cwb"]
        status = "outside_window"
        for info in infos:
            if not (info["analyze_start"] <= gps <= info["analyze_end"]):
                continue
            rel = gps - info["padded_start"]
            t_int = round(rel)
            status = "not_found"
            for cl in info["clusters"]:
                if cl["outcome"] != "rejected":
                    continue
                # Quick pre-filter on integer part of cluster start time
                if abs(int(cl["t_start"]) - t_int) > 1:
                    continue
                if cl["t_start"] <= rel <= cl["t_end"]:
                    status = "rejected_likelihood"
                    break
            break  # matched to a log file
        statuses.append(status)

    df = unmatched_cwb.copy()
    df["pycwb_status"] = statuses
    counts = pd.Series(statuses).value_counts().to_dict()
    log.info("Unmatched cWB classification: %s", counts)
    return df


def _pycwb_status_pages(pdf_pages, unmatched_cwb: pd.DataFrame,
                        ifo_list: list[str]) -> None:
    """Append classification summary pages for unmatched cWB triggers."""
    if "pycwb_status" not in unmatched_cwb.columns:
        return

    status_col = unmatched_cwb["pycwb_status"]
    counts = status_col.value_counts()

    # ── Page 1: bar chart summary ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(counts.index, counts.values,
                  color=["#d62728", "#ff7f0e", "#aec7e8"][:len(counts)])
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(val), ha="center", va="bottom", fontsize=10)
    ax.set_xlabel("pycWB status", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("Unmatched cWB triggers — pycWB status classification", fontsize=11)
    plt.tight_layout()
    pdf_pages.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # ── Page 2: parameter distributions per status ────────────────────────
    STATUS_COLORS = {
        "rejected_likelihood": "#d62728",
        "not_found":           "#ff7f0e",
        "outside_window":      "#aec7e8",
        "log_not_found":       "#9467bd",
    }
    DIST_COLS = [
        ("rho0_cwb",       "rho0"),
        ("rho_cwb",        "rho[0]"),
        ("likelihood_cwb", "Likelihood"),
        ("net_cc_cwb",     "netcc[0]"),
        ("q_veto_cwb",     "Qa"),
        ("penalty_cwb",    "Penalty"),
    ]
    for ifo in ifo_list:
        DIST_COLS.append((f"sSNR_{ifo}_cwb", f"sSNR ({ifo})"))

    ncols, nrows = 4, math.ceil(len(DIST_COLS) / 4)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 4 * nrows), squeeze=False)
    fig.suptitle("Unmatched cWB — parameter distributions by pycWB status",
                 fontsize=12)

    all_statuses = sorted(counts.index)
    for slot, (col, label) in enumerate(DIST_COLS):
        r, c = divmod(slot, ncols)
        ax = axes[r][c]
        if col not in unmatched_cwb.columns:
            ax.set_visible(False)
            continue
        all_vals = unmatched_cwb[col].to_numpy(dtype=np.float64)
        finite = all_vals[np.isfinite(all_vals)]
        if len(finite) < 2:
            ax.set_title(label, fontsize=9)
            continue
        lo = np.nanpercentile(finite, 0.5)
        hi = np.nanpercentile(finite, 99.5)
        if lo == hi:
            lo, hi = finite.min() - 0.5, finite.max() + 0.5
        bins = np.linspace(lo, hi, 40)
        for st in all_statuses:
            mask = (status_col == st).to_numpy()
            vals = unmatched_cwb.loc[mask, col].to_numpy(dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            vals = vals[(vals >= lo) & (vals <= hi)]
            if len(vals) == 0:
                continue
            counts_h, edges = np.histogram(vals, bins=bins)
            norm_h = counts_h / counts_h.sum() if counts_h.sum() > 0 else counts_h
            centres = 0.5 * (edges[:-1] + edges[1:])
            color = STATUS_COLORS.get(st, "gray")
            ax.step(centres, norm_h, where="mid", color=color, alpha=0.75,
                    label=f"{st} (N={len(vals)})")
        ax.set_xlabel(label, fontsize=8)
        ax.set_ylabel("Fraction", fontsize=8)
        ax.set_title(label, fontsize=9)
        ax.legend(fontsize=6, loc="upper right")

    for slot in range(len(DIST_COLS), nrows * ncols):
        r, c = divmod(slot, ncols)
        axes[r][c].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    pdf_pages.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # ── Page 3: rho0 vs netcc[0] scatter coloured by status ──────────────
    if "rho0_cwb" in unmatched_cwb.columns and "net_cc_cwb" in unmatched_cwb.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        for st in all_statuses:
            mask = (status_col == st).to_numpy()
            x = unmatched_cwb.loc[mask, "rho0_cwb"].to_numpy(dtype=np.float64)
            y = unmatched_cwb.loc[mask, "net_cc_cwb"].to_numpy(dtype=np.float64)
            m = np.isfinite(x) & np.isfinite(y)
            if m.any():
                ax.scatter(x[m], y[m], s=10, alpha=0.5,
                           color=STATUS_COLORS.get(st, "gray"),
                           label=f"{st} (N={m.sum()})", rasterized=True)
        ax.set_xlabel("rho0", fontsize=10)
        ax.set_ylabel("netcc[0]", fontsize=10)
        ax.set_title("Unmatched cWB: rho0 vs netcc[0] by pycWB status", fontsize=11)
        ax.legend(fontsize=8)
        plt.tight_layout()
        pdf_pages.savefig(fig, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Live ROOT file — cWB job windows and livetime
# ---------------------------------------------------------------------------

def load_live_root(live_file: str, n_ifo: int = 2) -> dict:
    """Read the ``liveTime`` tree from a cWB ``live_*.root`` file.

    Returns
    -------
    dict with keys:
      ``cwb_zero_lag_livetime`` : float — sum of ``live`` for zero-lag entries
      ``cwb_total_livetime``    : float — sum of ``live`` for all entries
      ``n_jobs``                : int   — number of zero-lag entries
      ``job_windows``           : list of ``(start_gps, stop_gps)`` from IFO[0]
      ``df``                    : pd.DataFrame — full raw table
    """
    log.info("Reading live ROOT file: %s", live_file)
    with uproot.open(f"{live_file}:liveTime") as tree:
        n = tree.num_entries
        log.info("  liveTime entries: %d", n)
        raw = tree.arrays(["run", "gps", "live", "lag", "slag", "start", "stop"],
                          library="np")

    live  = raw["live"].astype(np.float64)
    lag   = raw["lag"]
    start = raw["start"]
    stop  = raw["stop"]

    # Zero-lag entries: lag[i][0] == 0
    zl_mask = np.array([float(lag[i][0]) == 0.0 for i in range(n)], dtype=bool)

    cwb_zero_lag_livetime = float(live[zl_mask].sum())
    cwb_total_livetime    = float(live.sum())

    # Analysis windows from zero-lag entries using IFO[0] start/stop
    job_windows: list[tuple[float, float]] = []
    for i in np.where(zl_mask)[0]:
        t0 = float(start[i][0])
        t1 = float(stop[i][0])
        if t0 > 0 and t1 > t0:
            job_windows.append((t0, t1))

    log.info("  Zero-lag livetime: %.1f s  (%d job windows)",
             cwb_zero_lag_livetime, len(job_windows))

    rows: dict[str, np.ndarray] = {
        "run":   raw["run"].astype(int),
        "gps":   raw["gps"].astype(np.float64),
        "live":  live,
        "lag0":  np.array([float(lag[i][0])   for i in range(n)], dtype=np.float64),
        "start0": np.array([float(start[i][0]) for i in range(n)], dtype=np.float64),
        "stop0":  np.array([float(stop[i][0])  for i in range(n)], dtype=np.float64),
    }
    for k in range(1, n_ifo):
        rows[f"lag{k}"]   = np.array([float(lag[i][k])   if len(lag[i])   > k else np.nan for i in range(n)], dtype=np.float64)
        rows[f"start{k}"] = np.array([float(start[i][k]) if len(start[i]) > k else np.nan for i in range(n)], dtype=np.float64)
        rows[f"stop{k}"]  = np.array([float(stop[i][k])  if len(stop[i])  > k else np.nan for i in range(n)], dtype=np.float64)

    return {
        "cwb_zero_lag_livetime": cwb_zero_lag_livetime,
        "cwb_total_livetime":    cwb_total_livetime,
        "n_jobs":                int(zl_mask.sum()),
        "job_windows":           job_windows,
        "df":                    pd.DataFrame(rows),
    }


def load_pyc_progress(progress_file: str, catalog_file: str) -> dict:
    """Read pycWB livetime and job analysis windows from a progress Parquet file.

    Parameters
    ----------
    progress_file : str
        Path to ``progress*.parquet`` (columns: job_id, trial_idx, lag_idx,
        n_triggers, livetime, timestamp, status).
    catalog_file : str
        Path to the catalog Parquet file whose Arrow metadata contains the
        ``jobs`` JSON array (each entry has ``index``, ``analyze_start``,
        ``analyze_end``).

    Returns
    -------
    dict with keys:
      ``pyc_zero_lag_livetime`` : float — sum of livetime for lag_idx==0, status=="completed"
      ``pyc_total_livetime``    : float — sum of livetime for all completed rows
      ``n_jobs``                : int   — number of unique zero-lag completed jobs
      ``job_windows``           : list of (analyze_start, analyze_end) GPS tuples
      ``df``                    : pd.DataFrame — full progress table
    """
    log.info("Reading pycWB progress file: %s", progress_file)
    df = pq.read_table(progress_file).to_pandas()
    log.info("  %d progress rows", len(df))

    completed = df[df["status"] == "completed"]
    zl_mask   = (completed["lag_idx"] == 0)
    pyc_zero_lag_livetime = float(completed.loc[zl_mask, "livetime"].sum())
    pyc_total_livetime    = float(completed["livetime"].sum())
    n_jobs = int(zl_mask.sum())
    log.info("  Zero-lag livetime: %.1f s  (%d jobs)", pyc_zero_lag_livetime, n_jobs)

    # Extract job windows from catalog Arrow metadata
    job_windows: list[tuple[float, float]] = []
    try:
        meta = pq.read_metadata(catalog_file)
        kv   = {k.decode(): v.decode() for k, v in meta.metadata.items()
                if k not in (b"pandas", b"ARROW:schema")}
        if "jobs" in kv:
            import json as _json
            jobs = _json.loads(kv["jobs"])
            # Only include zero-lag jobs present in the progress file
            completed_job_ids = set(completed.loc[zl_mask, "job_id"].tolist())
            for job in jobs:
                jid = job.get("index") or job.get("job_id")
                t0  = job.get("analyze_start")
                t1  = job.get("analyze_end")
                if jid in completed_job_ids and t0 is not None and t1 is not None and t1 > t0:
                    job_windows.append((float(t0), float(t1)))
        log.info("  Extracted %d job windows from catalog metadata", len(job_windows))
    except Exception as exc:
        log.warning("Could not extract job windows from catalog metadata: %s", exc)

    return {
        "pyc_zero_lag_livetime": pyc_zero_lag_livetime,
        "pyc_total_livetime":    pyc_total_livetime,
        "n_jobs":                n_jobs,
        "job_windows":           job_windows,
        "df":                    df,
    }


def classify_unmatched_pyc_by_cwb_time(
        unmatched_pyc: pd.DataFrame,
        job_windows: list[tuple[float, float]]) -> pd.DataFrame:
    """Add ``cwb_window_status`` to unmatched pycWB triggers.

    Status values
    -------------
    ``inside_cwb_window``  : GPS time falls within a cWB job analysis window.
    ``outside_cwb_window`` : GPS time is outside all cWB job windows.
    """
    if not job_windows or "gps_time_pyc" not in unmatched_pyc.columns:
        df = unmatched_pyc.copy()
        df["cwb_window_status"] = "unknown"
        return df

    t = unmatched_pyc["gps_time_pyc"].to_numpy(dtype=np.float64)
    wins   = np.array(sorted(job_windows), dtype=np.float64)
    starts = wins[:, 0]
    stops  = wins[:, 1]

    statuses = []
    for ti in t:
        pos = np.searchsorted(starts, ti, side="right") - 1
        if pos >= 0 and starts[pos] <= ti <= stops[pos]:
            statuses.append("inside_cwb_window")
        else:
            statuses.append("outside_cwb_window")

    df = unmatched_pyc.copy()
    df["cwb_window_status"] = statuses
    counts = pd.Series(statuses).value_counts().to_dict()
    log.info("Unmatched pycWB window classification: %s", counts)
    return df


def classify_unmatched_cwb_by_pyc_time(
        unmatched_cwb: pd.DataFrame,
        job_windows: list[tuple[float, float]]) -> pd.DataFrame:
    """Add ``pyc_window_status`` to unmatched cWB triggers.

    Status values
    -------------
    ``inside_pyc_window``  : GPS time falls within a pycWB job analysis window.
    ``outside_pyc_window`` : GPS time is outside all pycWB job windows.
    """
    if not job_windows or "gps_time_cwb" not in unmatched_cwb.columns:
        df = unmatched_cwb.copy()
        df["pyc_window_status"] = "unknown"
        return df

    t = unmatched_cwb["gps_time_cwb"].to_numpy(dtype=np.float64)
    wins   = np.array(sorted(job_windows), dtype=np.float64)
    starts = wins[:, 0]
    stops  = wins[:, 1]

    statuses = []
    for ti in t:
        pos = np.searchsorted(starts, ti, side="right") - 1
        if pos >= 0 and starts[pos] <= ti <= stops[pos]:
            statuses.append("inside_pyc_window")
        else:
            statuses.append("outside_pyc_window")

    df = unmatched_cwb.copy()
    df["pyc_window_status"] = statuses
    counts = pd.Series(statuses).value_counts().to_dict()
    log.info("Unmatched cWB pycWB-window classification: %s", counts)
    return df


def _cwb_pyc_window_pages(pdf_pages,
                          unmatched_cwb: pd.DataFrame,
                          ifo_list: list[str]) -> None:
    """Append pages for unmatched cWB triggers classified by pycWB job windows.

    If all unmatched cWB triggers are inside pycWB windows (i.e. no livetime
    loss), a single informational page is shown instead of distribution plots.
    """
    if "pyc_window_status" not in unmatched_cwb.columns:
        return

    status_col = unmatched_cwb["pyc_window_status"]
    counts     = status_col.value_counts()
    n_outside  = counts.get("outside_pyc_window", 0)
    n_inside   = counts.get("inside_pyc_window",  0)
    n_total    = len(unmatched_cwb)

    STATUS_COLORS = {
        "inside_pyc_window":  "#ff7f0e",
        "outside_pyc_window": "#9467bd",
        "unknown":            "#aec7e8",
    }

    if n_total == 0 or n_outside == 0:
        # ── No livetime loss: single summary page ─────────────────────────────
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.axis("off")
        msg = (
            "No pycWB livetime loss from cWB perspective\n"
            "════════════════════════════════════════════\n\n"
            f"All {n_total} unmatched cWB trigger(s) fall inside pycWB analysis windows.\n"
            "No cWB events are outside the pycWB job time range."
        )
        ax.text(0.5, 0.5, msg, ha="center", va="center",
                transform=ax.transAxes, fontsize=13, fontfamily="monospace",
                bbox=dict(boxstyle="round", fc="#e8f5e9", ec="#2ca02c", lw=2))
        ax.set_title("Unmatched cWB — pycWB window check", fontsize=12)
        pdf_pages.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        return

    # ── Page 1: bar chart ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Unmatched cWB Triggers — pycWB Window Analysis", fontsize=13)

    ax = axes[0]
    bars = ax.bar(counts.index, counts.values,
                  color=[STATUS_COLORS.get(s, "gray") for s in counts.index])
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(val), ha="center", va="bottom", fontsize=11)
    ax.set_xlabel("Status", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("cWB-only triggers: inside vs outside pycWB windows", fontsize=10)
    ax.text(0.97, 0.97,
            f"Total unmatched cWB   : {n_total}\n"
            f"Inside  pycWB window  : {n_inside}  ({100*n_inside /max(n_total,1):.1f}%)\n"
            f"Outside pycWB window  : {n_outside} ({100*n_outside/max(n_total,1):.1f}%)",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="#f5f5f5", ec="gray"))

    ax2 = axes[1]
    for st, color in STATUS_COLORS.items():
        mask = (status_col == st).to_numpy()
        if not mask.any():
            continue
        gps = unmatched_cwb.loc[mask, "gps_time_cwb"].to_numpy(dtype=np.float64)
        rho = (unmatched_cwb.loc[mask, "rho0_cwb"].to_numpy(dtype=np.float64)
               if "rho0_cwb" in unmatched_cwb.columns else np.ones(mask.sum()))
        ax2.scatter(gps, rho, s=8, alpha=0.6, color=color,
                    label=f"{st} (N={mask.sum()})", rasterized=True)
    ax2.set_xlabel("GPS time (s)", fontsize=9)
    ax2.set_ylabel("rho0", fontsize=9)
    ax2.set_title("Unmatched cWB: GPS time vs rho0", fontsize=10)
    ax2.legend(fontsize=7)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf_pages.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # ── Page 2: parameter distributions by pycWB window status ───────────────
    all_statuses = sorted(counts.index)
    DIST_COLS = [
        ("rho0_cwb",       "rho0"),
        ("rho_cwb",        "rho[0]"),
        ("likelihood_cwb", "Likelihood"),
        ("net_cc_cwb",     "netcc[0]"),
        ("q_veto_cwb",     "Qa"),
        ("penalty_cwb",    "Penalty"),
    ]
    for ifo in ifo_list:
        DIST_COLS.append((f"sSNR_{ifo}_cwb", f"sSNR ({ifo})"))

    ncols = 4
    nrows = math.ceil(len(DIST_COLS) / ncols)
    fig, axes_d = plt.subplots(nrows, ncols,
                               figsize=(5 * ncols, 4 * nrows), squeeze=False)
    fig.suptitle("Unmatched cWB — distributions by pycWB window status", fontsize=12)

    for slot, (col, label) in enumerate(DIST_COLS):
        r, c = divmod(slot, ncols)
        ax = axes_d[r][c]
        if col not in unmatched_cwb.columns:
            ax.set_visible(False)
            continue
        all_vals = unmatched_cwb[col].to_numpy(dtype=np.float64)
        finite   = all_vals[np.isfinite(all_vals)]
        if len(finite) < 2:
            ax.set_title(label, fontsize=9)
            continue
        lo = np.nanpercentile(finite, 0.5)
        hi = np.nanpercentile(finite, 99.5)
        if lo == hi:
            lo, hi = finite.min() - 0.5, finite.max() + 0.5
        bins = np.linspace(lo, hi, 40)
        for st in all_statuses:
            mask = (status_col == st).to_numpy()
            vals = unmatched_cwb.loc[mask, col].to_numpy(dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            vals = vals[(vals >= lo) & (vals <= hi)]
            if len(vals) == 0:
                continue
            ch, edges = np.histogram(vals, bins=bins)
            nh = ch / ch.sum() if ch.sum() > 0 else ch
            centres = 0.5 * (edges[:-1] + edges[1:])
            ax.step(centres, nh, where="mid",
                    color=STATUS_COLORS.get(st, "gray"), alpha=0.75,
                    label=f"{st} (N={len(vals)})")
        ax.set_xlabel(label, fontsize=8)
        ax.set_ylabel("Fraction", fontsize=8)
        ax.set_title(label, fontsize=9)
        ax.legend(fontsize=6, loc="upper right")

    for slot in range(len(DIST_COLS), nrows * ncols):
        r, c = divmod(slot, ncols)
        axes_d[r][c].set_visible(False)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    pdf_pages.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _livetime_comparison_pages(pdf_pages,
                                live_data: dict,
                                pyc_progress: dict | None,
                                unmatched_pyc: pd.DataFrame,
                                ifo_list: list[str]) -> None:
    """Append livetime comparison and pycWB unmatched-time pages to the PDF."""
    cwb_zl      = live_data["cwb_zero_lag_livetime"]
    cwb_tot     = live_data["cwb_total_livetime"]
    n_cwb_jobs  = live_data["n_jobs"]
    cwb_windows = live_data["job_windows"]

    pyc_zl      = pyc_progress["pyc_zero_lag_livetime"] if pyc_progress else None
    pyc_tot     = pyc_progress["pyc_total_livetime"]    if pyc_progress else None
    n_pyc_jobs  = pyc_progress["n_jobs"]                if pyc_progress else None
    pyc_windows = pyc_progress["job_windows"]           if pyc_progress else []

    # ── Page 1: livetime bar chart + job-window timelines ────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Livetime Comparison: cWB vs pycWB", fontsize=13)

    ax = axes[0]
    bar_labels = ["cWB zero-lag"]
    bar_days   = [cwb_zl / 86400]
    bar_cols   = ["#1f77b4"]
    bar_secs   = [cwb_zl]
    if pyc_zl is not None:
        bar_labels.append("pycWB zero-lag")
        bar_days.append(pyc_zl / 86400)
        bar_cols.append("#2ca02c")
        bar_secs.append(pyc_zl)
    bars = ax.bar(bar_labels, bar_days, color=bar_cols, width=0.5)
    for bar, sec in zip(bars, bar_secs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{sec:.1f} s\n({sec / 86400:.4f} d)",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Livetime (days)", fontsize=10)
    ax.set_title("Zero-lag Livetime", fontsize=11)
    extra = f"cWB total (all lags): {cwb_tot:.1f} s\ncWB jobs: {n_cwb_jobs}"
    if pyc_zl is not None:
        extra += f"\npycWB total: {pyc_tot:.1f} s\npycWB jobs: {n_pyc_jobs}"
        ratio = pyc_zl / cwb_zl if cwb_zl > 0 else np.nan
        extra += f"\npycWB / cWB ratio: {ratio:.4f}"
    ax.text(0.97, 0.97, extra, transform=ax.transAxes, ha="right", va="top",
            fontsize=9, bbox=dict(boxstyle="round", fc="#f5f5f5", ec="gray"))

    # cWB job-window timeline
    ax2 = axes[1]
    if cwb_windows:
        t_ref = sorted(cwb_windows)[0][0]
        for idx, (t0, t1) in enumerate(sorted(cwb_windows)):
            ax2.barh(idx, t1 - t0, left=t0 - t_ref, height=0.7,
                     color="#1f77b4", alpha=0.6)
        ax2.set_xlabel(f"Time offset from GPS {t_ref:.0f} (s)", fontsize=9)
        ax2.set_ylabel("Job index", fontsize=9)
        ax2.set_title(f"cWB Analysis Windows ({len(cwb_windows)} jobs)", fontsize=10)
    else:
        ax2.text(0.5, 0.5, "No cWB job windows",
                 ha="center", va="center", transform=ax2.transAxes)
        t_ref = 0.0

    # pycWB job-window timeline
    ax3 = axes[2]
    if pyc_windows:
        pyc_t_ref = sorted(pyc_windows)[0][0]
        for idx, (t0, t1) in enumerate(sorted(pyc_windows)):
            ax3.barh(idx, t1 - t0, left=t0 - pyc_t_ref, height=0.7,
                     color="#2ca02c", alpha=0.6)
        ax3.set_xlabel(f"Time offset from GPS {pyc_t_ref:.0f} (s)", fontsize=9)
        ax3.set_ylabel("Job index", fontsize=9)
        ax3.set_title(f"pycWB Analysis Windows ({len(pyc_windows)} jobs)", fontsize=10)
    else:
        ax3.text(0.5, 0.5, "No pycWB job windows",
                 ha="center", va="center", transform=ax3.transAxes)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf_pages.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # ── Page 2: pycWB unmatched — inside vs outside cWB window ───────────────
    if "cwb_window_status" not in unmatched_pyc.columns:
        return

    status_col = unmatched_pyc["cwb_window_status"]
    counts     = status_col.value_counts()
    STATUS_COLORS = {
        "inside_cwb_window":  "#ff7f0e",
        "outside_cwb_window": "#9467bd",
        "unknown":            "#aec7e8",
    }
    n_inside  = counts.get("inside_cwb_window",  0)
    n_outside = counts.get("outside_cwb_window", 0)
    n_total   = len(unmatched_pyc)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Unmatched pycWB Triggers — cWB Window Analysis", fontsize=13)

    ax = axes[0]
    bars = ax.bar(counts.index, counts.values,
                  color=[STATUS_COLORS.get(s, "gray") for s in counts.index])
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(val), ha="center", va="bottom", fontsize=11)
    ax.set_xlabel("Status", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("pycWB-only triggers: inside vs outside cWB windows", fontsize=10)
    ax.text(0.97, 0.97,
            f"Total unmatched pycWB : {n_total}\n"
            f"Inside  cWB window    : {n_inside}  ({100*n_inside /max(n_total,1):.1f}%)\n"
            f"Outside cWB window    : {n_outside} ({100*n_outside/max(n_total,1):.1f}%)",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="#f5f5f5", ec="gray"))

    ax2 = axes[1]
    if cwb_windows:
        t_ref = sorted(cwb_windows)[0][0]
        for t0, t1 in sorted(cwb_windows):
            ax2.axvspan(t0 - t_ref, t1 - t_ref, alpha=0.12, color="#1f77b4")
    else:
        t_ref = 0.0
    for st, color in STATUS_COLORS.items():
        mask = (status_col == st).to_numpy()
        if not mask.any():
            continue
        gps = unmatched_pyc.loc[mask, "gps_time_pyc"].to_numpy(dtype=np.float64)
        gps_plot = gps - t_ref if cwb_windows else gps
        rho = (unmatched_pyc.loc[mask, "rho0_pyc"].to_numpy(dtype=np.float64)
               if "rho0_pyc" in unmatched_pyc.columns else np.ones(mask.sum()))
        ax2.scatter(gps_plot, rho, s=8, alpha=0.6, color=color,
                    label=f"{st} (N={mask.sum()})", rasterized=True)
    ax2.set_xlabel(
        f"Time offset from GPS {t_ref:.0f} (s)" if cwb_windows else "GPS time (s)",
        fontsize=9)
    ax2.set_ylabel("rho0", fontsize=9)
    ax2.set_title("Unmatched pycWB: GPS offset vs rho0", fontsize=10)
    ax2.legend(fontsize=7)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf_pages.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # ── Page 3: parameter distributions by window status ─────────────────────
    all_statuses = sorted(counts.index)
    DIST_COLS = [
        ("rho0_pyc",       "rho0"),
        ("rho_pyc",        "rho[0]"),
        ("likelihood_pyc", "Likelihood"),
        ("net_cc_pyc",     "netcc[0]"),
        ("q_veto_pyc",     "Qa"),
        ("penalty_pyc",    "Penalty"),
    ]
    for ifo in ifo_list:
        DIST_COLS.append((f"sSNR_{ifo}_pyc", f"sSNR ({ifo})"))

    ncols = 4
    nrows = math.ceil(len(DIST_COLS) / ncols)
    fig, axes_d = plt.subplots(nrows, ncols,
                               figsize=(5 * ncols, 4 * nrows), squeeze=False)
    fig.suptitle("Unmatched pycWB — distributions by cWB window status", fontsize=12)

    for slot, (col, label) in enumerate(DIST_COLS):
        r, c = divmod(slot, ncols)
        ax = axes_d[r][c]
        if col not in unmatched_pyc.columns:
            ax.set_visible(False)
            continue
        all_vals = unmatched_pyc[col].to_numpy(dtype=np.float64)
        finite   = all_vals[np.isfinite(all_vals)]
        if len(finite) < 2:
            ax.set_title(label, fontsize=9)
            continue
        lo = np.nanpercentile(finite, 0.5)
        hi = np.nanpercentile(finite, 99.5)
        if lo == hi:
            lo, hi = finite.min() - 0.5, finite.max() + 0.5
        bins = np.linspace(lo, hi, 40)
        for st in all_statuses:
            mask = (status_col == st).to_numpy()
            vals = unmatched_pyc.loc[mask, col].to_numpy(dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            vals = vals[(vals >= lo) & (vals <= hi)]
            if len(vals) == 0:
                continue
            ch, edges = np.histogram(vals, bins=bins)
            nh = ch / ch.sum() if ch.sum() > 0 else ch
            centres = 0.5 * (edges[:-1] + edges[1:])
            ax.step(centres, nh, where="mid",
                    color=STATUS_COLORS.get(st, "gray"), alpha=0.75,
                    label=f"{st} (N={len(vals)})")
        ax.set_xlabel(label, fontsize=8)
        ax.set_ylabel("Fraction", fontsize=8)
        ax.set_title(label, fontsize=9)
        ax.legend(fontsize=6, loc="upper right")

    for slot in range(len(DIST_COLS), nrows * ncols):
        r, c = divmod(slot, ncols)
        axes_d[r][c].set_visible(False)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    pdf_pages.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

FEATURE_GROUPS = {
    "Core reconstructed": [
        ("likelihood",    "likelihood_cwb",   "likelihood_pyc",  "Likelihood"),
        ("ecor",          "ecor_cwb",         "ecor_pyc",        "Coherent energy"),
        ("rho0",          "rho0_cwb",         "rho0_pyc",        "rho0 (XGB)"),
        ("rho",           "rho_cwb",          "rho_pyc",         "rho[0]"),
        ("norm",          "norm_cwb",         "norm_pyc",        "Norm"),
        ("penalty",       "penalty_cwb",       "penalty_pyc",    "Penalty"),
    ],
    "Network quality": [
        ("netcc0",   "net_cc_cwb",    "net_cc_pyc",    "netcc[0]"),
        ("netcc1",   "sky_cc_cwb",    "sky_cc_pyc",    "netcc[1]  (sky CC)"),
        ("netcc2",   "subnet_cc_cwb", "subnet_cc_pyc", "netcc[2]  (subnet CC)"),
        ("Qa",       "q_veto_cwb",    "q_veto_pyc",    "Qa  (Qveto[0])"),
        ("Qp",       "q_factor_cwb",  "q_factor_pyc",  "Qp  (Qveto[1])"),
    ],
    "Sky location": [
        ("phi",      "phi_cwb",         "phi_pyc",        "phi[0]  (deg)"),
        ("theta",    "theta_cwb",       "theta_pyc",      "theta[0]  (deg)"),
        ("ra",       "ra_cwb",          "ra_pyc",         "RA  phi[2] (deg)"),
        ("dec",      "dec_cwb",         "dec_pyc",        "Dec  theta[2] (deg)"),
        ("sky_area", "sky_err_50_cwb",  "sky_err_50_pyc", "50% sky area (deg²)"),
    ],
    "XGBoost features": [
        ("sSNR0_lik", "sSNR0_over_lik_cwb", "sSNR0_over_lik_pyc", "sSNR[0]/likelihood"),
    ],
    # Frequency group is populated dynamically per IFO in build_report
    "Frequency": [],
}


def build_report(merged: pd.DataFrame,
                 unmatched_cwb: pd.DataFrame,
                 unmatched_pyc: pd.DataFrame,
                 ifo_list: list[str],
                 out_pdf: str, out_csv: str,
                 ref_df: pd.DataFrame | None = None,
                 live_data: dict | None = None,
                 pyc_progress: dict | None = None) -> tuple:
    """Generate the PDF report and statistics CSV, return the stats table.

    If *unmatched_cwb* contains a ``pycwb_status`` column (added by
    :func:`classify_unmatched_cwb_from_logs`), extra classification pages
    are appended to the PDF.

    If *ref_df* is provided (output of :func:`match_ref_to_pyc`), reference-
    event found/missing pages are also appended.

    If *live_data* is provided (output of :func:`load_live_root`), livetime
    comparison pages and pycWB unmatched-trigger window classification pages
    are appended to the PDF.  *pyc_progress* (output of
    :func:`load_pyc_progress`) is shown alongside for comparison.
    """

    # Add per-IFO sSNR features dynamically
    for i, ifo in enumerate(ifo_list):
        key  = f"sSNR_{ifo}"
        cwb  = f"sSNR_{ifo}_cwb"
        pyc  = f"sSNR_{ifo}_pyc"
        lbl  = f"sSNR[{i}]  ({ifo})"
        FEATURE_GROUPS["XGBoost features"].append((key, cwb, pyc, lbl))

    # Add per-IFO frequency/bandwidth/duration features dynamically
    for i, ifo in enumerate(ifo_list):
        FEATURE_GROUPS["Frequency"].extend([
            (f"freq_{ifo}",      f"freq_{ifo}_cwb",      f"freq_{ifo}_pyc",      f"Freq[{i}]  ({ifo}, Hz)"),
            (f"freq_low_{ifo}",  f"freq_low_{ifo}_cwb",  f"freq_low_{ifo}_pyc",  f"Freq low[{i}]  ({ifo}, Hz)"),
            (f"freq_high_{ifo}", f"freq_high_{ifo}_cwb", f"freq_high_{ifo}_pyc", f"Freq high[{i}]  ({ifo}, Hz)"),
            (f"bw_{ifo}",        f"bandwidth_{ifo}_cwb", f"bandwidth_{ifo}_pyc", f"Bandwidth[{i}]  ({ifo}, Hz)"),
            (f"dur_{ifo}",       f"duration_{ifo}_cwb",  f"duration_{ifo}_pyc",  f"Duration[{i}]  ({ifo}, s)"),
        ])

    # Compute angular separation for sky consistency
    if all(c in merged.columns for c in ["phi_cwb", "theta_cwb", "phi_pyc", "theta_pyc"]):
        merged["ang_sep_deg"] = angular_separation_deg(
            merged["phi_cwb"].to_numpy(), merged["theta_cwb"].to_numpy(),
            merged["phi_pyc"].to_numpy(), merged["theta_pyc"].to_numpy(),
        )
    else:
        merged["ang_sep_deg"] = np.nan

    records = []

    with pdf_backend.PdfPages(out_pdf) as pdf_pages:

        # ── Page 0: Summary text ─────────────────────────────────────────────
        fig = plt.figure(figsize=(11, 8.5))
        ax  = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")

        n_cwb   = len(merged) + len(unmatched_cwb)
        n_pyc   = len(merged) + len(unmatched_pyc)
        n_match = len(merged)

        merged["gps_time_diff"] = merged["gps_time_pyc"] - merged["gps_time_cwb"]
        dt = merged["gps_time_diff"].to_numpy()
        dt_mask = np.isfinite(dt)
        dt_mean = np.mean(dt[dt_mask]) if dt_mask.any() else np.nan
        dt_std  = np.std( dt[dt_mask]) if dt_mask.any() else np.nan

        ang = merged["ang_sep_deg"].to_numpy()
        ang_mask = np.isfinite(ang)
        ang_median = np.median(ang[ang_mask]) if ang_mask.any() else np.nan
        ang_90     = np.percentile(ang[ang_mask], 90) if ang_mask.any() else np.nan

        # Unmatched pycWB: inside / outside cWB analysis windows
        if "cwb_window_status" in unmatched_pyc.columns:
            wc = unmatched_pyc["cwb_window_status"].value_counts()
            n_unm_inside  = wc.get("inside_cwb_window",  0)
            n_unm_outside = wc.get("outside_cwb_window", 0)
            _unm_window_line = (
                f"\n            ── Unmatched pycWB — cWB window breakdown ────────────────────"
                f"\n            Inside  cWB window    : {n_unm_inside}"
                f"  ({100*n_unm_inside /max(len(unmatched_pyc),1):.1f}%)"
                f"\n            Outside cWB window    : {n_unm_outside}"
                f"  ({100*n_unm_outside/max(len(unmatched_pyc),1):.1f}%)"
            )
        else:
            _unm_window_line = ""

        # Livetime section
        if live_data is not None:
            cwb_zl  = live_data["cwb_zero_lag_livetime"]
            cwb_tot = live_data["cwb_total_livetime"]
            n_cwb_jobs = live_data["n_jobs"]
            if pyc_progress is not None:
                pyc_zl  = pyc_progress["pyc_zero_lag_livetime"]
                pyc_tot = pyc_progress["pyc_total_livetime"]
                n_pyc_jobs = pyc_progress["n_jobs"]
                _lt_pyc   = f"{pyc_zl:.1f} s  ({pyc_zl/86400:.4f} d)"
                _lt_ratio = f"{pyc_zl/cwb_zl:.4f}" if cwb_zl > 0 else "N/A"
                _pyc_lines = (
                    f"\n            pycWB zero-lag        : {_lt_pyc}"
                    f"\n            pycWB total livetime  : {pyc_tot:.1f} s  ({pyc_tot/86400:.4f} d)"
                    f"\n            pycWB analysis jobs   : {n_pyc_jobs}"
                    f"\n            pycWB / cWB ratio     : {_lt_ratio}"
                )
            else:
                _pyc_lines = "\n            pycWB zero-lag        : N/A"
            _livetime_section = (
                f"\n            ── Livetime ────────────────────────────────────────────────────"
                f"\n            cWB zero-lag livetime : {cwb_zl:.1f} s  ({cwb_zl/86400:.4f} d)"
                f"\n            cWB total livetime    : {cwb_tot:.1f} s  ({cwb_tot/86400:.4f} d)"
                f"\n            cWB analysis jobs     : {n_cwb_jobs}"
                + _pyc_lines
            )
        else:
            _livetime_section = ""

        text = textwrap.dedent(f"""\
            pycWB vs cWB  —  Consistency Report
            ====================================

            Parquet (pycWB) : {merged.attrs.get('parquet_file', 'N/A')}
            ROOT   (cWB)    : {merged.attrs.get('root_file',    'N/A')}
            IFOs            : {', '.join(ifo_list)}

            ── Trigger counts ───────────────────────────────────────────────
            cWB triggers          : {n_cwb}  (matched: {n_match}, unmatched: {len(unmatched_cwb)})
            pycWB triggers        : {n_pyc}  (matched: {n_match}, unmatched: {len(unmatched_pyc)})
            Matched pairs         : {n_match}
            Match rate (cWB)      : {100*n_match/max(n_cwb,1):.1f} %
            Match rate (pycWB)    : {100*n_match/max(n_pyc,1):.1f} %
        """) + _unm_window_line + _livetime_section + textwrap.dedent(f"""

            ── GPS time residuals (pycWB - cWB) ────────────────────────────
            Mean Δt               : {dt_mean:+.4f} s
            Std  Δt               : {dt_std:.4f} s

            ── Sky-location angular separation ─────────────────────────────
            Median separation     : {ang_median:.3f} °
            90th percentile       : {ang_90:.3f} °
        """)
        ax.text(0.05, 0.95, text, transform=ax.transAxes,
                fontsize=11, va="top", fontfamily="monospace",
                bbox=dict(boxstyle="round", fc="#f5f5f5", ec="gray"))
        pdf_pages.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ── GPS Δt histogram ─────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("GPS Time Residuals (pycWB − cWB)", fontsize=12)
        ax0, ax1 = axes
        _residual_panel(ax0, merged["gps_time_cwb"].to_numpy(),
                        merged["gps_time_pyc"].to_numpy(), "GPS time (s)", color="darkorange")
        # Angular separation histogram
        ax1.hist(ang[ang_mask], bins=50, color="purple", alpha=0.7, edgecolor="none")
        ax1.axvline(ang_median, color="red", lw=1.2, ls="--",
                    label=f"median={ang_median:.2f}°")
        ax1.set_xlabel("Angular separation (degrees)", fontsize=9)
        ax1.set_ylabel("Counts", fontsize=9)
        ax1.set_title("Sky-Location Angular Separation", fontsize=10)
        ax1.legend(fontsize=8)
        plt.tight_layout()
        pdf_pages.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ── Feature-by-feature scatter + residual pages ───────────────────────
        for group_name, features in FEATURE_GROUPS.items():
            n = len(features)
            ncols = 4
            nrows = math.ceil(n / ncols)

            # Scatter page
            fig_s, axes_s = plt.subplots(nrows, ncols,
                                          figsize=(4.5 * ncols, 4.5 * nrows),
                                          squeeze=False)
            fig_s.suptitle(f"{group_name}  —  Scatter (cWB  vs  pycWB)", fontsize=13)
            # Residual page
            fig_r, axes_r = plt.subplots(nrows, ncols,
                                          figsize=(4.5 * ncols, 4 * nrows),
                                          squeeze=False)
            fig_r.suptitle(f"{group_name}  —  Residuals (pycWB − cWB)", fontsize=13)

            for idx, (key, cwb_col, pyc_col, label) in enumerate(features):
                r, c = divmod(idx, ncols)
                x = merged[cwb_col].to_numpy(dtype=np.float64) if cwb_col in merged.columns else np.full(len(merged), np.nan)
                y = merged[pyc_col].to_numpy(dtype=np.float64) if pyc_col in merged.columns else np.full(len(merged), np.nan)

                _scatter_panel(axes_s[r, c], x, y, f"cWB  {label}", f"pycWB  {label}", label)
                _residual_panel(axes_r[r, c], x, y, label)

                rec = _stats_row(x, y)
                rec["group"]   = group_name
                rec["feature"] = key
                rec["label"]   = label
                records.append(rec)

            # Hide unused axes
            for idx in range(len(features), nrows * ncols):
                r, c = divmod(idx, ncols)
                axes_s[r, c].set_visible(False)
                axes_r[r, c].set_visible(False)

            plt.figure(fig_s.number); plt.tight_layout(rect=[0, 0, 1, 0.96])
            pdf_pages.savefig(fig_s, bbox_inches="tight"); plt.close(fig_s)
            plt.figure(fig_r.number); plt.tight_layout(rect=[0, 0, 1, 0.96])
            pdf_pages.savefig(fig_r, bbox_inches="tight"); plt.close(fig_r)

        # ── Unmatched-trigger distributions ──────────────────────────────────
        # Build dicts of matched values for overlay comparison
        matched_cwb_vals = {c: merged[c].to_numpy(dtype=np.float64)
                            for c in merged.columns if c.endswith("_cwb")}
        matched_pyc_vals = {c: merged[c].to_numpy(dtype=np.float64)
                            for c in merged.columns if c.endswith("_pyc")}
        _distribution_pages(pdf_pages, unmatched_cwb, unmatched_pyc,
                            matched_cwb_vals, matched_pyc_vals, ifo_list)

        # ── Log-based classification of unmatched cWB triggers ───────────────
        if "pycwb_status" in unmatched_cwb.columns:
            _pycwb_status_pages(pdf_pages, unmatched_cwb, ifo_list)

        # ── cWB unmatched triggers classified by pycWB job windows ───────────
        if "pyc_window_status" in unmatched_cwb.columns:
            _cwb_pyc_window_pages(pdf_pages, unmatched_cwb, ifo_list)

        # ── Livetime comparison + pycWB unmatched window classification ───────
        if live_data is not None:
            _livetime_comparison_pages(pdf_pages, live_data, pyc_progress,
                                       unmatched_pyc, ifo_list)

        # ── Reference-event found/missing pages ──────────────────────────────
        if ref_df is not None:
            _ref_events_pages(pdf_pages, ref_df, ifo_list)

        # ── Angular separation vs rho0 ────────────────────────────────────────
        if "rho0_cwb" in merged.columns:
            fig, ax = plt.subplots(figsize=(8, 5))
            mask = np.isfinite(merged["ang_sep_deg"]) & np.isfinite(merged["rho0_cwb"])
            ax.scatter(merged["rho0_cwb"][mask], merged["ang_sep_deg"][mask],
                       s=8, alpha=0.4, color="teal", rasterized=True)
            ax.set_xlabel("cWB  rho0", fontsize=10)
            ax.set_ylabel("Sky angular separation (deg)", fontsize=10)
            ax.set_title("Sky-location discrepancy vs. rho0", fontsize=11)
            ax.set_yscale("log")
            pdf_pages.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # ── Statistics table page ─────────────────────────────────────────────
        stats_df = pd.DataFrame(records).set_index("feature")
        numeric_cols = ["N", "mean_diff", "std_diff", "rms_diff", "max_abs_diff", "pearson_r"]
        fig, ax = plt.subplots(figsize=(15, max(4, 0.35 * len(stats_df) + 2)))
        ax.axis("off")
        tbl_data = stats_df[["label", "N", "mean_diff", "std_diff",
                               "rms_diff", "max_abs_diff", "pearson_r"]].copy()
        for col in ["mean_diff", "std_diff", "rms_diff", "max_abs_diff", "pearson_r"]:
            tbl_data[col] = tbl_data[col].map(lambda v: f"{v:.4g}" if pd.notna(v) else "—")
        tbl = ax.table(
            cellText=tbl_data.reset_index().values.tolist(),
            colLabels=["Feature", "Label", "N",
                       "Mean diff", "Std diff", "RMS diff", "Max|diff|", "Pearson r"],
            loc="center", cellLoc="right",
        )
        tbl.auto_set_font_size(False); tbl.set_fontsize(8)
        tbl.auto_set_column_width(range(8))
        ax.set_title("Summary statistics (pycWB − cWB)", fontsize=12, pad=10)
        pdf_pages.savefig(fig, bbox_inches="tight"); plt.close(fig)

    log.info("Report saved → %s", out_pdf)

    # Save matched CSV
    merged.to_csv(out_csv, index=False, float_format="%.9f")
    log.info("Matched triggers CSV → %s", out_csv)

    # Save unmatched CSVs (derived from out_csv stem)
    csv_path = Path(out_csv)
    out_csv_unmatched_cwb = str(csv_path.parent / (csv_path.stem + "_unmatched_cwb" + csv_path.suffix))
    out_csv_unmatched_pyc = str(csv_path.parent / (csv_path.stem + "_unmatched_pyc" + csv_path.suffix))
    unmatched_cwb.to_csv(out_csv_unmatched_cwb, index=False, float_format="%.9f")
    log.info("Unmatched cWB CSV   → %s", out_csv_unmatched_cwb)
    unmatched_pyc.to_csv(out_csv_unmatched_pyc, index=False, float_format="%.9f")
    log.info("Unmatched pycWB CSV → %s", out_csv_unmatched_pyc)

    # Save reference-events CSV (with ref_status + matched pycwb columns)
    out_csv_ref = None
    if ref_df is not None:
        out_csv_ref = str(csv_path.parent / (csv_path.stem + "_ref_events" + csv_path.suffix))
        ref_df.to_csv(out_csv_ref, index=False, float_format="%.9f")
        log.info("Reference events CSV → %s", out_csv_ref)

    return stats_df, out_csv_unmatched_cwb, out_csv_unmatched_pyc, out_csv_ref


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--parquet", default=_DEFAULT_PARQUET,
                   help="Path to pycWB Parquet catalog  [%(default)s]")
    p.add_argument("--root", default=_DEFAULT_ROOT,
                   help="Path to cWB ROOT waveburst file  [%(default)s]")
    p.add_argument("--ifo", nargs="+", default=["L1", "H1"],
                   help="IFO names in network order  [%(default)s]")
    p.add_argument("--tree", default="waveburst",
                   help="ROOT tree name  [%(default)s]")
    p.add_argument("--tol", type=float, default=0.05,
                   help="GPS match tolerance in seconds  [%(default)s]")
    p.add_argument("--out", default=_DEFAULT_OUT,
                   help="Output PDF report path  [%(default)s]")
    p.add_argument("--csv", default=_DEFAULT_CSV,
                   help="Output matched-trigger CSV path  [%(default)s]")
    p.add_argument("--log", default=None,
                   help="Path to pycWB log file or log directory for unmatched-cWB "
                        "classification (optional)")
    p.add_argument("--ref_events", default=None,
                   help="Reference events CSV (first column = GPS time). "
                        "Matched against pycWB catalog; found/missing statistics "
                        "are added to the report (optional)")
    p.add_argument("--live", default=None,
                   help="Path to cWB live_*.root file.  Enables livetime comparison "
                        "and classifies unmatched pycWB triggers as inside/outside "
                        "cWB analysis windows (optional)")
    p.add_argument("--progress", default=None,
                   help="Path to pycWB progress Parquet file (progress*.parquet). "
                        "Provides pycWB zero-lag livetime and job analysis windows "
                        "for the report (optional)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ── Load ─────────────────────────────────────────────────────────────────
    df_cwb = load_root(args.root, args.ifo, tree_name=args.tree)
    df_pyc = load_parquet(args.parquet, args.ifo)

    # ── Match ────────────────────────────────────────────────────────────────
    merged, unmatched_cwb, unmatched_pyc = match_triggers(df_cwb, df_pyc, tol=args.tol)
    merged.attrs["parquet_file"] = args.parquet
    merged.attrs["root_file"]    = args.root

    if len(merged) == 0:
        log.error("No matched triggers found — check --tol or the input files.")
        sys.exit(1)

    # ── Classify unmatched cWB triggers via log (optional) ───────────────────
    if args.log:
        unmatched_cwb = classify_unmatched_cwb_from_logs(unmatched_cwb, args.log)

    # ── Match reference events to pycWB catalog (optional) ───────────────────
    ref_df = None
    if args.ref_events:
        df_ref_raw = load_ref_events(args.ref_events)
        ref_df = match_ref_to_pyc(df_ref_raw, df_pyc, tolerance=args.tol)

    # ── Load live ROOT file: cWB analysis windows + livetime (optional) ───────
    live_data    = None
    pyc_progress = None
    if args.live:
        live_data = load_live_root(args.live, n_ifo=len(args.ifo))
        unmatched_pyc = classify_unmatched_pyc_by_cwb_time(
            unmatched_pyc, live_data["job_windows"])
    if args.progress:
        pyc_progress = load_pyc_progress(args.progress, args.parquet)
        unmatched_cwb = classify_unmatched_cwb_by_pyc_time(
            unmatched_cwb, pyc_progress["job_windows"])

    # ── Report ───────────────────────────────────────────────────────────────
    stats_df, csv_unm_cwb, csv_unm_pyc, csv_ref = build_report(
        merged, unmatched_cwb, unmatched_pyc, args.ifo, args.out, args.csv,
        ref_df=ref_df, live_data=live_data, pyc_progress=pyc_progress)

    # Print summary to stdout
    print("\n── Consistency statistics (pycWB − cWB) ──")
    print(stats_df[["label", "N", "mean_diff", "rms_diff", "pearson_r"]].to_string())
    if live_data is not None:
        cwb_zl = live_data["cwb_zero_lag_livetime"]
        print("\n── Livetime ──")
        print(f"cWB zero-lag livetime : {cwb_zl:.1f} s  ({cwb_zl/86400:.4f} d)")
        if pyc_progress is not None:
            pyc_zl = pyc_progress["pyc_zero_lag_livetime"]
            print(f"pycWB zero-lag        : {pyc_zl:.1f} s  ({pyc_zl/86400:.4f} d)")
            print(f"pycWB / cWB ratio     : {pyc_zl/cwb_zl:.4f}")
        if "cwb_window_status" in unmatched_pyc.columns:
            wc = unmatched_pyc["cwb_window_status"].value_counts()
            print("\n── Unmatched pycWB window breakdown ──")
            for status, count in wc.items():
                print(f"  {status:25s}: {count}")
    print(f"\nReport                → {args.out}")
    print(f"CSV (matched)         → {args.csv}")
    print(f"CSV (unmatched cWB)   → {csv_unm_cwb}")
    print(f"CSV (unmatched pycWB) → {csv_unm_pyc}")
    if csv_ref:
        print(f"CSV (ref events)      → {csv_ref}")


if __name__ == "__main__":
    main()
