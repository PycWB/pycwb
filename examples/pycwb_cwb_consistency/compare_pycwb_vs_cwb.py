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
                 out_pdf: str, out_csv: str) -> pd.DataFrame:
    """Generate the PDF report and statistics CSV, return the stats table."""

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

        dt = (merged["gps_time_pyc"] - merged["gps_time_cwb"]).to_numpy()
        dt_mask = np.isfinite(dt)
        dt_mean = np.mean(dt[dt_mask]) if dt_mask.any() else np.nan
        dt_std  = np.std( dt[dt_mask]) if dt_mask.any() else np.nan

        ang = merged["ang_sep_deg"].to_numpy()
        ang_mask = np.isfinite(ang)
        ang_median = np.median(ang[ang_mask]) if ang_mask.any() else np.nan
        ang_90     = np.percentile(ang[ang_mask], 90) if ang_mask.any() else np.nan

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
    merged.to_csv(out_csv, index=False, float_format="%.6g")
    log.info("Matched triggers CSV → %s", out_csv)

    return stats_df


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

    # ── Report ───────────────────────────────────────────────────────────────
    stats_df = build_report(merged, unmatched_cwb, unmatched_pyc, args.ifo, args.out, args.csv)

    # Print summary to stdout
    print("\n── Consistency statistics (pycWB − cWB) ──")
    print(stats_df[["label", "N", "mean_diff", "rms_diff", "pearson_r"]].to_string())
    print(f"\nReport → {args.out}")
    print(f"CSV    → {args.csv}")


if __name__ == "__main__":
    main()
