"""XGBoost feature correlation heatmap — interactive diagnostic step.

Generates an interactive lower-triangle grid of 2D histograms showing
pairwise correlations between XGBoost training features (1D histograms
on the diagonal).  Primary output is a self-contained HTML file using
Plotly; PNG export is available as an option.

Workflow action
---------------
``postprocess.xgb_heatmap.plot_xgb_heatmap``

Parameters (via YAML ``args``)
------------------------------
work_dir : str
    Base directory; relative paths are resolved against this.
catalog : str
    Path to the catalog parquet file to visualize.
label : str, optional
    Label for the run shown in titles (default: stem of catalog path).
features : list[str], optional
    List of feature column names to include.  Defaults to the standard
    XGBoost training feature set.  Columns must exist in the catalog
    (checked at runtime; missing columns are skipped with a warning).
log_features : list[str], optional
    Subset of *features* to display on a log₁₀ scale.  Only features
    whose values are strictly positive are included in the log₁₀
    transform (non-positive values are dropped).
bins : int, default 40
    Number of bins per axis for 2D histograms.
cmap : str, default "Blues"
    Matplotlib / Plotly-compatible colormap name.
output_file : str or list[str], default "xgb_heatmap.html"
    Output path(s).  Extension determines format:
    ``.html`` → interactive Plotly; ``.png`` → static matplotlib.
    Pass a list for both: ``["xgb_heatmap.html", "xgb_heatmap.png"]``.
title : str, optional
    Custom title.  Auto-generated when omitted.
width : int, default 1400
    Plot width in pixels (HTML only).
height : int, default 1200
    Plot height in pixels (HTML only).
downsample : int, optional
    Maximum points for 2D histograms (random sample).  No limit by default.

Returns
-------
dict
    ``{"output_file": str}`` or ``{"output_files": [str, ...]}``
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from pycwb.post_production.action_spec import action_spec

logger = logging.getLogger(__name__)

# ── Default XGBoost training features (raw parquet column names) ──────
# These match columns in pycWB catalog parquet files.
DEFAULT_FEATURES = [
    "packet_norm",
    "net_cc",
    "penalty",
    "q_veto",
    "q_factor",
    "sSNR0_over_lik",   # auto-computed: signal_energy_{ifo0} / likelihood
    "sSNR1_over_lik",   # auto-computed: signal_energy_{ifo1} / likelihood
    "rho0",             # auto-computed: sqrt(coherent_energy / penalty)
]

DEFAULT_LOG_FEATURES = {
    "penalty",
    "packet_norm",
    "sSNR0_over_lik",
    "sSNR1_over_lik",
}

# ── Auto-computed feature recipes ─────────────────────────────────────
# Each entry: (target_name, recipe_fn)
# recipe_fn receives the full DataFrame and returns a float64 numpy array.


def _compute_rho0(df: pd.DataFrame) -> np.ndarray:
    """rho0 = sqrt(coherent_energy / penalty), with safeguards."""
    ecor = df["coherent_energy"].to_numpy(dtype=np.float64) if "coherent_energy" in df.columns else np.full(len(df), np.nan)
    penalty = df["penalty"].to_numpy(dtype=np.float64) if "penalty" in df.columns else np.full(len(df), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where((ecor > 0) & (penalty > 0), ecor / penalty, np.nan)
        return np.sqrt(ratio)


def _compute_sSNR0_over_lik(df: pd.DataFrame) -> np.ndarray:
    """sSNR[0] / likelihood using first IFO signal_energy."""
    # Try common IFO column names
    for ifo in ["L1", "H1", "V1", "K1"]:
        col = f"signal_energy_{ifo}"
        if col in df.columns:
            ssnr = df[col].to_numpy(dtype=np.float64)
            break
    else:
        return np.full(len(df), np.nan)
    lik = df["likelihood"].to_numpy(dtype=np.float64) if "likelihood" in df.columns else np.full(len(df), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(lik != 0, ssnr / lik, np.nan)


def _compute_sSNR1_over_lik(df: pd.DataFrame) -> np.ndarray:
    """sSNR[1] / likelihood using second IFO signal_energy."""
    for ifo in ["H1", "V1", "K1"]:
        col = f"signal_energy_{ifo}"
        if col in df.columns:
            ssnr = df[col].to_numpy(dtype=np.float64)
            break
    else:
        return np.full(len(df), np.nan)
    lik = df["likelihood"].to_numpy(dtype=np.float64) if "likelihood" in df.columns else np.full(len(df), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(lik != 0, ssnr / lik, np.nan)


AUTO_FEATURES: dict[str, callable] = {
    "rho0": _compute_rho0,
    "sSNR0_over_lik": _compute_sSNR0_over_lik,
    "sSNR1_over_lik": _compute_sSNR1_over_lik,
}


# ── Helpers ────────────────────────────────────────────────────────────

def _resolve(work_dir: str, path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(work_dir, path)


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def _load_catalog(parquet_path: str) -> pd.DataFrame:
    """Read a pycWB parquet catalog via pyarrow."""
    logger.info("Reading catalog: %s", parquet_path)
    table = pq.read_table(parquet_path)
    df = table.to_pandas()
    logger.info("  %d triggers", len(df))
    return df


def _extract_features(
    df: pd.DataFrame,
    features: list[str],
    log_features: set[str],
    downsample: Optional[int],
) -> tuple[np.ndarray, list[str], int]:
    """Extract feature matrix, applying log₁₀ to specified columns.

    Supports auto-computed features defined in :data:`AUTO_FEATURES`.
    """
    arrays: list[np.ndarray] = []
    display_names: list[str] = []
    for feat in features:
        vals: np.ndarray | None = None

        if feat in AUTO_FEATURES:
            # Auto-computed feature
            vals = AUTO_FEATURES[feat](df)
        elif feat in df.columns:
            vals = df[feat].to_numpy(dtype=np.float64)
        else:
            logger.warning("Feature '%s' not found in catalog — skipping", feat)
            continue

        if vals is None:
            continue

        if feat in log_features:
            pos = np.isfinite(vals) & (vals > 0)
            vals = np.where(pos, np.log10(vals), np.nan)
            display_names.append(f"log₁₀ {feat}")
        else:
            display_names.append(feat)
        arrays.append(vals)

    if not arrays:
        raise ValueError("No valid features found in catalog.")

    # Row mask: all features must be finite
    valid = np.ones(len(df), dtype=bool)
    for a in arrays:
        valid = valid & np.isfinite(a)

    mat = np.column_stack([a[valid] for a in arrays])
    n_valid = int(valid.sum())
    logger.info("  %d rows with all features valid", n_valid)

    if downsample and downsample < n_valid:
        rng = np.random.default_rng(42)
        idx = rng.choice(n_valid, downsample, replace=False)
        mat = mat[idx]
        n_valid = downsample
        logger.info("  downsampled to %d points", n_valid)

    return mat, display_names, n_valid


# ────────────────────────────────────────────────────────────────────────
# Matplotlib PNG backend
# ────────────────────────────────────────────────────────────────────────

def _make_png(
    mat: np.ndarray,
    feature_names: list[str],
    label: str,
    output_file: str,
    title: str,
    bins: int,
    cmap: str,
    n_points: int,
) -> str:
    """Generate a static PNG with matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    # Normalize cmap name (matplotlib is case-sensitive)
    cmap_name = cmap.lower() if cmap.lower() in plt.colormaps() else cmap

    n = len(feature_names)
    fig = plt.figure(figsize=(n * 2.2, n * 2.2))
    gs = GridSpec(n, n, figure=fig, hspace=0.06, wspace=0.06)

    for i in range(n):
        for j in range(n):
            ax = fig.add_subplot(gs[i, j])
            if j > i:
                ax.set_visible(False)
                continue

            if i == j:
                vals = mat[:, i]
                lo, hi = np.percentile(vals, [0.5, 99.5])
                if lo == hi:
                    lo, hi = lo - 0.5, hi + 0.5
                ax.hist(vals, bins=np.linspace(lo, hi, bins),
                        density=True, alpha=0.7, color="steelblue",
                        edgecolor="white", linewidth=0.3)
                ax.set_ylabel(feature_names[i], fontsize=8, fontweight="bold")
            else:
                xv, yv = mat[:, j], mat[:, i]
                fm = np.isfinite(xv) & np.isfinite(yv)
                xf, yf = xv[fm], yv[fm]
                if len(xf) > 10:
                    xlo, xhi = np.percentile(xf, [0.5, 99.5])
                    ylo, yhi = np.percentile(yf, [0.5, 99.5])
                    ax.hist2d(xf, yf, bins=bins,
                              range=[[xlo, xhi], [ylo, yhi]],
                              cmap=cmap_name, rasterized=True)
                    try:
                        r_val = np.corrcoef(xf, yf)[0, 1]
                        ax.text(0.95, 0.95, f"r={r_val:.3f}",
                                transform=ax.transAxes, fontsize=6,
                                ha="right", va="top",
                                bbox=dict(boxstyle="round,pad=0.2",
                                         facecolor="white", alpha=0.7))
                    except Exception:
                        pass

            if i == n - 1:
                ax.set_xlabel(feature_names[j], fontsize=7)
            else:
                ax.set_xticklabels([])
            if j > 0:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=5)

    fig.suptitle(f"{title}\n{label} ({n_points:,} events)",
                 fontsize=14, fontweight="bold")
    _ensure_dir(output_file)
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("PNG saved: %s", output_file)
    return output_file


# ────────────────────────────────────────────────────────────────────────
# Plotly HTML backend (interactive)
# ────────────────────────────────────────────────────────────────────────

def _make_html(
    mat: np.ndarray,
    feature_names: list[str],
    label: str,
    output_file: str,
    title: str,
    bins: int,
    cmap: str,
    n_points: int,
    width: int,
    height: int,
) -> str:
    """Generate an interactive HTML heatmap with Plotly."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    n = len(feature_names)
    fig = make_subplots(
        rows=n, cols=n,
        horizontal_spacing=0.02,
        vertical_spacing=0.03,
    )

    # Diagonal: 1D histograms
    for i in range(n):
        vals = mat[:, i]
        lo, hi = np.percentile(vals, [0.5, 99.5])
        if lo == hi:
            lo, hi = lo - 0.5, hi + 0.5
        hist, edges = np.histogram(vals, bins=bins, range=(lo, hi), density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        mean_val = float(np.mean(vals))
        median_val = float(np.median(vals))
        std_val = float(np.std(vals))

        fig.add_trace(
            go.Bar(
                x=centers, y=hist,
                marker=dict(color="steelblue", line=dict(width=0)),
                showlegend=False,
                hovertemplate=(
                    f"<b>{feature_names[i]}</b><br>"
                    f"Value: %{{x:.3g}}<br>"
                    f"Density: %{{y:.3g}}<br>"
                    f"μ={mean_val:.3g}  σ={std_val:.3g}  median={median_val:.3g}"
                    f"<extra></extra>"
                ),
            ),
            row=i + 1, col=i + 1,
        )

    # Lower triangle: 2D histograms
    for i in range(n):
        for j in range(i):
            xv, yv = mat[:, j], mat[:, i]
            fm = np.isfinite(xv) & np.isfinite(yv)
            xf, yf = xv[fm], yv[fm]
            if len(xf) < 10:
                continue

            xlo, xhi = np.percentile(xf, [0.5, 99.5])
            ylo, yhi = np.percentile(yf, [0.5, 99.5])
            if xlo == xhi:
                xlo, xhi = xlo - 0.5, xhi + 0.5
            if ylo == yhi:
                ylo, yhi = ylo - 0.5, yhi + 0.5

            h2d, xe, ye = np.histogram2d(xf, yf, bins=bins,
                                          range=[[xlo, xhi], [ylo, yhi]])
            r_val: float = float(np.corrcoef(xf, yf)[0, 1]) if len(xf) > 1 else 0.0

            fig.add_trace(
                go.Heatmap(
                    z=h2d.T,
                    x=0.5 * (xe[:-1] + xe[1:]),
                    y=0.5 * (ye[:-1] + ye[1:]),
                    colorscale=cmap,
                    showscale=False,
                    hovertemplate=(
                        f"<b>{feature_names[j]}</b>: %{{x:.3g}}<br>"
                        f"<b>{feature_names[i]}</b>: %{{y:.3g}}<br>"
                        f"Count: %{{z:.0f}}<br>"
                        f"r={r_val:.4f}<extra></extra>"
                    ),
                ),
                row=i + 1, col=j + 1,
            )

            # Pearson r annotation in corner
            fig.add_annotation(
                xref=f"x{(i + 1) * n + (j + 1)}",
                yref=f"y{(i + 1) * n + (j + 1)}",
                x=0.95, y=0.95, xanchor="right", yanchor="top",
                text=f"r={r_val:.3f}",
                showarrow=False,
                font=dict(size=9, color="white"),
                bgcolor="rgba(0,0,0,0.4)",
                borderpad=2,
            )

    # Axis labels (bottom row / left column only)
    for i in range(n):
        fig.update_xaxes(title_text=feature_names[i],
                        row=n, col=i + 1,
                        title_font=dict(size=10))
        fig.update_yaxes(title_text=feature_names[i],
                        row=i + 1, col=1,
                        title_font=dict(size=10))

    # Hide upper-triangle subplots — they have no data and waste space
    for i in range(n):
        for j in range(i + 1, n):
            fig.update_xaxes(visible=False, row=i + 1, col=j + 1)
            fig.update_yaxes(visible=False, row=i + 1, col=j + 1)

    fig.update_layout(
        title=dict(
            text=f"{title}<br><sup>{label} ({n_points:,} events)</sup>",
            font=dict(size=18),
        ),
        width=width,
        height=height,
        template="plotly_white",
        margin=dict(l=80, r=40, t=100, b=40),
    )

    _ensure_dir(output_file)
    fig.write_html(output_file, include_plotlyjs="cdn",
                   full_html=True, auto_open=False)
    logger.info("HTML saved: %s", output_file)
    return output_file


# ────────────────────────────────────────────────────────────────────────
# Main action
# ────────────────────────────────────────────────────────────────────────

@action_spec(
    outputs=["output_file"],
    inputs=["catalog"],
    display_name="XGB feature heatmap",
    description=(
        "Interactive lower-triangle 2D correlation heatmap "
        "for XGBoost training features"
    ),
)
def plot_xgb_heatmap(
    work_dir: str,
    catalog: str,
    label: Optional[str] = None,
    features: Optional[list[str]] = None,
    log_features: Optional[list[str]] = None,
    bins: int = 40,
    cmap: str = "Blues",
    output_file: Union[str, list[str]] = "xgb_heatmap.html",
    title: Optional[str] = None,
    width: int = 1400,
    height: int = 1200,
    downsample: Optional[int] = None,
    **kwargs,
) -> dict:
    """Generate XGBoost feature correlation heatmap.

    Reads *catalog* (parquet), extracts *features*, and produces a
    lower-triangle grid: 1D histograms on the diagonal, 2D heatmaps
    (with Pearson *r*) in the lower triangle.  Output format is
    determined by the ``output_file`` extension(s).
    """
    # ── Resolve paths ─────────────────────────────────────────────────
    catalog_path = _resolve(work_dir, catalog)

    # ── Set defaults ──────────────────────────────────────────────────
    if label is None:
        label = Path(catalog).stem

    if features is None:
        features = list(DEFAULT_FEATURES)

    if log_features is None:
        log_features = list(DEFAULT_LOG_FEATURES)

    if title is None:
        title = "XGBoost Training Features — Correlation Heatmap"

    log_set = set(log_features)

    # ── Load & extract ────────────────────────────────────────────────
    df = _load_catalog(catalog_path)
    mat, display_names, n_valid = _extract_features(
        df, features, log_set, downsample,
    )

    if mat.shape[1] < 2:
        raise ValueError(
            f"Need at least 2 valid features; got {mat.shape[1]}. "
            f"Available columns in catalog: {sorted(df.columns)}"
        )

    # ── Generate output(s) ────────────────────────────────────────────
    # Normalize cmap name (matplotlib is case-sensitive, Plotly isn't)
    targets: list[str] = (
        output_file if isinstance(output_file, list) else [output_file]
    )

    outputs: list[str] = []
    for out in targets:
        resolved = _resolve(work_dir, out)
        ext = os.path.splitext(resolved)[1].lower()

        if ext in (".html", ".htm"):
            outputs.append(
                _make_html(mat, display_names, label, resolved,
                          title, bins, cmap, n_valid, width, height)
            )
        elif ext in (".png",):
            outputs.append(
                _make_png(mat, display_names, label, resolved,
                         title, bins, cmap, n_valid)
            )
        else:
            logger.warning(
                "Unrecognized extension '%s' — defaulting to HTML", ext,
            )
            html_path = os.path.splitext(resolved)[0] + ".html"
            outputs.append(
                _make_html(mat, display_names, label, html_path,
                          title, bins, cmap, n_valid, width, height)
            )

    if len(outputs) == 1:
        return {"output_file": outputs[0]}
    return {"output_files": outputs}
