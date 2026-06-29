"""Reusable histogram plotting helpers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure


def plot_1d_histogram(
    values: np.ndarray | Sequence[np.ndarray],
    colors: str | Sequence[str],
    label: str | None = None,
    ax: Axes | None = None,
    bins: int = 20,
    xlim: tuple[float, float] | None = None,
    x_edges: np.ndarray | None = None,
) -> tuple[Figure, Axes]:
    """Plot one or more one-dimensional histograms.

    Non-finite values are dropped before computing the default bin edges. The
    function returns the owning figure and axes; callers decide whether and
    where to save the plot.
    """
    series = _as_series(values)
    palette = _as_palette(colors, len(series))
    finite_series = [_finite_array(item) for item in series]
    nonempty_series = [item for item in finite_series if len(item) > 0]
    if not nonempty_series:
        raise ValueError("No finite values available for histogram")
    combined = np.concatenate(nonempty_series)

    if x_edges is None:
        spread = float(np.std(combined))
        if spread == 0.0:
            spread = 0.5
        x_min = float(np.min(combined) - spread)
        x_max = float(np.max(combined) + spread)
        x_edges = np.linspace(x_min, x_max, bins + 1)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    for data, color in zip(finite_series, palette):
        ax.hist(data, bins=x_edges, color=color, alpha=0.5, label=label)

    if label:
        ax.legend()
    if xlim:
        ax.set_xlim(*xlim)
    return fig, ax


def plot_2d_histogram(
    x: np.ndarray,
    y: np.ndarray,
    xbins: int = 100,
    ybins: int = 100,
    xscale: str = "lin",
    yscale: str = "lin",
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    cmap: str = "viridis",
    xlabel: str = "",
    ylabel: str = "",
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot a two-dimensional histogram with logarithmic color scaling."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) == 0 or len(y) == 0:
        raise ValueError("No finite x/y pairs available for histogram")

    x_bins = _bin_edges(x, xbins, xscale)
    y_bins = _bin_edges(y, ybins, yscale)
    hist, xedges, yedges = np.histogram2d(x, y, bins=[x_bins, y_bins])
    hist = np.where(hist == 0, np.nan, hist.astype(float))

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    mesh = ax.pcolormesh(
        xedges,
        yedges,
        hist.T,
        norm=LogNorm(),
        cmap=cmap,
        shading="auto",
    )
    fig.colorbar(mesh, ax=ax)

    if xscale == "log":
        ax.set_xscale("log")
    if yscale == "log":
        ax.set_yscale("log")

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xlim(*(xlim or (float(xedges[0]), float(xedges[-1]))))
    ax.set_ylim(*(ylim or (float(yedges[0]), float(yedges[-1]))))
    ax.grid(ls="dotted")
    return fig, ax


def _as_series(values: np.ndarray | Sequence[np.ndarray]) -> list[np.ndarray]:
    if isinstance(values, np.ndarray):
        return [values]
    return [np.asarray(item) for item in values]


def _as_palette(colors: str | Sequence[str], size: int) -> list[str]:
    if isinstance(colors, str):
        return [colors] * size
    palette = list(colors)
    if len(palette) != size:
        raise ValueError("colors must be a string or match the number of series")
    return palette


def _finite_array(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def _bin_edges(values: np.ndarray, bins: int, scale: str) -> np.ndarray:
    low = float(np.min(values))
    high = float(np.max(values))
    if low == high:
        pad = 0.5 if low == 0 else abs(low) * 0.05
        low -= pad
        high += pad
    if scale == "log":
        if low <= 0:
            raise ValueError("Log-scaled histogram data must be positive")
        return np.logspace(np.log10(low), np.log10(high), bins)
    if scale != "lin":
        raise ValueError("scale must be either 'lin' or 'log'")
    return np.linspace(low, high, bins)
