"""Time-frequency plots of per-pixel cluster statistics.

The layout intentionally follows cWB's ``watplot::plot(netcluster*, ...)``
monster-event display.  In particular, pixels from every WDM resolution are
expanded onto the finest common grid and the axes are cropped around the
cluster.  Plotting the old sparse map directly as a GWpy ``Spectrogram`` kept
the unused frequency interval from 0 Hz to the cluster in view, which squeezed
high-frequency events into the upper corner of the image.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import numpy as np


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _StatisticsMap:
    values: np.ndarray
    time_edges: np.ndarray
    frequency_edges: np.ndarray
    xlim: tuple[float, float]
    ylim: tuple[float, float]
    total: float
    npix: int
    min_dt: float
    max_dt: float
    min_df: float
    max_df: float
    display_df: float


def _prepare_statistics_map(cluster, key: str) -> _StatisticsMap:
    """Build the multiresolution display grid used by cWB.

    Pixel ``time`` is a WDM coefficient index rather than a time-bin index.
    The conversion and the rectangle expansion below mirror
    ``wat/watplot.cc``.  Keeping that logic here also avoids allocating the
    large zero-frequency prefix produced by ``Cluster.get_sparse_map``.
    """
    if key not in {"likelihood", "null"}:
        raise ValueError("key must be either 'likelihood' or 'null'")

    pa = cluster.pixel_arrays
    if pa is None or len(pa) == 0:
        raise ValueError("cannot plot statistics for an empty cluster")

    core = np.asarray(pa.core, dtype=bool)
    valid = (
        core
        & np.isfinite(pa.rate)
        & (pa.rate > 0)
        & (pa.layers > 1)
    )
    if not np.any(valid):
        raise ValueError("cluster has no valid core pixels to plot")

    times = np.asarray(pa.time[valid], dtype=np.int64)
    frequencies = np.asarray(pa.frequency[valid], dtype=np.int64)
    layers = np.asarray(pa.layers[valid], dtype=np.int64)
    rates = np.asarray(pa.rate[valid], dtype=np.float64)
    statistics = np.asarray(getattr(pa, key)[valid], dtype=np.float64)
    statistics = np.maximum(statistics, 0.0)

    # For WDM pixels, rate * (layers - 1) is the original analysis rate.
    # Round individual estimates first to tolerate float32 pixel rates.
    analysis_rates = np.rint(rates * (layers - 1)).astype(np.int64)
    analysis_rate = int(np.max(analysis_rates))
    if analysis_rate <= 0:
        raise ValueError("could not infer a positive analysis rate")
    if not np.allclose(analysis_rates, analysis_rate, rtol=0, atol=1):
        raise ValueError(
            "inconsistent WDM analysis rates in cluster: "
            f"{sorted(set(analysis_rates.tolist()))}"
        )

    min_layers = int(np.min(layers))
    max_layers = int(np.max(layers))
    min_rate = analysis_rate / (max_layers - 1)
    max_rate = analysis_rate / (min_layers - 1)

    min_dt = 1.0 / max_rate
    max_dt = 1.0 / min_rate
    min_df = min_rate / 2.0
    max_df = max_rate / 2.0

    pixel_dt = 1.0 / rates
    pixel_start = np.floor_divide(times, layers) / rates - pixel_dt / 2.0
    min_time = float(np.min(pixel_start))
    max_time = float(np.max(pixel_start + pixel_dt))

    initial_min_time = min_time - max_dt
    initial_max_time = max_time + max_dt
    n_time = int((initial_max_time - initial_min_time) * max_rate)
    n_frequency = 2 * (max_layers - 1)
    display_df = analysis_rate / (2.0 * n_frequency)
    values = np.zeros((n_time, n_frequency), dtype=np.float64)

    for time, frequency, layer, rate, statistic in zip(
        pixel_start, frequencies, layers, rates, statistics
    ):
        time_scale = int(max_rate / rate)
        frequency_scale = int(
            2 * (max_layers - 1) / int(layer - 1)
        )
        # Preserve cWB's truncation here.  Half-bin starts occur naturally in
        # WDM maps; rounding them changes which pixels overlap and therefore
        # changes the displayed likelihood maximum.
        time_index = int((time - initial_min_time) * max_rate)
        frequency_index = int(frequency * frequency_scale)
        frequency_start = frequency_index - frequency_scale // 2

        time_slice = slice(time_index, time_index + time_scale)
        frequency_slice = slice(
            frequency_start, frequency_start + frequency_scale
        )
        values[time_slice, frequency_slice] += statistic

    time_edges = initial_min_time + np.arange(n_time + 1) / max_rate
    frequency_edges = np.arange(n_frequency + 1) * display_df

    pixel_frequency = frequencies * rates / 2.0
    min_frequency = float(np.min(pixel_frequency))
    max_frequency = float(np.max(pixel_frequency))
    frequency_margin = max(
        (max_frequency - min_frequency) / 10.0,
        2.0 * max_df,
    )
    ylim = (
        max(0.0, min_frequency - frequency_margin),
        min(analysis_rate / 2.0, max_frequency + frequency_margin),
    )

    time_margin = max(
        (max_time - min_time) / 10.0,
        2.0 * max_dt,
    )
    xlim = (
        max(initial_min_time, min_time - time_margin),
        min(initial_max_time, max_time + time_margin),
    )

    return _StatisticsMap(
        values=values,
        time_edges=time_edges,
        frequency_edges=frequency_edges,
        xlim=xlim,
        ylim=ylim,
        total=float(np.sum(statistics)),
        npix=int(np.count_nonzero(statistics > 0)),
        min_dt=min_dt,
        max_dt=max_dt,
        min_df=min_df,
        max_df=max_df,
        display_df=display_df,
    )


def plot_statistics(cluster, key="likelihood", gps_shift=0, filename=None):
    """Plot a cluster likelihood or null map in the cWB CED layout.

    Parameters
    ----------
    cluster
        Cluster whose core pixels will be displayed.
    key : {"likelihood", "null"}
        Per-pixel statistic to plot.
    gps_shift : float
        GPS offset printed below the relative event-time axis.
    filename : path-like, optional
        Output image path.  When omitted, the figure is returned without
        writing a file.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.  It is closed before return when ``filename`` is
        supplied, matching the previous function's resource-management
        behavior; its axes remain available for inspection.
    """
    statistics_map = _prepare_statistics_map(cluster, key)

    fig, ax = plt.subplots(figsize=(8, 5.75), dpi=100)
    cmap = plt.get_cmap("turbo", 256).with_extremes(bad="white")
    masked_values = np.ma.masked_less_equal(statistics_map.values.T, 0)
    mesh = ax.pcolormesh(
        statistics_map.time_edges,
        statistics_map.frequency_edges,
        masked_values,
        cmap=cmap,
        vmin=0,
        vmax=(
            float(np.max(statistics_map.values))
            if np.any(statistics_map.values > 0)
            else 1.0
        ),
        shading="flat",
        rasterized=True,
    )

    ax.set_xlim(*statistics_map.xlim)
    ax.set_ylim(*statistics_map.ylim)
    ax.set_xlabel(f"Time (sec) : GPS OFFSET = {float(gps_shift):.3f}")
    ax.set_ylabel("Frequency (Hz)")
    ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax.minorticks_on()
    ax.grid(which="major", color="black", linestyle=":", linewidth=0.8)

    label = "Likelihood" if key == "likelihood" else "Null"
    ax.set_title(
        f"{label} {statistics_map.total:3.0f} - "
        f"dt(ms) [{1000 * statistics_map.min_dt:.6g}:"
        f"{1000 * statistics_map.max_dt:.6g}] - "
        f"df(hz) [{statistics_map.min_df:.6g}:"
        f"{statistics_map.max_df:.6g}] - "
        f"npix {statistics_map.npix}",
        fontfamily="serif",
        fontstyle="italic",
        fontsize=12,
    )

    colorbar = fig.colorbar(mesh, ax=ax, pad=0.012, fraction=0.035)
    colorbar.locator = MaxNLocator(
        nbins=6,
        steps=[1, 2, 2.5, 5, 10],
    )
    colorbar.update_ticks()
    colorbar.ax.tick_params(length=3)
    fig.subplots_adjust(left=0.10, right=0.91, bottom=0.13, top=0.90)

    if filename is not None:
        fig.savefig(filename)
        logger.info("Plot %s saved to %s", key, filename)
        plt.close(fig)

    return fig
