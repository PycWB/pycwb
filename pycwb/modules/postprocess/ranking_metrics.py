"""Small ranking-statistic metrics used by postprocess reports."""

from __future__ import annotations

import numpy as np


def cumulative_event_rate(
    values: np.ndarray,
    livetime: float,
    binwidth: float = 0.05,
    *,
    thresholds: np.ndarray | None = None,
    comparison: str = ">=",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return cumulative event rate above each ranking threshold.

    Parameters
    ----------
    values:
        Ranking statistic values.
    livetime:
        Live time in seconds.
    binwidth:
        Ranking-statistic bin width.
    thresholds:
        Explicit threshold grid, e.g. the grid from a cWB report.
    comparison:
        ``>=`` (native default) or ``>`` (cWB report convention).

    Returns
    -------
    tuple
        ``thresholds, rates, xerr, yerr`` where ``rates`` are in
        ``1 / livetime`` units.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if comparison not in (">", ">="):
        raise ValueError("comparison must be '>' or '>='")
    if len(values) == 0 and thresholds is None:
        raise ValueError("No finite ranking values available")
    if not np.isfinite(livetime) or livetime <= 0:
        raise ValueError("livetime must be positive")
    if not np.isfinite(binwidth) or binwidth <= 0:
        raise ValueError("binwidth must be positive")

    if thresholds is None:
        x_min = float(values.min() - 0.5)
        x_max = float(values.max() + 0.5)
        n_bins = max(1, int((x_max - x_min) / binwidth))
        thresholds = x_min + np.arange(n_bins) * binwidth
    thresholds = np.asarray(thresholds, dtype=float)
    if thresholds.ndim != 1 or not np.all(np.isfinite(thresholds)):
        raise ValueError("thresholds must be a finite one-dimensional array")
    side = "left" if comparison == ">=" else "right"
    counts = len(values) - np.searchsorted(np.sort(values), thresholds, side=side)
    rates = counts / livetime
    xerr = np.zeros(len(thresholds))
    yerr = np.sqrt(counts) / livetime
    return thresholds, rates, xerr, yerr
