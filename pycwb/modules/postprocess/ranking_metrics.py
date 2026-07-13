"""Small ranking-statistic metrics used by postprocess reports."""

from __future__ import annotations

import numpy as np


def cumulative_event_rate(
    values: np.ndarray,
    livetime: float,
    binwidth: float = 0.05,
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

    Returns
    -------
    tuple
        ``thresholds, rates, xerr, yerr`` where ``rates`` are in
        ``1 / livetime`` units.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        raise ValueError("No finite ranking values available")
    if livetime <= 0:
        raise ValueError("livetime must be positive")
    if binwidth <= 0:
        raise ValueError("binwidth must be positive")

    x_min = float(values.min() - 0.5)
    x_max = float(values.max() + 0.5)
    n_bins = max(1, int((x_max - x_min) / binwidth))
    thresholds = x_min + np.arange(n_bins) * binwidth
    counts = len(values) - np.searchsorted(np.sort(values), thresholds, side="left")
    rates = counts / livetime
    xerr = np.zeros(n_bins)
    yerr = np.sqrt(counts) / livetime
    return thresholds, rates, xerr, yerr

