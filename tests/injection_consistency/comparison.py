"""comparison.py — GPS-time matching, column-level statistics, and consistency
assertions for the pycWB e2e consistency test.

Adapted from ``examples/pycwb_cwb_consistency/compare_pyc_runs.py``.
"""
from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats

from .helpers import get_tolerance, per_ifo_columns

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# GPS-time matching
# ---------------------------------------------------------------------------

def match_triggers_by_gps(
    baseline_events: list[dict[str, Any]],
    new_df: pd.DataFrame,
    tol: float = 0.05,
) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Match baseline events to new-run triggers by GPS time.

    Parameters
    ----------
    baseline_events : list[dict]
        Baseline events loaded from ``baseline.json``.
    new_df : pd.DataFrame
        New-run Parquet catalog converted to DataFrame.
    tol : float
        GPS matching tolerance in seconds (default 0.05 s = 50 ms).

    Returns
    -------
    matched : list[tuple[dict, dict]]
        Pairs of (baseline_event, new_event_dict) that matched.
    unmatched_baseline : list[dict]
        Baseline events with no counterpart in the new run.
    unmatched_new : list[dict]
        New-run events with no counterpart in the baseline.
    """
    # Build GPS arrays
    base_times = np.array([e.get("gps_time", np.nan) for e in baseline_events], dtype=np.float64)
    new_times = new_df["gps_time"].to_numpy(dtype=np.float64)

    # Sort both by GPS time
    sort_base = np.argsort(base_times)
    sort_new = np.argsort(new_times)
    ts_base = base_times[sort_base]
    ts_new = new_times[sort_new]

    matched_base_idx: list[int] = []
    matched_new_idx: list[int] = []
    used_new: set[int] = set()

    for pos, orig_i in enumerate(sort_base):
        t = ts_base[pos]
        # Binary search for closest new-run GPS time
        ins = np.searchsorted(ts_new, t)
        best_j: int | None = None
        best_dt = np.inf
        for j in (ins - 1, ins):
            if 0 <= j < len(ts_new):
                dt = abs(ts_new[j] - t)
                orig_j = sort_new[j]
                if dt < best_dt and orig_j not in used_new:
                    best_dt = dt
                    best_j = orig_j
        if best_j is not None and best_dt <= tol:
            matched_base_idx.append(orig_i)
            matched_new_idx.append(best_j)
            used_new.add(best_j)

    all_base = set(range(len(baseline_events)))
    all_new = set(range(len(new_df)))
    unm_base_idx = sorted(all_base - set(matched_base_idx))
    unm_new_idx = sorted(all_new - set(matched_new_idx))

    # Build result pairs
    new_records = new_df.to_dict(orient="records")

    matched = [
        (baseline_events[i], new_records[j])
        for i, j in zip(matched_base_idx, matched_new_idx)
    ]
    unmatched_base = [baseline_events[i] for i in unm_base_idx]
    unmatched_new = [new_records[i] for i in unm_new_idx]

    return matched, unmatched_base, unmatched_new


# ---------------------------------------------------------------------------
# Column statistics
# ---------------------------------------------------------------------------

def _safe_float(val: Any) -> float | None:
    """Convert a value to float, returning None for non-numeric / NaN / Inf."""
    if val is None:
        return None
    try:
        v = float(val)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    except (TypeError, ValueError):
        return None


def compare_columns(
    matched_pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    columns: list[str],
    tolerances: dict[str, dict[str, float]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Compute per-column statistics for matched event pairs.

    Parameters
    ----------
    matched_pairs : list[tuple[dict, dict]]
        List of (baseline, new) dict pairs.
    columns : list[str]
        Column names to compare.
    tolerances : dict or None
        Tolerance dict; if None, uses defaults from ``helpers.get_tolerance``.

    Returns
    -------
    dict
        ``{column: {"n": int, "mean_base": float, "mean_new": float,
        "rms_diff": float, "max_abs_diff": float, "pearson_r": float,
        "passed": bool}}``
    """
    if tolerances is None:
        tolerances = {}

    results: dict[str, dict[str, Any]] = {}

    for col in columns:
        x_vals: list[float] = []
        y_vals: list[float] = []

        for base_ev, new_ev in matched_pairs:
            x = _safe_float(base_ev.get(col))
            y = _safe_float(new_ev.get(col))
            if x is not None and y is not None:
                x_vals.append(x)
                y_vals.append(y)

        if len(x_vals) < 2:
            results[col] = {
                "n": len(x_vals),
                "mean_base": float(np.mean(x_vals)) if x_vals else None,
                "mean_new": float(np.mean(y_vals)) if y_vals else None,
                "rms_diff": None,
                "max_abs_diff": None,
                "pearson_r": None,
                "passed": len(x_vals) > 0,
            }
            continue

        x_arr = np.array(x_vals, dtype=np.float64)
        y_arr = np.array(y_vals, dtype=np.float64)
        diff = y_arr - x_arr

        rms_diff = float(np.sqrt(np.mean(diff ** 2)))
        max_abs = float(np.max(np.abs(diff)))
        pearson = float(stats.pearsonr(x_arr, y_arr)[0]) if len(x_arr) > 1 else 1.0

        tol = tolerances.get(col, get_tolerance(col))
        abs_tol = tol.get("abs", 0.0)
        rel_tol = tol.get("rel", 0.0)
        mean_base = float(np.mean(x_arr))
        threshold = max(abs_tol, rel_tol * abs(mean_base)) if mean_base != 0 else abs_tol
        passed = rms_diff <= threshold or threshold == 0.0

        results[col] = {
            "n": len(x_arr),
            "mean_base": mean_base,
            "mean_new": float(np.mean(y_arr)),
            "rms_diff": rms_diff,
            "max_abs_diff": max_abs,
            "pearson_r": pearson,
            "passed": passed,
        }

    return results


# ---------------------------------------------------------------------------
# Assertion helper
# ---------------------------------------------------------------------------

def assert_consistency(
    matched_pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    columns: list[str],
    tolerances: dict[str, dict[str, float]] | None = None,
) -> None:
    """Assert that all columns are within their tolerance thresholds.

    Raises ``AssertionError`` with a descriptive message listing all
    failing columns and their RMS / threshold values.
    """
    stats = compare_columns(matched_pairs, columns, tolerances)

    failures: list[str] = []
    for col, s in stats.items():
        if not s["passed"]:
            tol = (tolerances or {}).get(col, get_tolerance(col))
            threshold = max(
                tol.get("abs", 0.0),
                tol.get("rel", 0.0) * abs(s["mean_base"] or 0.0),
            )
            failures.append(
                f"  {col}: RMS={s['rms_diff']:.4g} > threshold={threshold:.4g} "
                f"(mean_base={s['mean_base']:.4g}, mean_new={s['mean_new']:.4g}, "
                f"N={s['n']})"
            )

    if failures:
        msg = (
            f"Consistency check FAILED for {len(failures)}/{len(stats)} columns:\n"
            + "\n".join(failures)
        )
        raise AssertionError(msg)


# ---------------------------------------------------------------------------
# Load new-run Parquet
# ---------------------------------------------------------------------------

def load_new_catalog(path: str | Path, ifo_list: list[str] | None = None) -> pd.DataFrame:
    """Load a pycWB Parquet catalog and return a pandas DataFrame.

    Parameters
    ----------
    path : str or Path
        Path to the catalog Parquet file.
    ifo_list : list[str] or None
        IFO list; currently unused (for future schema validation).
    """
    table = pq.read_table(str(path))
    return table.to_pandas()
