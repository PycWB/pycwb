"""Shared background rates for any catalog source."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa

from pycwb.modules.postprocess.lag_filters import recorded_zero_lag_count
from pycwb.modules.postprocess.ranking_metrics import cumulative_event_rate
from pycwb.post_production.action_spec import action_spec

# A Parquet path, a pandas DataFrame, or an Arrow table.
TableSource = str | Path | pd.DataFrame | pa.Table

logger = logging.getLogger(__name__)

# Lag selection belongs to trigger_selection; these former options are ignored.
_RETIRED_SELECTION_ARGUMENTS = ("exclude_zero_lag", "unshifted_job_ids")


def _frame(value):
    if isinstance(value, (str, Path)):
        return pd.read_parquet(value)
    if isinstance(value, pd.DataFrame):
        return value.copy()
    return value.to_pandas()


@action_spec(
    inputs=["triggers", "progress"],
    outputs=[],
    description="Compute background rates on a common threshold grid",
)
def process_background(
    triggers: TableSource,
    progress: TableSource,
    *,
    ranking_par: str = "rho",
    thresholds: Sequence[float] | np.ndarray | None = None,
    comparison: str = ">=",
    trigger_query: str | None = None,
    work_dir: str | Path = ".",
    **kwargs: Any,
) -> dict[str, Any]:
    """Process native or adapted tables/Parquet identically; rates are Hz.

    Triggers and exposure are used as given: this function applies no lag
    selection. Select background upstream, for example with
    ``trigger_selection`` and its ``triggers_file`` and ``progress_file``
    outputs. Every trigger needs a completed job/lag exposure row; a warning is
    logged when triggers carry no time or segment shift (zero lag).
    Event-quality cuts do not reduce exposure. Supply already veto-adjusted
    progress for time vetoes. A trained ranking can be supplied as a column;
    this function neither trains a model nor creates missing ranking values.
    """
    retired = [name for name in _RETIRED_SELECTION_ARGUMENTS if name in kwargs]
    if retired:
        logger.warning(
            "process_background ignores %s; it uses triggers and exposure as "
            "given. Select background upstream with trigger_selection.",
            ", ".join(retired),
        )

    def resolve(value):
        return Path(work_dir) / value if isinstance(value, (str, Path)) else value

    events, live = _frame(resolve(triggers)), _frame(resolve(progress))
    for name, frame, columns in (
        ("triggers", events, ["job_id", "lag_idx", ranking_par]),
        ("progress", live, ["job_id", "lag_idx", "livetime"]),
    ):
        missing = set(columns) - set(frame.columns)
        if missing:
            raise ValueError(f"{name} missing columns: {sorted(missing)}")
    if "status" in live:
        live = live[live.status == "completed"]
    if live.duplicated(["job_id", "lag_idx"]).any():
        raise ValueError("Duplicate job/lag exposure rows")
    exposure = live.livetime.to_numpy(dtype=float)
    if not np.all(np.isfinite(exposure)) or np.any(exposure < 0):
        raise ValueError("Livetime must be finite and nonnegative")
    keys = ["job_id", "lag_idx"]
    available = pd.MultiIndex.from_frame(live[keys])
    covered = pd.MultiIndex.from_frame(events[keys]).isin(available)
    if not covered.all():
        raise ValueError("Triggers without completed job/lag exposure")
    unshifted = recorded_zero_lag_count(events)
    if unshifted:
        logger.warning(
            "%d background triggers have no time or segment shift (zero lag); "
            "rates include them as given.", unshifted,
        )
    if trigger_query:
        events = events.query(trigger_query)
    values = events[ranking_par].to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Nonfinite ranking values in {ranking_par}")
    livetime = float(live.livetime.sum())
    grid, rate, _, error = cumulative_event_rate(
        values, livetime, thresholds=thresholds, comparison=comparison
    )
    curve = pd.DataFrame(
        {
            "threshold": grid,
            "count": np.rint(rate * livetime).astype(np.int64),
            "far_hz": rate,
            "far_error_hz": error,
        }
    )
    return {
        "triggers": events.reset_index(drop=True),
        "livetime": livetime,
        "n_triggers": len(events),
        "curve": curve,
        "comparison": comparison,
    }
