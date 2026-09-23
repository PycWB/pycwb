"""Shared background selection and cumulative rates for any catalog source."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pycwb.modules.postprocess.lag_filters import nonzero_lag_mask
from pycwb.modules.postprocess.ranking_metrics import cumulative_event_rate
from pycwb.post_production.action_spec import action_spec


def _frame(value):
    if isinstance(value, (str, Path)):
        return pd.read_parquet(value)
    if isinstance(value, pd.DataFrame):
        return value.copy()
    return value.to_pandas()


@action_spec(
    inputs=["triggers", "progress"],
    outputs=[],
    description="Select background and compute rates on a common threshold grid",
)
def process_background(
    triggers,
    progress,
    *,
    ranking_par="rho",
    thresholds=None,
    comparison=">=",
    trigger_query=None,
    exclude_zero_lag=True,
    unshifted_job_ids=None,
    work_dir=".",
    **kwargs,
):
    """Process native or adapted tables/Parquet identically; rates are Hz.

    Apply the same completed job/lag exposure selection to events and progress.
    Event-quality cuts do not reduce exposure. Supply already veto-adjusted
    progress for time vetoes. A trained ranking can be supplied as a column;
    this function neither trains a model nor creates missing ranking values.
    """

    def resolve(value):
        return Path(work_dir) / value if isinstance(value, (str, Path)) else value

    if unshifted_job_ids is None:
        source = resolve(triggers)
        if isinstance(source, Path):
            from pycwb.modules.postprocess.lag_filters import (
                try_unshifted_job_ids_from_catalog,
            )

            unshifted_job_ids = try_unshifted_job_ids_from_catalog(str(source))
        elif hasattr(source, "schema"):
            metadata = source.schema.metadata or {}
            jobs = json.loads(metadata.get(b"jobs", b"[]"))
            if jobs:
                unshifted_job_ids = {
                    int(job["index"])
                    for job in jobs
                    if all(abs(float(x)) <= 1e-12 for x in (job.get("shift") or []))
                }
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
    if exclude_zero_lag:
        live = live[nonzero_lag_mask(live, unshifted_job_ids)]
        events = events[nonzero_lag_mask(events, unshifted_job_ids)]
    # Guard against mismatched shift metadata between the two tables.
    available = pd.MultiIndex.from_frame(live[keys])
    if not pd.MultiIndex.from_frame(events[keys]).isin(available).all():
        raise ValueError("Trigger/progress zero-lag selection disagrees")
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
