"""Fake-openbox postprocess report action."""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd

from pycwb.post_production.action_spec import action_spec
from pycwb.modules.postprocess import far
from pycwb.modules.postprocess import report_plots
from pycwb.modules.postprocess.lag_filters import (
    nonzero_lag_mask,
    try_unshifted_job_ids_from_catalog,
)

logger = logging.getLogger(__name__)


@action_spec(
    outputs=[],
    inputs=["catalog_file", "intervals_file"],
    description="Select fake-openbox FAR intervals and plot them with openbox-style significance",
)
def fake_openbox_report(
    work_dir: str,
    catalog_file: str,
    intervals_file: str,
    far_rho_data: Optional[dict] = None,
    ranking_par: str = "rho",
    output_dir: str = "public",
    ifo_order: Optional[list[str]] = None,
    fake_openbox_n: int = 3,
    fake_openbox_seed: int = 150914,
    exclude_zero_lag: bool = True,
    **kwargs,
) -> dict:
    """Select random FAR intervals and report them as fake openbox data."""
    def _resolve(p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(work_dir, p)

    cat_path = _resolve(catalog_file)
    intervals_path = _resolve(intervals_file)
    out_dir = _resolve(output_dir)
    os.makedirs(out_dir, exist_ok=True)

    far_rho_data = far.resolve_far_rho_data(far_rho_data, out_dir, kwargs)

    intervals = read_intervals_file(intervals_path)
    if intervals.empty:
        raise ValueError(f"No intervals available for fake openbox: {intervals_path}")
    required = {"shift_key", "lag_idx", "livetime"}
    missing = required - set(intervals.columns)
    if missing:
        raise KeyError(f"Fake openbox intervals missing columns: {sorted(missing)}")

    n_select = min(int(fake_openbox_n), len(intervals))
    selected_intervals = intervals.sample(n=n_select, random_state=int(fake_openbox_seed)).sort_values(
        ["shift_key", "lag_idx"],
    ).reset_index(drop=True)
    selected_keys = [
        (str(row["shift_key"]), int(row["lag_idx"]))
        for _, row in selected_intervals.iterrows()
    ]

    schema_columns = far.parquet_schema_names(cat_path)
    segment_cols = segment_lag_columns_for_config_order(cat_path, schema_columns, ifo_order=ifo_order)
    if not segment_cols:
        raise KeyError("Fake openbox matching requires trigger segment_lag_* columns")
    interval_shift_cols = interval_shift_columns(selected_intervals)

    columns = far.trigger_read_columns(cat_path, ranking_par, extra_columns=segment_cols)
    if "lag_idx" not in columns:
        raise KeyError("Fake openbox matching requires trigger lag_idx column")
    frames_by_key: dict[tuple[str, int], list[pd.DataFrame]] = {key: [] for key in selected_keys}
    trigger_unshifted_jobs = None
    if exclude_zero_lag:
        trigger_unshifted_jobs = try_unshifted_job_ids_from_catalog(cat_path)

    for chunk in far.iter_parquet_row_groups(cat_path, columns):
        if exclude_zero_lag:
            chunk = chunk[nonzero_lag_mask(chunk, unshifted_job_ids=trigger_unshifted_jobs)]
        if chunk.empty:
            continue
        fallback_shift_keys = None
        if not interval_shift_cols or len(interval_shift_cols) != len(segment_cols):
            fallback_shift_keys = shift_key_series(chunk, segment_cols)

        lag_values = pd.to_numeric(chunk["lag_idx"], errors="coerce")
        for _, interval in selected_intervals.iterrows():
            shift_key = str(interval["shift_key"])
            lag_idx = int(interval["lag_idx"])
            key = (shift_key, lag_idx)
            mask = lag_values.eq(lag_idx)
            if fallback_shift_keys is None:
                for segment_col, shift_col in zip(segment_cols, interval_shift_cols):
                    target = float(interval[shift_col])
                    values = pd.to_numeric(chunk[segment_col], errors="coerce").to_numpy(dtype=float)
                    mask = mask & pd.Series(np.isclose(values, target, rtol=0.0, atol=1e-12), index=chunk.index)
            else:
                mask = mask & fallback_shift_keys.eq(shift_key)
            selected = chunk[mask]
            if not selected.empty:
                selected = selected.copy()
                selected["shift_key"] = shift_key
                frames_by_key[key].append(selected)

    selected_path = os.path.join(out_dir, "fake_openbox_intervals.csv")
    selected_intervals.to_csv(selected_path, index=False)

    cols = far.trigger_report_columns(ranking_par, include_shift_key=True) + [
        "far_attached",
        "ifar_years",
        "significance",
        "p_value",
    ]

    reports = []
    combined_frames = []
    empty_columns = list(dict.fromkeys(columns + ["shift_key"]))
    for idx, interval in selected_intervals.iterrows():
        shift_key = str(interval["shift_key"])
        lag_idx = int(interval["lag_idx"])
        interval_livetime = float(interval["livetime"])
        interval_livetime_years = interval_livetime / 86400.0 / 365.25
        interval_frames = frames_by_key.get((shift_key, lag_idx), [])
        interval_df = (
            pd.concat(interval_frames, ignore_index=True)
            if interval_frames else pd.DataFrame(columns=empty_columns)
        )
        interval_df = far.attach_far_and_significance(interval_df, far_rho_data, ranking_par, interval_livetime)
        fake_id = f"fake_openbox_{idx + 1:02d}"
        plot_label = f"Fake openbox {idx + 1}: slag {shift_key}, lag {lag_idx}"
        interval_csv = os.path.join(out_dir, f"{fake_id}_triggers.csv")
        interval_cols = [col for col in cols if col in interval_df.columns]
        interval_df[interval_cols].to_csv(interval_csv, index=False)
        report_plots.plot_zero_lag(
            interval_df,
            ranking_par,
            interval_livetime_years,
            out_dir,
            output_prefix=fake_id,
            plot_label=plot_label,
            far_rho_data=far_rho_data,
        )
        interval_df = interval_df.copy()
        interval_df["fake_openbox_id"] = fake_id
        combined_frames.append(interval_df)
        reports.append({
            "id": fake_id,
            "shift_key": shift_key,
            "lag_idx": lag_idx,
            "livetime": interval_livetime,
            "livetime_years": float(interval_livetime_years),
            "n_triggers": int(len(interval_df)),
            "triggers_csv": interval_csv,
            "report_png": os.path.join(out_dir, f"{fake_id}_report.png"),
            "poisson_png": os.path.join(out_dir, f"{fake_id}_poisson.png"),
            "max_significance": float(interval_df["significance"].max()) if len(interval_df) > 0 else 0.0,
        })

    combined = pd.concat(combined_frames, ignore_index=True) if combined_frames else pd.DataFrame()
    csv_path = os.path.join(out_dir, "fake_openbox_triggers.csv")
    combined_cols = ["fake_openbox_id"] + [col for col in cols if col in combined.columns]
    combined[combined_cols].to_csv(csv_path, index=False)
    fake_livetime = float(selected_intervals["livetime"].sum())
    logger.info(
        "Fake openbox: %d reports, %d triggers, %.0f s combined live time",
        len(reports), len(combined), fake_livetime,
    )
    return {
        "fake_openbox_n_intervals": int(len(selected_intervals)),
        "fake_openbox_n": int(len(combined)),
        "livetime": fake_livetime,
        "livetime_years": float(fake_livetime / 86400.0 / 365.25),
        "intervals_csv": selected_path,
        "triggers_csv": csv_path,
        "reports": reports,
        "max_significance": max((item["max_significance"] for item in reports), default=0.0),
    }


def read_intervals_file(path: str) -> pd.DataFrame:
    """Read fake-openbox interval CSV/parquet artifacts."""
    if path.endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_parquet(path)


def interval_shift_columns(intervals: pd.DataFrame) -> list[str]:
    """Return interval shift columns ordered by numeric suffix."""
    def suffix_index(column: str) -> int:
        return int(column[len("shift_"):])

    return sorted(
        [
            col for col in intervals.columns
            if col.startswith("shift_") and col[len("shift_"):].isdigit()
        ],
        key=suffix_index,
    )


def segment_lag_columns_for_config_order(
    path: str,
    schema_columns: list[str],
    ifo_order: Optional[list[str]] = None,
) -> list[str]:
    """Return segment-lag columns ordered like config['ifo'] when available."""
    segment_cols = [col for col in schema_columns if col.startswith("segment_lag_")]
    if not segment_cols:
        return []

    ifos = ifo_order
    if not ifos:
        try:
            from pycwb.modules.catalog.catalog import Catalog

            config = Catalog.open(path).config
        except Exception:
            config = {}

        ifos = config.get("ifo") if isinstance(config, dict) else None
    if not ifos:
        return segment_cols

    by_ifo = {
        col[len("segment_lag_"):]: col
        for col in segment_cols
    }
    ordered = [by_ifo[ifo] for ifo in ifos if ifo in by_ifo]
    ordered.extend(col for col in segment_cols if col not in ordered)
    return ordered


def shift_key_series(df: pd.DataFrame, segment_cols: list[str]) -> pd.Series:
    """Build cWB-style shift keys for one already-pruned trigger chunk."""
    if not segment_cols:
        raise KeyError("Fake openbox matching requires trigger segment_lag_* columns")

    parts = []
    for col in segment_cols:
        values = pd.to_numeric(df[col], errors="coerce")
        parts.append(values.map(format_shift_value).astype(str))

    out = parts[0]
    for part in parts[1:]:
        out = out + "," + part
    return out


def format_shift_value(value) -> str:
    """Format one shift value for a cWB-style shift key."""
    if pd.isna(value):
        return "nan"
    return f"{float(value):.12g}"


def shift_key(shift: tuple[float, ...]) -> str:
    """Format a shift tuple as a cWB-style shift key."""
    return ",".join(f"{value:.12g}" for value in shift)

