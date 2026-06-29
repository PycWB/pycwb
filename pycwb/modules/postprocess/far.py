"""FAR, trigger ranking, and report table helpers."""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

import numpy as np
import pandas as pd

from pycwb.post_production.action_spec import action_spec
from pycwb.modules.postprocess import report_plots
from pycwb.modules.postprocess.lag_filters import (
    nonzero_lag_mask,
    try_unshifted_job_ids_from_catalog,
    unshifted_job_ids_from_progress,
    zero_lag_mask,
)

logger = logging.getLogger(__name__)

PARQUET_BATCH_SIZE = 100_000
LOUDEST_BACKGROUND_N = 10

BASE_TRIGGER_COLUMNS = [
    "id",
    "job_id",
    "lag_idx",
    "ifar",
    "gps_time",
    "net_cc",
    "likelihood",
    "coherent_energy",
]


@action_spec(
    outputs=[],
    inputs=["catalog_file", "progress_file", "job_ids_file", "livetime"],
    description="Compute FAR vs ranking parameter and save plots",
)
def far_rho_plot(
    work_dir: str,
    catalog_file: str,
    progress_file: str,
    job_ids_file: Optional[str] = None,
    livetime: Optional[float] = None,
    ranking_par: str = "rho",
    exclude_zero_lag: bool = True,
    bin_size: float = 0.1,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    output_dir: str = "public",
    **kwargs,
) -> dict:
    """Compute FAR vs ranking parameter and save report artifacts."""
    def _resolve(p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(work_dir, p)

    cat_path = _resolve(catalog_file)
    prog_path = _resolve(progress_file)
    jobs_path = _resolve(job_ids_file) if job_ids_file else None
    out_dir = _resolve(output_dir)
    trigger_unshifted_jobs = try_unshifted_job_ids_from_catalog(cat_path)
    progress_unshifted_jobs = unshifted_jobs_for_progress(prog_path)

    columns = available_columns(
        cat_path,
        [ranking_par] + lag_filter_columns(cat_path, include_segment_shift=True),
        required=[ranking_par],
    )
    loudest_columns = trigger_read_columns(cat_path, ranking_par)

    n_triggers = 0
    n_valid = 0
    bin_min: Optional[float] = None
    bin_max: Optional[float] = None
    for chunk in iter_parquet_row_groups(cat_path, columns):
        if exclude_zero_lag:
            chunk = chunk[nonzero_lag_mask(chunk, unshifted_job_ids=trigger_unshifted_jobs)]
        n_triggers += len(chunk)
        values = pd.to_numeric(chunk[ranking_par], errors="coerce").dropna().to_numpy()
        if len(values) == 0:
            continue
        n_valid += len(values)
        chunk_min = float(values.min())
        chunk_max = float(values.max())
        bin_min = chunk_min if bin_min is None else min(bin_min, chunk_min)
        bin_max = chunk_max if bin_max is None else max(bin_max, chunk_max)

    if vmin is not None:
        bin_min = float(vmin)
    if vmax is not None:
        bin_max = float(vmax)

    if bin_min is None or bin_max is None:
        raise ValueError(f"No valid '{ranking_par}' values found in {cat_path}")
    logger.info("Triggers: %d total, %d with valid %s", n_triggers, n_valid, ranking_par)
    logger.info("Bin range: [%s, %s], bin_size=%s", bin_min, bin_max, bin_size)

    if livetime is None:
        if jobs_path is None:
            raise ValueError("far_rho_plot requires either livetime or job_ids_file")
        livetime = sum_progress_livetime(
            prog_path,
            zero_lag=False if exclude_zero_lag else None,
            job_ids=read_job_ids(jobs_path),
            unshifted_job_ids=progress_unshifted_jobs,
        )
    else:
        livetime = float(livetime)
    livetime_years = livetime / 86400.0 / 365.25
    logger.info("Live time: %.0f s = %.2f yr", livetime, livetime_years)

    bins = np.arange(bin_min, bin_max + bin_size, bin_size)
    if len(bins) < 2:
        bins = np.array([bin_min, bin_min + bin_size])
    hist = np.zeros(len(bins) - 1, dtype=np.int64)
    loudest_triggers: Optional[pd.DataFrame] = None
    for chunk in iter_parquet_row_groups(cat_path, loudest_columns):
        if exclude_zero_lag:
            chunk = chunk[nonzero_lag_mask(chunk, unshifted_job_ids=trigger_unshifted_jobs)]
        values = pd.to_numeric(chunk[ranking_par], errors="coerce").dropna().to_numpy()
        if len(values) > 0:
            hist += np.histogram(values, bins=bins)[0]
        loudest_triggers = update_loudest_triggers(loudest_triggers, chunk, ranking_par)

    cum_hist = np.cumsum(hist[::-1])[::-1]
    far_values = cum_hist / max(livetime_years, 1e-10)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    data = {
        "bins": bin_centers.tolist(),
        "far": far_values.tolist(),
        "n_events": hist.tolist(),
        "cum_events": cum_hist.tolist(),
        "ranking_par": ranking_par,
        "livetime": livetime,
        "livetime_years": livetime_years,
    }

    os.makedirs(out_dir, exist_ok=True)
    report_plots.plot_far_rho(data, out_dir)
    report_plots.plot_n_events(data, out_dir)
    loudest_csv, loudest_n = write_loudest_background_triggers(
        loudest_triggers, out_dir, ranking_par, data,
    )

    json_path = os.path.join(out_dir, "far_rho.json")
    json_data = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in data.items()}
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    logger.info("FAR data → %s", json_path)
    logger.info("Loudest BKG triggers → %s", loudest_csv)

    return {
        "far_rho": data,
        "loudest_background_triggers": {
            "csv": loudest_csv,
            "n": loudest_n,
        },
    }


def unshifted_jobs_for_progress(progress_path: str) -> Optional[set[int]]:
    """Return unshifted job IDs from a progress parquet when available."""
    try:
        return unshifted_job_ids_from_progress(progress_path)
    except (FileNotFoundError, ValueError, KeyError):
        logger.warning(
            "Could not read unshifted job metadata for progress file %s; "
            "falling back to row shift columns when available",
            progress_path,
        )
        return None


def parquet_schema_names(path: str) -> list[str]:
    """Return parquet schema column names without loading the table."""
    import pyarrow.parquet as pq

    return list(pq.ParquetFile(path).schema.names)


def available_columns(path: str, requested: list[str], required: Optional[list[str]] = None) -> list[str]:
    """Return requested parquet columns that exist, validating required names."""
    names = parquet_schema_names(path)
    available = set(names)
    missing = [col for col in (required or []) if col not in available]
    if missing:
        raise KeyError(f"Parquet file {path} missing required columns: {missing}")

    columns: list[str] = []
    for col in requested:
        if col in available and col not in columns:
            columns.append(col)
    return columns


def lag_filter_columns(path: str, include_segment_shift: bool = True) -> list[str]:
    """Return available lag/shift columns needed for zero-lag classification."""
    columns: list[str] = []
    for col in parquet_schema_names(path):
        if col in {"job_id", "lag_idx", "lag", "time_lag"} or col.startswith("time_lag_"):
            columns.append(col)
        elif include_segment_shift and (
            col in {"segment_lag", "segment_shift", "shift"}
            or col.startswith("segment_lag_")
            or col.startswith("segment_shift_")
            or col.startswith("shift_")
        ):
            columns.append(col)
    return columns


def iter_parquet_row_groups(path: str, columns: list[str]):
    """Yield pandas DataFrames from parquet batches with Arrow memory release."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=PARQUET_BATCH_SIZE, columns=columns, use_threads=True):
        df = batch.to_pandas(split_blocks=True, self_destruct=True)
        yield df
        del df, batch
        pa.default_memory_pool().release_unused()


def read_job_ids(path: Optional[str]) -> Optional[set[int]]:
    """Read a newline-delimited job-id file."""
    if path is None:
        return None
    with open(path) as f:
        return {int(line.strip()) for line in f if line.strip()}


def trigger_report_columns(ranking_par: str, include_shift_key: bool = False) -> list[str]:
    """Return report columns for trigger CSV outputs."""
    columns = list(BASE_TRIGGER_COLUMNS)
    if ranking_par not in columns:
        columns.insert(3, ranking_par)
    if include_shift_key:
        columns.insert(3, "shift_key")
    return columns


def trigger_read_columns(path: str, ranking_par: str, extra_columns: Optional[list[str]] = None) -> list[str]:
    """Return available columns needed to read trigger report chunks."""
    requested = trigger_report_columns(ranking_par)
    requested.extend(lag_filter_columns(path, include_segment_shift=True))
    requested.extend(extra_columns or [])
    return available_columns(path, requested, required=[ranking_par])


def update_loudest_triggers(
    current: Optional[pd.DataFrame],
    chunk: pd.DataFrame,
    ranking_par: str,
    n: int = LOUDEST_BACKGROUND_N,
) -> Optional[pd.DataFrame]:
    """Update the top-n loudest trigger table with one parquet chunk."""
    if chunk.empty:
        return current

    ranked = chunk.copy()
    ranked[ranking_par] = pd.to_numeric(ranked[ranking_par], errors="coerce")
    ranked = ranked.dropna(subset=[ranking_par])
    if ranked.empty:
        return current

    ranked = ranked.nlargest(n, ranking_par)
    if current is not None and not current.empty:
        ranked = pd.concat([current, ranked], ignore_index=True)
        ranked = ranked.nlargest(n, ranking_par)
    return ranked.reset_index(drop=True)


def write_loudest_background_triggers(
    df: Optional[pd.DataFrame],
    out_dir: str,
    ranking_par: str,
    far_rho_data: Optional[dict] = None,
) -> tuple[str, int]:
    """Write the loudest-background trigger CSV."""
    csv_path = os.path.join(out_dir, "loudest_background_triggers.csv")
    if df is None or df.empty:
        pd.DataFrame(columns=trigger_report_columns(ranking_par)).to_csv(csv_path, index=False)
        return csv_path, 0

    df = df.sort_values(ranking_par, ascending=False).reset_index(drop=True).copy()
    df.insert(0, "bkg_rank", np.arange(1, len(df) + 1, dtype=int))
    if far_rho_data is not None:
        bins = np.asarray(far_rho_data["bins"], dtype=float)
        far_values = np.asarray(far_rho_data["far"], dtype=float)
        rho_vals = pd.to_numeric(df[ranking_par], errors="coerce").to_numpy(dtype=float)
        idx = np.searchsorted(bins, rho_vals, side="right") - 1
        idx = np.clip(idx, 0, len(far_values) - 1)
        attached_far = far_values[idx]
        df["far_attached"] = attached_far
        df["ifar_years"] = 1.0 / np.maximum(attached_far, 1e-30)

    cols = [
        col for col in ["bkg_rank", *trigger_report_columns(ranking_par, include_shift_key=True)]
        if col in df.columns
    ]
    for col in ["far_attached", "ifar_years"]:
        if col in df.columns and col not in cols:
            cols.append(col)
    extra_cols = [
        col for col in df.columns
        if (
            col.startswith(("time_lag_", "segment_lag_", "segment_shift_", "shift_"))
            or col in {"time_lag", "segment_lag", "segment_shift", "shift"}
        ) and col not in cols
    ]
    cols.extend(extra_cols)
    df[cols].to_csv(csv_path, index=False)
    return csv_path, len(df)


def sum_progress_livetime(
    progress_path: str,
    *,
    zero_lag: Optional[bool],
    job_ids: Optional[set[int]] = None,
    unshifted_job_ids: Optional[set[int]] = None,
) -> float:
    """Sum completed progress livetime with optional zero/nonzero-lag filtering."""
    requested = ["job_id", "lag_idx", "lag", "livetime", "status"]
    requested.extend(lag_filter_columns(progress_path, include_segment_shift=True))
    columns = available_columns(progress_path, requested, required=["livetime"])
    total = 0.0

    for prog in iter_parquet_row_groups(progress_path, columns):
        if job_ids is not None and "job_id" in prog.columns:
            prog = prog[prog["job_id"].isin(job_ids)]
        if "status" in prog.columns:
            prog = prog[prog["status"] == "completed"]
        if prog.empty:
            continue

        if zero_lag is None:
            selected = prog
        else:
            mask = zero_lag_mask(prog, unshifted_job_ids=unshifted_job_ids)
            selected = prog[mask if zero_lag else ~mask]
        total += float(pd.to_numeric(selected["livetime"], errors="coerce").sum())

    return total


def resolve_far_rho_data(far_rho_data: Optional[dict], out_dir: str, kwargs: dict) -> dict:
    """Resolve FAR/rho data from an argument, context key, or output JSON."""
    if far_rho_data is None:
        far_rho_data = kwargs.get("far_rho")
    if isinstance(far_rho_data, dict) and "far_rho" in far_rho_data and "bins" not in far_rho_data:
        far_rho_data = far_rho_data["far_rho"]
    if far_rho_data is None:
        json_path = os.path.join(out_dir, "far_rho.json")
        if os.path.exists(json_path):
            with open(json_path) as f:
                far_rho_data = json.load(f)
    if far_rho_data is None:
        raise ValueError("far_rho_data not provided and far_rho.json not found")
    return far_rho_data


def attach_far_and_significance(
    df: pd.DataFrame,
    far_rho_data: dict,
    ranking_par: str,
    livetime_seconds: float,
) -> pd.DataFrame:
    """Attach FAR, IFAR, p-value, and significance columns to trigger rows."""
    df = df.copy()
    bins = np.asarray(far_rho_data["bins"], dtype=float)
    far_values = np.asarray(far_rho_data["far"], dtype=float)
    if len(bins) == 0 or len(far_values) == 0:
        raise ValueError("far_rho_data must contain non-empty 'bins' and 'far' arrays")

    rho_vals = pd.to_numeric(df[ranking_par], errors="coerce").to_numpy(dtype=float)
    idx = np.searchsorted(bins, rho_vals, side="right") - 1
    idx = np.clip(idx, 0, len(far_values) - 1)
    attached_far = far_values[idx].copy()

    far_nonzero = far_values[far_values > 0]
    if len(far_nonzero) > 0:
        far_min = far_nonzero.min()
        attached_far = np.maximum(attached_far, far_min)

    df["far_attached"] = attached_far
    df["ifar_years"] = 1.0 / np.maximum(attached_far, 1e-30)

    from scipy.stats import poisson

    expected = attached_far * livetime_seconds / 86400 / 365.25
    p_values = 1.0 - poisson.cdf(0, expected)
    df["p_value"] = p_values
    df["significance"] = -np.log10(np.maximum(p_values, 1e-300))
    return df
