"""Post-production reports — workflow-compatible FAR & zero-lag plots.

Works directly with filtered parquet catalogs (pandas DataFrames) and
progress files, avoiding Catalog-schema dependencies.

Workflow actions
----------------
``postprocess.report.far_rho_plot``
    Compute FAR vs ranking parameter from a BKG catalog and plot.

``postprocess.report.zero_lag_report``
    Compute zero-lag significance and plot triggers with FAR attached.

``postprocess.report.standard_background_report``
    Composite action that calls the smaller background report actions and
    returns their summaries.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

import numpy as np
import pandas as pd

from pycwb.post_production.action_spec import action_spec
from pycwb.modules.postprocess.lag_filters import (
    nonzero_lag_mask,
    try_unshifted_job_ids_from_catalog,
    unshifted_job_ids_from_progress,
    zero_lag_mask,
)

logger = logging.getLogger(__name__)


_PARQUET_BATCH_SIZE = 100_000

_BASE_TRIGGER_COLUMNS = [
    "id",
    "job_id",
    "lag_idx",
    "ifar",
    "gps_time",
    "net_cc",
    "likelihood",
    "coherent_energy",
]


def _unshifted_jobs_for_progress(progress_path: str) -> Optional[set[int]]:
    try:
        return unshifted_job_ids_from_progress(progress_path)
    except (FileNotFoundError, ValueError, KeyError):
        logger.warning(
            "Could not read unshifted job metadata for progress file %s; "
            "falling back to row shift columns when available",
            progress_path,
        )
        return None


def _parquet_schema_names(path: str) -> list[str]:
    import pyarrow.parquet as pq

    return list(pq.ParquetFile(path).schema.names)


def _available_columns(path: str, requested: list[str], required: Optional[list[str]] = None) -> list[str]:
    names = _parquet_schema_names(path)
    available = set(names)
    missing = [col for col in (required or []) if col not in available]
    if missing:
        raise KeyError(f"Parquet file {path} missing required columns: {missing}")

    columns: list[str] = []
    for col in requested:
        if col in available and col not in columns:
            columns.append(col)
    return columns


def _lag_filter_columns(path: str, include_segment_shift: bool = True) -> list[str]:
    columns: list[str] = []
    for col in _parquet_schema_names(path):
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


def _iter_parquet_row_groups(path: str, columns: list[str]):
    import pyarrow.parquet as pq
    import pyarrow as pa

    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=_PARQUET_BATCH_SIZE, columns=columns, use_threads=True):
        df = batch.to_pandas(split_blocks=True, self_destruct=True)
        yield df
        del df, batch
        pa.default_memory_pool().release_unused()


def _read_job_ids(path: Optional[str]) -> Optional[set[int]]:
    if path is None:
        return None
    with open(path) as f:
        return {int(line.strip()) for line in f if line.strip()}


def _trigger_report_columns(ranking_par: str, include_shift_key: bool = False) -> list[str]:
    columns = list(_BASE_TRIGGER_COLUMNS)
    if ranking_par not in columns:
        columns.insert(3, ranking_par)
    if include_shift_key:
        columns.insert(3, "shift_key")
    return columns


def _trigger_read_columns(path: str, ranking_par: str, extra_columns: Optional[list[str]] = None) -> list[str]:
    requested = _trigger_report_columns(ranking_par)
    requested.extend(_lag_filter_columns(path, include_segment_shift=True))
    requested.extend(extra_columns or [])
    return _available_columns(path, requested, required=[ranking_par])


def _sum_progress_livetime(
    progress_path: str,
    *,
    zero_lag: Optional[bool],
    job_ids: Optional[set[int]] = None,
    unshifted_job_ids: Optional[set[int]] = None,
) -> float:
    requested = ["job_id", "lag_idx", "lag", "livetime", "status"]
    requested.extend(_lag_filter_columns(progress_path, include_segment_shift=True))
    columns = _available_columns(progress_path, requested, required=["livetime"])
    total = 0.0

    for prog in _iter_parquet_row_groups(progress_path, columns):
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


def _resolve_far_rho_data(far_rho_data: Optional[dict], out_dir: str, kwargs: dict) -> dict:
    if far_rho_data is None:
        far_rho_data = kwargs.get("far_rho")
    # Unwrap if double-wrapped (output_alias stores {far_rho: {...}})
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


# ---------------------------------------------------------------------------
# far_rho_plot
# ---------------------------------------------------------------------------

@action_spec(
    outputs=[],
    inputs=['catalog_file', 'progress_file', 'job_ids_file', 'livetime'],
    description='Compute FAR vs ranking parameter and save plots',
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
    output_dir: str = "public",
    **kwargs,
) -> dict:
    """Compute FAR vs ranking parameter and save plots.

    Parameters
    ----------
    work_dir : str
        Base directory.
    catalog_file : str
        Path to BKG catalog parquet.
    progress_file : str
        Path to progress parquet.
    job_ids_file : str, optional
        Path to job list file. Used to compute live time only when
        ``livetime`` is not provided.
    livetime : float, optional
        Live time in seconds. Preferred for interval-based splits because a
        job list alone cannot describe selected ``(shift, lag_idx)`` rows.
    ranking_par : str
        Column name for ranking (default ``"rho"``).
    exclude_zero_lag : bool
        Exclude physically unshifted zero-lag rows from live time.
    bin_size : float
        Histogram bin size for ranking parameter.
    output_dir : str
        Directory for output plots (relative to *work_dir*).

    Returns
    -------
    dict
        ``far_rho`` data with keys ``bins``, ``far``, ``n_events``, ``livetime_years``.
    """
    def _resolve(p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(work_dir, p)

    cat_path = _resolve(catalog_file)
    prog_path = _resolve(progress_file)
    jobs_path = _resolve(job_ids_file) if job_ids_file else None
    out_dir = _resolve(output_dir)
    trigger_unshifted_jobs = try_unshifted_job_ids_from_catalog(cat_path)
    progress_unshifted_jobs = _unshifted_jobs_for_progress(prog_path)

    # ── Stream trigger ranking values ────────────────────────────────────
    columns = _available_columns(
        cat_path,
        [ranking_par] + _lag_filter_columns(cat_path, include_segment_shift=True),
        required=[ranking_par],
    )

    n_triggers = 0
    n_valid = 0
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    for chunk in _iter_parquet_row_groups(cat_path, columns):
        if exclude_zero_lag:
            chunk = chunk[nonzero_lag_mask(chunk, unshifted_job_ids=trigger_unshifted_jobs)]
        n_triggers += len(chunk)
        values = pd.to_numeric(chunk[ranking_par], errors="coerce").dropna().to_numpy()
        if len(values) == 0:
            continue
        n_valid += len(values)
        chunk_min = float(values.min())
        chunk_max = float(values.max())
        vmin = chunk_min if vmin is None else min(vmin, chunk_min)
        vmax = chunk_max if vmax is None else max(vmax, chunk_max)

    if vmin is None or vmax is None:
        raise ValueError(f"No valid '{ranking_par}' values found in {cat_path}")
    logger.info("Triggers: %d total, %d with valid %s", n_triggers, n_valid, ranking_par)

    # ── Live time ────────────────────────────────────────────────────────
    if livetime is None:
        if jobs_path is None:
            raise ValueError("far_rho_plot requires either livetime or job_ids_file")
        livetime = _sum_progress_livetime(
            prog_path,
            zero_lag=False if exclude_zero_lag else None,
            job_ids=_read_job_ids(jobs_path),
            unshifted_job_ids=progress_unshifted_jobs,
        )
    else:
        livetime = float(livetime)
    livetime_years = livetime / 86400.0 / 365.25
    logger.info("Live time: %.0f s = %.2f yr", livetime, livetime_years)

    # ── Histogram & FAR ──────────────────────────────────────────────────
    bins = np.arange(vmin, vmax + bin_size, bin_size)
    if len(bins) < 2:
        bins = np.array([vmin, vmin + bin_size])
    hist = np.zeros(len(bins) - 1, dtype=np.int64)
    for chunk in _iter_parquet_row_groups(cat_path, columns):
        if exclude_zero_lag:
            chunk = chunk[nonzero_lag_mask(chunk, unshifted_job_ids=trigger_unshifted_jobs)]
        values = pd.to_numeric(chunk[ranking_par], errors="coerce").dropna().to_numpy()
        if len(values) > 0:
            hist += np.histogram(values, bins=bins)[0]
    # Cumulative from high to low
    cum_hist = np.cumsum(hist[::-1])[::-1]
    far = cum_hist / max(livetime_years, 1e-10)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    data = {
        "bins": bin_centers.tolist(),
        "far": far.tolist(),
        "n_events": hist.tolist(),
        "cum_events": cum_hist.tolist(),
        "ranking_par": ranking_par,
        "livetime": livetime,
        "livetime_years": livetime_years,
    }

    # ── Plots ────────────────────────────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)
    plot_far_rho(data, out_dir)
    plot_n_events(data, out_dir)

    # Save JSON
    json_path = os.path.join(out_dir, "far_rho.json")
    # Convert numpy types for JSON
    json_data = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in data.items()}
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    logger.info("FAR data → %s", json_path)

    return {"far_rho": data}


# ---------------------------------------------------------------------------
# zero_lag_report
# ---------------------------------------------------------------------------

@action_spec(
    outputs=[],
    inputs=['catalog_file', 'progress_file', 'job_ids_file'],
    description='Compute zero-lag significance and plot triggers with FAR',
)
def zero_lag_report(
    work_dir: str,
    catalog_file: str,
    progress_file: str,
    job_ids_file: Optional[str] = None,
    far_rho_data: Optional[dict] = None,
    ranking_par: str = "rho",
    output_dir: str = "public",
    public_alerts_file: Optional[str] = None,
    public_alert_time_window: float = 1.0,
    **kwargs,
) -> dict:
    """Plot zero-lag triggers with FAR values and Poisson significance.

    Parameters
    ----------
    work_dir : str
        Base directory.
    catalog_file : str
        Path to zero-lag BKG catalog parquet.
    progress_file : str
        Path to progress parquet.
    job_ids_file : str, optional
        Optional job list for a restricted zero-lag slice. If omitted, all
        physically unshifted ``lag_idx == 0`` jobs in the progress/catalog
        are used.
    far_rho_data : dict, optional
        FAR data from :func:`far_rho_plot`.  If not provided, reads from
        ``kwargs["far_rho"]`` or ``{output_dir}/far_rho.json``.
    ranking_par : str
        Ranking column name.
    output_dir : str
        Output directory.
    public_alerts_file : str, optional
        Two-column public alert table: ``candidate_id gps_time``.
    public_alert_time_window : float
        Maximum absolute GPS-time difference, in seconds, for matching a
        public alert to a zero-lag trigger.

    Returns
    -------
    dict
        ``zero_lag`` trigger info with FAR attached.
    """
    def _resolve(p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(work_dir, p)

    cat_path = _resolve(catalog_file)
    prog_path = _resolve(progress_file)
    jobs_path = _resolve(job_ids_file) if job_ids_file else None
    out_dir = _resolve(output_dir)
    trigger_unshifted_jobs = try_unshifted_job_ids_from_catalog(cat_path)
    progress_unshifted_jobs = _unshifted_jobs_for_progress(prog_path)

    # ── Resolve FAR data ─────────────────────────────────────────────────
    far_rho_data = _resolve_far_rho_data(far_rho_data, out_dir, kwargs)

    # ── Stream zero-lag triggers ─────────────────────────────────────────
    job_ids = _read_job_ids(jobs_path)
    columns = _trigger_read_columns(cat_path, ranking_par)
    frames = []
    for chunk in _iter_parquet_row_groups(cat_path, columns):
        if job_ids is not None and "job_id" in chunk.columns:
            chunk = chunk[chunk["job_id"].isin(job_ids)]
        if chunk.empty:
            continue
        mask = zero_lag_mask(chunk, unshifted_job_ids=trigger_unshifted_jobs)
        selected = chunk[mask]
        if not selected.empty:
            frames.append(selected.copy())
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
    logger.info("Zero-lag triggers (unshifted): %d", len(df))
    logger.info("Zero-lag triggers: %d", len(df))

    # ── Live time ────────────────────────────────────────────────────────
    zl_livetime = _sum_progress_livetime(
        prog_path,
        zero_lag=True,
        job_ids=job_ids,
        unshifted_job_ids=progress_unshifted_jobs,
    )
    zl_livetime_years = zl_livetime / 86400.0 / 365.25
    logger.info("Zero-lag live time: %.0f s = %.4f yr", zl_livetime, zl_livetime_years)

    df = _attach_far_and_significance(df, far_rho_data, ranking_par, zl_livetime)

    # ── Plot ─────────────────────────────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)
    known_candidates = {}
    if public_alerts_file:
        known_candidates = load_public_alert_candidates(
            df,
            _resolve(public_alerts_file),
            public_alert_time_window,
        )
    plot_zero_lag(
        df,
        ranking_par,
        zl_livetime_years,
        out_dir,
        known_candidates=known_candidates,
    )

    # Save CSV
    csv_path = os.path.join(out_dir, "zero_lag_triggers.csv")
    cols = [c for c in ["id", "job_id", "lag_idx", ranking_par, "ifar", "far_attached",
                         "ifar_years", "significance", "p_value", "gps_time",
                         "net_cc", "likelihood", "coherent_energy"]
            if c in df.columns]
    df[cols].to_csv(csv_path, index=False)
    logger.info("Zero-lag table → %s", csv_path)

    return {
        "zero_lag_n": len(df),
        "livetime_years": float(zl_livetime_years),
        "max_significance": float(df["significance"].max()) if len(df) > 0 else 0.0,
        "known_candidate_n": len(known_candidates),
    }


@action_spec(
    outputs=[],
    inputs=['catalog_file', 'intervals_file'],
    description='Select fake-openbox FAR intervals and plot them with openbox-style significance',
)
def fake_openbox_report(
    work_dir: str,
    catalog_file: str,
    intervals_file: str,
    far_rho_data: Optional[dict] = None,
    ranking_par: str = "rho",
    output_dir: str = "public",
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

    far_rho_data = _resolve_far_rho_data(far_rho_data, out_dir, kwargs)

    intervals = _read_intervals_file(intervals_path)
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

    schema_columns = _parquet_schema_names(cat_path)
    segment_cols = [col for col in schema_columns if col.startswith("segment_lag_")]
    if not segment_cols:
        raise KeyError("Fake openbox matching requires trigger segment_lag_* columns")
    interval_shift_cols = _interval_shift_columns(selected_intervals)

    columns = _trigger_read_columns(cat_path, ranking_par, extra_columns=segment_cols)
    if "lag_idx" not in columns:
        raise KeyError("Fake openbox matching requires trigger lag_idx column")
    frames_by_key: dict[tuple[str, int], list[pd.DataFrame]] = {key: [] for key in selected_keys}
    trigger_unshifted_jobs = None
    if exclude_zero_lag:
        trigger_unshifted_jobs = try_unshifted_job_ids_from_catalog(cat_path)

    for chunk in _iter_parquet_row_groups(cat_path, columns):
        if exclude_zero_lag:
            chunk = chunk[nonzero_lag_mask(chunk, unshifted_job_ids=trigger_unshifted_jobs)]
        if chunk.empty:
            continue
        fallback_shift_keys = None
        if not interval_shift_cols or len(interval_shift_cols) != len(segment_cols):
            fallback_shift_keys = _shift_key_series(chunk, segment_cols)

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

    cols = _trigger_report_columns(ranking_par, include_shift_key=True) + [
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
        interval_df = _attach_far_and_significance(interval_df, far_rho_data, ranking_par, interval_livetime)
        fake_id = f"fake_openbox_{idx + 1:02d}"
        plot_label = f"Fake openbox {idx + 1}: slag {shift_key}, lag {lag_idx}"
        interval_csv = os.path.join(out_dir, f"{fake_id}_triggers.csv")
        interval_cols = [col for col in cols if col in interval_df.columns]
        interval_df[interval_cols].to_csv(interval_csv, index=False)
        plot_zero_lag(
            interval_df,
            ranking_par,
            interval_livetime_years,
            out_dir,
            output_prefix=fake_id,
            plot_label=plot_label,
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


@action_spec(
    outputs=[],
    inputs=[
        'catalog_file',
        'progress_file',
        'job_ids_file',
        'livetime',
        'zero_lag_catalog_file',
        'zero_lag_job_ids_file',
        'fake_openbox_intervals_file',
    ],
    display_name='Background report',
    description='Generate the standard background FAR and zero-lag report',
    help=(
        "Composite action for common background-report production. It calls "
        "far_rho_plot first, then zero_lag_report with the resulting FAR data."
    ),
    composite=True,
)
def standard_background_report(
    work_dir: str,
    catalog_file: str,
    progress_file: str,
    job_ids_file: Optional[str] = None,
    livetime: Optional[float] = None,
    ranking_par: str = "rho",
    exclude_zero_lag: bool = True,
    bin_size: float = 0.1,
    output_dir: str = "public",
    include_zero_lag: bool = True,
    zero_lag_catalog_file: Optional[str] = None,
    zero_lag_job_ids_file: Optional[str] = None,
    include_fake_openbox: bool = False,
    fake_openbox_intervals_file: Optional[str] = None,
    fake_openbox_n: int = 3,
    fake_openbox_seed: int = 150914,
    public_alerts_file: Optional[str] = None,
    public_alert_time_window: float = 1.0,
    **kwargs,
) -> dict:
    """Generate standard background report artifacts."""
    far_result = far_rho_plot(
        work_dir=work_dir,
        catalog_file=catalog_file,
        progress_file=progress_file,
        job_ids_file=job_ids_file,
        livetime=livetime,
        ranking_par=ranking_par,
        exclude_zero_lag=exclude_zero_lag,
        bin_size=bin_size,
        output_dir=output_dir,
        **kwargs,
    )
    result = {"far_rho": far_result}
    if include_zero_lag:
        zero_lag_jobs = zero_lag_job_ids_file
        if zero_lag_catalog_file is None and zero_lag_jobs is None:
            zero_lag_jobs = job_ids_file
        result["zero_lag"] = zero_lag_report(
            work_dir=work_dir,
            catalog_file=zero_lag_catalog_file or catalog_file,
            progress_file=progress_file,
            job_ids_file=zero_lag_jobs,
            far_rho_data=far_result,
            ranking_par=ranking_par,
            output_dir=output_dir,
            public_alerts_file=public_alerts_file,
            public_alert_time_window=public_alert_time_window,
            **kwargs,
        )
    if include_fake_openbox:
        if fake_openbox_intervals_file is None:
            raise ValueError("include_fake_openbox=True requires fake_openbox_intervals_file")
        result["fake_openbox"] = fake_openbox_report(
            work_dir=work_dir,
            catalog_file=catalog_file,
            intervals_file=fake_openbox_intervals_file,
            far_rho_data=far_result,
            ranking_par=ranking_par,
            output_dir=output_dir,
            fake_openbox_n=fake_openbox_n,
            fake_openbox_seed=fake_openbox_seed,
            exclude_zero_lag=exclude_zero_lag,
            **kwargs,
        )
    return result


def _attach_far_and_significance(
    df: pd.DataFrame,
    far_rho_data: dict,
    ranking_par: str,
    livetime_seconds: float,
) -> pd.DataFrame:
    """Attach FAR, IFAR, p-value, and significance columns to trigger rows."""
    df = df.copy()
    bins = np.asarray(far_rho_data["bins"], dtype=float)
    far = np.asarray(far_rho_data["far"], dtype=float)
    if len(bins) == 0 or len(far) == 0:
        raise ValueError("far_rho_data must contain non-empty 'bins' and 'far' arrays")

    rho_vals = pd.to_numeric(df[ranking_par], errors="coerce").to_numpy(dtype=float)
    idx = np.searchsorted(bins, rho_vals, side="right") - 1
    idx = np.clip(idx, 0, len(far) - 1)
    attached_far = far[idx]

    df["far_attached"] = attached_far
    df["ifar_years"] = 1.0 / np.maximum(attached_far, 1e-30)

    from scipy.stats import poisson
    expected = attached_far * livetime_seconds / 86400 / 365.25
    p_values = 1.0 - poisson.cdf(0, expected)
    df["p_value"] = p_values
    df["significance"] = -np.log10(np.maximum(p_values, 1e-300))
    return df


def _read_intervals_file(path: str) -> pd.DataFrame:
    if path.endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_parquet(path)


def _interval_shift_columns(intervals: pd.DataFrame) -> list[str]:
    def suffix_index(column: str) -> int:
        return int(column[len("shift_"):])

    return sorted(
        [
            col for col in intervals.columns
            if col.startswith("shift_") and col[len("shift_"):].isdigit()
        ],
        key=suffix_index,
    )


def _shift_key_series(df: pd.DataFrame, segment_cols: list[str]) -> pd.Series:
    """Build cWB-style shift keys for one already-pruned trigger chunk."""
    if not segment_cols:
        raise KeyError("Fake openbox matching requires trigger segment_lag_* columns")

    parts = []
    for col in segment_cols:
        values = pd.to_numeric(df[col], errors="coerce")
        parts.append(values.map(_format_shift_value).astype(str))

    out = parts[0]
    for part in parts[1:]:
        out = out + "," + part
    return out


def _format_shift_value(value) -> str:
    if pd.isna(value):
        return "nan"
    return f"{float(value):.12g}"


def _shift_key(shift: tuple[float, ...]) -> str:
    return ",".join(f"{value:.12g}" for value in shift)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def plot_far_rho(data: dict, out_dir: str) -> None:
    """Plot false-alarm rate vs ranking parameter (step plot, log-y).

    Parameters
    ----------
    data : dict
        FAR data from :func:`far_rho_plot`, with keys ``bins``, ``far``,
        ``ranking_par``.
    out_dir : str
        Directory to write ``far_rho.png``.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    bins = np.array(data["bins"])
    far = np.array(data["far"])
    rp = data["ranking_par"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.step(bins, far, where="mid", color="steelblue", linewidth=1.5)
    ax.set_xlabel(rp)
    ax.set_ylabel("FAR [yr⁻¹]")
    ax.set_yscale("log")
    ax.set_title(f"False Alarm Rate vs {rp}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "far_rho.png"), dpi=120)
    plt.close(fig)


def plot_n_events(data: dict, out_dir: str) -> None:
    """Plot cumulative event count vs ranking parameter (step plot, log-y).

    Parameters
    ----------
    data : dict
        FAR data from :func:`far_rho_plot`, with keys ``bins``,
        ``cum_events``, ``ranking_par``.
    out_dir : str
        Directory to write ``far_rho_n_events.png``.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    bins = np.array(data["bins"])
    n_events = np.array(data["cum_events"])
    rp = data["ranking_par"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.step(bins, n_events, where="mid", color="crimson", linewidth=1.5)
    ax.set_xlabel(rp)
    ax.set_ylabel("Cumulative events (≥ threshold)")
    ax.set_yscale("log")
    ax.set_title(f"Cumulative Events vs {rp}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "far_rho_n_events.png"), dpi=120)
    plt.close(fig)


def load_public_alert_candidates(
    zero_lag_df: pd.DataFrame,
    public_alerts_file: str,
    gps_time_window: float,
) -> dict[int, str]:
    """Match two-column public alerts to zero-lag triggers by GPS time.

    The returned dictionary is keyed by the index of the zero-lag/IFAR array,
    so plotting can compute the IFAR value and cumulative rank directly.
    """
    if "gps_time" not in zero_lag_df.columns:
        raise KeyError("zero-lag catalog must contain gps_time to mark public alerts")
    if not os.path.exists(public_alerts_file):
        raise FileNotFoundError(f"public_alerts_file not found: {public_alerts_file}")

    trigger_gps = pd.to_numeric(zero_lag_df["gps_time"], errors="coerce").reset_index(drop=True)
    known_candidates: dict[int, str] = {}

    with open(public_alerts_file) as f:
        for line in f:
            fields = line.split()
            if len(fields) < 2 or fields[0].startswith("#"):
                continue
            try:
                alert_gps = float(fields[1])
            except ValueError:
                continue

            time_delta = (trigger_gps - alert_gps).abs()
            if not time_delta.notna().any():
                continue
            trigger_index = int(time_delta.idxmin())
            if time_delta.iloc[trigger_index] <= gps_time_window:
                candidate_id = fields[0]
                if trigger_index in known_candidates:
                    known_candidates[trigger_index] += f", {candidate_id}"
                else:
                    known_candidates[trigger_index] = candidate_id

    logger.info("Public alert candidates matched from %s: %d", public_alerts_file, len(known_candidates))
    return known_candidates


def plot_zero_lag(
    df: pd.DataFrame,
    ranking_par: str,
    livetime_years: float,
    out_dir: str,
    known_candidates: Optional[dict[int, str]] = None,
    output_prefix: str = "zero_lag",
    plot_label: str = "Zero-lag",
) -> None:
    """Plot zero-lag trigger IFAR scatter + significance histogram.

    Produces two subplots:
    1. IFAR vs *ranking_par* scatter, colour-coded by Poisson significance.
    2. Significance distribution histogram with 3σ / 5σ markers.

    Parameters
    ----------
    df : pd.DataFrame
        Zero-lag triggers with columns *ranking_par*, ``significance``,
        ``ifar_years``.
    ranking_par : str
        Column name for the ranking statistic (e.g. ``"rho"``).
    livetime_years : float
        Zero-lag live time in years (for plot title).
    out_dir : str
        Directory to write ``zero_lag_report.png``.
    known_candidates : dict, optional
        Public alert names keyed by index in the zero-lag IFAR array.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    if len(df) == 0:
        logger.warning("No %s triggers to plot", plot_label)
        return

    rho = df[ranking_par].values
    sig = df["significance"].values if "significance" in df.columns else np.zeros(len(df))
    ifar = df["ifar_years"].values if "ifar_years" in df.columns else np.zeros(len(df))

    # ── Figure 1: IFAR scatter + significance histogram ──────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.scatter(rho, ifar, c=sig, cmap="viridis", s=30, alpha=0.8, edgecolors="gray", linewidth=0.3)
    ax1.set_xlabel(ranking_par)
    ax1.set_ylabel("IFAR [yr]")
    ax1.set_yscale("log")
    ax1.set_title(f"{plot_label} triggers (livetime={livetime_years:.1f} yr)")
    ax1.grid(True, alpha=0.3)
    cbar = plt.colorbar(ax1.collections[0], ax=ax1, label="significance")

    ax2.hist(sig, bins=30, color="steelblue", edgecolor="white", alpha=0.8)
    ax2.axvline(3, color="crimson", linestyle="--", alpha=0.7, label="3σ")
    ax2.axvline(5, color="darkred", linestyle="--", alpha=0.7, label="5σ")
    ax2.set_xlabel("Significance [-log₁₀(p)]")
    ax2.set_ylabel("Count")
    ax2.set_title("Significance Distribution")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{output_prefix}_report.png"), dpi=120)
    plt.close(fig)

    # ── Figure 2: Cumulative events vs IFAR with Poisson confidence ──────
    plot_zero_lag_poisson(
        ifar,
        livetime_years,
        out_dir,
        known_candidates=known_candidates,
        output_prefix=output_prefix,
        plot_label=plot_label,
    )


def plot_zero_lag_poisson(
    ifar: np.ndarray,
    livetime_years: float,
    out_dir: str,
    known_candidates: Optional[dict[int, str]] = None,
    output_prefix: str = "zero_lag",
    plot_label: str = "Zero-lag",
) -> None:
    """Plot zero-lag cumulative events vs IFAR with Poisson confidence bands.

    Follows the approach in ``Make_PP_IFAR.C``:

    * Foreground (zero-lag) events are drawn as a red stepwise cumulative
      count sorted by descending IFAR.
    * Expected background is the diagonal
      :math:`N_{\\text{bkg}} = T / \\text{IFAR}` (a 1/x line in log-log).
    * Poisson confidence belts (1σ, 2σ, 3σ) are shaded around the expected
      background using :func:`scipy.stats.poisson.ppf`.

    Parameters
    ----------
    ifar : np.ndarray
        IFAR values in years, one per zero-lag trigger.
    livetime_years : float
        Zero-lag live time in years.
    out_dir : str
        Directory to write ``zero_lag_poisson.png``.
    known_candidates : dict, optional
        Public alert names keyed by index in the IFAR array.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    ifar = np.asarray(ifar, dtype=float)
    valid_event_indices = np.where(np.isfinite(ifar) & (ifar > 0))[0]
    if len(valid_event_indices) == 0:
        return

    # ── Sort by IFAR descending (most significant first) ─────────────────
    order = valid_event_indices[np.argsort(-ifar[valid_event_indices])]
    sorted_ifar = ifar[order]

    # Draw foreground as separate horizontal/vertical segments.  This avoids
    # dense step-path artifacts on log axes at very low IFAR.
    n_events = len(sorted_ifar)
    foreground_ifar = sorted_ifar[::-1]
    foreground_count = np.arange(n_events, 0, -1, dtype=float)

    ranks = {int(event_index): rank + 1 for rank, event_index in enumerate(order)}
    markers = []
    for event_index, candidate_id in (known_candidates or {}).items():
        if event_index in ranks and np.isfinite(ifar[event_index]) and ifar[event_index] > 0:
            markers.append((float(ifar[event_index]), float(ranks[event_index]), candidate_id))

    ifar_max = foreground_ifar.max()
    ifar_min = max(foreground_ifar.min(), 1e-10)

    # ── Log-spaced IFAR bins for background & belts ──────────────────────
    # Extend far beyond foreground to fill the entire lower-right plot area.
    # Use a fixed upper limit based on decades beyond ifar_max.
    n_bins = 800
    ifar_min_bg = max(ifar_min * 0.3, 1e-10)
    # Extend 4 decades beyond ifar_max to cover full plot
    ifar_max_bg = ifar_max * 1e4
    ifar_bg = np.logspace(np.log10(ifar_min_bg), np.log10(ifar_max_bg), n_bins)
    mu = livetime_years / ifar_bg  # expected background events

    # ── Poisson confidence intervals ─────────────────────────────────────
    # FAP values matching Make_PP_IFAR: 1σ, 2σ, 3σ
    # C++ order: {FAP2/2, FAP1/2, FAP0/2, 1-FAP0/2, 1-FAP1/2, 1-FAP2/2}
    # Returns:  [lower_3σ, lower_2σ, lower_1σ, upper_1σ, upper_2σ, upper_3σ]
    FAP0 = 1.0 - 0.682689  # 1σ
    FAP1 = 1.0 - 0.954499  # 2σ
    FAP2 = 1.0 - 0.997300  # 3σ
    sigma_labels = ["3σ", "2σ", "1σ"]
    alphas = [0.25, 0.4, 0.6]   # 3σ lightest (drawn first), 1σ darkest (drawn last)

    try:
        from scipy.stats import poisson

        # Compute Poisson percentiles directly (avoids continues_poisson CDF bug for small mu)
        percentiles = np.array([FAP2 / 2, FAP1 / 2, FAP0 / 2,
                                1 - FAP0 / 2, 1 - FAP1 / 2, 1 - FAP2 / 2])
        conf = np.array([poisson.ppf(percentiles, m) for m in mu])

        # ── Plot ─────────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 6))

        # Poisson belts: pairs are (conf[:,k], conf[:,5-k]) for k=0,1,2
        # k=0 → (lower_3σ, upper_3σ), k=1 → (lower_2σ, upper_2σ), k=2 → (lower_1σ, upper_1σ)
        for k in range(3):
            lower = conf[:, k]
            upper = conf[:, 5 - k]
            ax.fill_between(ifar_bg, lower, upper,
                            color="gray", alpha=alphas[k],
                            label=sigma_labels[k], linewidth=0)

        # Expected background line
        ax.plot(ifar_bg, mu, color="black", linewidth=0.8, linestyle="--",
                label="Expected BKG")

        # Foreground (zero-lag) step plot
        ax.step(foreground_ifar, foreground_count, where="pre", color="red", linewidth=1.5,
                label="Foreground")
        if len(markers) > 0:
            marker_ifar = [marker[0] for marker in markers]
            marker_rank = [marker[1] for marker in markers]
            ax.scatter(
                marker_ifar,
                marker_rank,
                s=34,
                color="blue",
                edgecolors="navy",
                linewidths=0.6,
                label="Known candidates",
                zorder=6,
            )
            for marker_ifar, marker_rank, candidate_id in markers:
                ax.annotate(
                    candidate_id,
                    (marker_ifar, marker_rank),
                    xytext=(6, 4),
                    textcoords="offset points",
                    fontsize=8,
                    color="black",
                    zorder=7,
                )

        # Loudest event marker
        if n_events > 0:
            loudest_ifar = foreground_ifar[-1]
            loudest_nevt = foreground_count[-1]
            ax.plot(loudest_ifar, loudest_nevt, marker="o", color="red",
                    markersize=6, markeredgecolor="darkred", markeredgewidth=1)

        ax.set_xlabel("IFAR [yr]")
        ax.set_ylabel("Cumulative Number of Events")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(ifar_min_bg, ifar_max_bg)
        y_max = max(mu[0], foreground_count[0])
        ax.set_ylim(0.1, y_max * 1.5)
        ax.set_title(f"cWB {plot_label} Events vs IFAR\nlivetime = {livetime_years:.4f} yr")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3, which="both")

        fig.tight_layout()
        poisson_path = os.path.join(out_dir, f"{output_prefix}_poisson.png")
        fig.savefig(poisson_path, dpi=150)
        plt.close(fig)
        logger.info("%s Poisson plot → %s", plot_label, poisson_path)

    except Exception as e:
        logger.warning("Poisson confidence intervals failed: %s", e)
