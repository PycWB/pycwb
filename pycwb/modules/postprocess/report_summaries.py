"""Data readers, reducers, and figure builders for the postproduction report.

These helpers turn parquet/CSV/JSON artifacts into the small summaries and
figure dictionaries embedded in the report.  They depend only on
:mod:`pycwb.modules.postprocess.report_context` for shared primitives.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

from pycwb.modules.catalog.catalog import Catalog
from pycwb.modules.postprocess.lag_filters import (
    try_unshifted_job_ids_from_catalog,
    zero_lag_mask,
)
from pycwb.modules.postprocess.report_context import (
    DEFAULT_MAX_BINS,
    SECONDS_PER_YEAR,
    ReportContext,
    _aligned_finite_arrays,
    _array_to_list,
    _compact_mapping,
    _dataframe_table,
    _downsample_arrays,
    _empty_table,
    _format_cell,
    _format_int,
    _frequency_series,
    _livetime_dict,
    _load_json_if_reasonable,
    _nested_get,
    _parquet_columns,
    _parquet_file_info,
    _read_text_preview,
    _safe_id,
    _to_float,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data readers and reducers
# ---------------------------------------------------------------------------


def _select_bkg_livetime(
    explicit_livetime: Optional[float],
    progress_summary: dict[str, Any],
    interval_summary: dict[str, Any],
    far_data: dict[str, Any],
) -> dict[str, Any]:
    """Choose the BKG live time used by the report.

    Workflow context should normally pass the split live time, but reports are
    scientific review artifacts, so a badly scaled manual value should not
    silently dominate when progress/interval/FAR metadata agree otherwise.
    """
    candidates: list[tuple[str, Optional[float]]] = [
        ("progress_file", _nested_get(progress_summary, ["total_livetime", "seconds"])),
        ("intervals_file", _nested_get(interval_summary, ["total_livetime", "seconds"])),
        ("far_json", _nested_get(far_data, ["livetime", "seconds"])),
    ]
    candidates = [(name, _to_float(value)) for name, value in candidates]
    candidates = [(name, value) for name, value in candidates if value is not None and value > 0]

    explicit = _to_float(explicit_livetime)
    if explicit is None or explicit <= 0:
        if candidates:
            name, value = candidates[0]
            return {"seconds": value, "source": name}
        return {"seconds": None, "source": "unavailable"}

    if not candidates:
        return {"seconds": explicit, "source": "explicit"}

    name, measured = candidates[0]
    rel_diff = abs(explicit - measured) / max(abs(measured), 1.0)
    if rel_diff > 0.05:
        return {
            "seconds": measured,
            "source": name,
            "warning": (
                "Explicit live time differs from measured live time by "
                f"{rel_diff:.1%}; using {name}."
            ),
        }
    return {"seconds": explicit, "source": "explicit"}


def _read_catalog_metadata(path: Optional[str]) -> dict[str, Any]:
    if not path or not os.path.exists(path):
        return {"version": "", "config": {}, "jobs": []}
    try:
        catalog = Catalog.open(path)
        return {
            "version": catalog.version,
            "config": catalog.config,
            "jobs": catalog.jobs,
        }
    except Exception as exc:
        logger.warning("Catalog metadata read via Catalog.open failed for %s: %s", path, exc)

    try:
        metadata = pq.read_schema(path).metadata or {}
        return {
            "version": metadata.get(b"pycwb_version", b"").decode(),
            "config": json.loads(metadata[b"config"].decode()) if b"config" in metadata else {},
            "jobs": json.loads(metadata[b"jobs"].decode()) if b"jobs" in metadata else [],
        }
    except Exception as exc:
        logger.warning("Parquet metadata read failed for %s: %s", path, exc)
        return {"version": "", "config": {}, "jobs": []}


def _build_scored_bkg_summary(
    ctx: ReportContext,
    catalog_file: Optional[str],
    ranking_par: str,
    livetime: Optional[float],
    max_bins: int,
    table_limit: int,
) -> dict[str, Any]:
    path = ctx.resolve(catalog_file)
    info = _parquet_file_info(path) if path and os.path.exists(path) else {}
    if not path or not os.path.exists(path):
        return {"info": info, "loudest_events": _empty_table(), "figures": []}

    existing = _parquet_columns(path)
    freq_cols = [col for col in existing if col.startswith("central_freq_")]
    needed = [
        "id", "job_id", "lag_idx", "trial_idx", ranking_par, "xgb_prob",
        "ifar", "gps_time", "net_cc", "likelihood", "coherent_energy",
        "central_freq", "frequency",
    ] + freq_cols
    cols = [col for col in needed if col in existing]
    if ranking_par not in cols:
        return {
            "info": info,
            "loudest_events": _empty_table(),
            "figures": [],
            "warning": f"Ranking parameter '{ranking_par}' not found in scored catalog.",
        }

    df = pd.read_parquet(path, columns=cols)
    df[ranking_par] = pd.to_numeric(df[ranking_par], errors="coerce")
    if "gps_time" in df.columns:
        df["gps_time"] = pd.to_numeric(df["gps_time"], errors="coerce")
    for col in ["xgb_prob", "ifar", "net_cc", "likelihood", "coherent_energy"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    frequency = _frequency_series(df)
    if frequency is not None:
        df["_frequency"] = frequency

    if livetime and livetime > 0:
        valid_ranking = df[ranking_par].fillna(-np.inf).to_numpy()
        order = np.argsort(-valid_ranking)
        ranks = np.empty(len(df), dtype=float)
        ranks[order] = np.arange(1, len(df) + 1, dtype=float)
        livetime_years = livetime / SECONDS_PER_YEAR
        if livetime_years > 0:
            df["_far_per_year"] = ranks / livetime_years
            df["_ifar_years"] = 1.0 / np.maximum(df["_far_per_year"], 1e-300)

    loudest_cols = [
        "id", "job_id", "lag_idx", "trial_idx", ranking_par, "xgb_prob",
        "_ifar_years", "_far_per_year", "gps_time", "_frequency",
        "net_cc", "likelihood", "coherent_energy",
    ]
    loudest = _dataframe_table(
        df.sort_values(ranking_par, ascending=False),
        preferred_columns=loudest_cols,
        limit=table_limit,
    )

    figures = []
    if "gps_time" in df.columns:
        figures.append(_binned_figure(
            df, "gps_time", ranking_par, max_bins,
            figure_id="bkg_rank_time",
            title=f"{ranking_par} vs GPS time",
            x_label="GPS time",
            y_label=ranking_par,
            agg="max",
        ))
        if "_far_per_year" in df.columns:
            figures.append(_binned_figure(
                df, "gps_time", "_far_per_year", max_bins,
                figure_id="bkg_far_time",
                title="Loudest-event FAR vs GPS time",
                x_label="GPS time",
                y_label="FAR [yr^-1]",
                agg="min",
                log_y=True,
            ))
    if "_frequency" in df.columns:
        figures.append(_binned_figure(
            df, "_frequency", ranking_par, max_bins,
            figure_id="bkg_rank_frequency",
            title=f"{ranking_par} vs frequency",
            x_label="Frequency [Hz]",
            y_label=ranking_par,
            agg="max",
        ))
        if "_far_per_year" in df.columns:
            figures.append(_binned_figure(
                df, "_frequency", "_far_per_year", max_bins,
                figure_id="bkg_far_frequency",
                title="Loudest-event FAR vs frequency",
                x_label="Frequency [Hz]",
                y_label="FAR [yr^-1]",
                agg="min",
                log_y=True,
            ))

    figures = [fig for fig in figures if fig]
    return {
        "info": info,
        "loudest_events": loudest,
        "figures": figures,
    }


def _load_far_curve_data(
    ctx: ReportContext,
    binned_far_json: Optional[str],
    far_json: Optional[str],
    max_plot_points: int,
) -> dict[str, Any]:
    source = None
    data = None
    far_units = "yr^-1"

    binned_path = ctx.resolve(binned_far_json)
    if binned_path and os.path.exists(binned_path):
        data = _load_json_if_reasonable(binned_path)
        source = ctx.display_path(binned_far_json)

    if data is None:
        far_path = ctx.resolve(far_json)
        if far_path and os.path.exists(far_path):
            data = _load_json_if_reasonable(far_path)
            source = ctx.display_path(far_json)
            far_units = "s^-1"

    if data is None:
        return {"source": source, "figures": [], "warning": "FAR data unavailable or too large to parse."}

    try:
        livetime_seconds = None
        if isinstance(data, dict):
            x = np.asarray(data.get("bins") or data.get("rho") or [], dtype=float)
            far = np.asarray(data.get("far") or [], dtype=float)
            cumulative = np.asarray(data.get("cum_events") or data.get("n_events") or [], dtype=float)
            ranking_par = str(data.get("ranking_par") or "rho")
            far_per_year = far
            livetime_seconds = _to_float(data.get("livetime"))
        elif isinstance(data, list):
            capped = data
            if len(capped) > max_plot_points:
                idx = np.linspace(0, len(capped) - 1, max_plot_points).round().astype(int)
                capped = [capped[int(i)] for i in idx]
            x = np.asarray([item.get("rho", item.get("ranking", np.nan)) for item in capped], dtype=float)
            far = np.asarray([item.get("far", np.nan) for item in capped], dtype=float)
            cumulative = np.asarray([item.get("n_events", idx + 1) for idx, item in enumerate(capped)], dtype=float)
            ranking_par = "rho"
            far_per_year = far * SECONDS_PER_YEAR if far_units == "s^-1" else far
            if data and _to_float(data[0].get("far")):
                first_far = _to_float(data[0].get("far"))
                first_n = _to_float(data[0].get("n_events"))
                if first_far and first_far > 0:
                    livetime_seconds = (first_n or 1.0) / first_far
        else:
            return {"source": source, "figures": [], "warning": "FAR data format not recognized."}
    except Exception as exc:
        return {"source": source, "figures": [], "warning": f"Could not parse FAR data: {exc}"}

    x, far_per_year, cumulative = _aligned_finite_arrays(x, far_per_year, cumulative)
    x, far_per_year, cumulative = _downsample_arrays(max_plot_points, x, far_per_year, cumulative)
    if len(x) == 0:
        return {"source": source, "figures": [], "warning": "FAR data contains no finite points."}

    ifar_years = 1.0 / np.maximum(far_per_year, 1e-300)
    figures = [
        _xy_figure(
            "far_rho_curve",
            f"FAR vs {ranking_par}",
            x,
            far_per_year,
            ranking_par,
            "FAR [yr^-1]",
            log_y=True,
        ),
        _xy_figure(
            "cumulative_ifar_curve",
            "Cumulative BKG count vs IFAR",
            ifar_years,
            cumulative,
            "IFAR [yr]",
            "Cumulative count",
            log_x=True,
            log_y=True,
        ),
    ]
    return {
        "source": source,
        "n_points": int(len(x)),
        "livetime": _livetime_dict(livetime_seconds),
        "figures": figures,
    }


def _progress_summary(path: Optional[str]) -> dict[str, Any]:
    if not path or not os.path.exists(path):
        return {"info": {}, "status_counts": [], "by_lag": _empty_table()}
    info = _parquet_file_info(path)
    existing = _parquet_columns(path)
    cols = [col for col in ["job_id", "lag_idx", "livetime", "status", "n_triggers"] if col in existing]
    if not cols:
        return {"info": info, "status_counts": [], "by_lag": _empty_table()}

    df = pd.read_parquet(path, columns=cols)
    if "livetime" in df.columns:
        df["livetime"] = pd.to_numeric(df["livetime"], errors="coerce").fillna(0.0)
    if "status" in df.columns:
        status_counts = [
            {"status": str(status), "count": int(count)}
            for status, count in df["status"].value_counts(dropna=False).items()
        ]
        completed = df[df["status"] == "completed"].copy()
    else:
        status_counts = []
        completed = df.copy()

    total_livetime = float(completed["livetime"].sum()) if "livetime" in completed.columns else None
    n_jobs = int(completed["job_id"].nunique()) if "job_id" in completed.columns else None
    by_lag = _empty_table()
    if {"lag_idx", "livetime"}.issubset(completed.columns):
        grouped = completed.groupby("lag_idx", dropna=False).agg(
            livetime=("livetime", "sum"),
            n_rows=("livetime", "size"),
        ).reset_index()
        if "n_triggers" in completed.columns:
            triggers = completed.groupby("lag_idx", dropna=False)["n_triggers"].sum().reset_index(name="n_triggers")
            grouped = grouped.merge(triggers, on="lag_idx", how="left")
        by_lag = _dataframe_table(grouped.sort_values("lag_idx"), limit=100)

    return {
        "info": info,
        "total_livetime": _livetime_dict(total_livetime),
        "n_jobs": n_jobs,
        "status_counts": status_counts,
        "by_lag": by_lag,
    }


def _zero_lag_livetime_summary(
    progress_path: Optional[str],
    catalog_path: Optional[str],
) -> dict[str, Any]:
    if not progress_path or not os.path.exists(progress_path):
        return {
            "source": "",
            "livetime": _livetime_dict(None),
            "n_jobs": None,
            "n_rows": 0,
            "warning": "Zero-lag progress file is unavailable.",
        }

    existing = _parquet_columns(progress_path)
    cols = [
        col for col in [
            "job_id", "lag_idx", "lag", "livetime", "status",
            "time_lag", "segment_lag", "shift",
        ] if col in existing
    ]
    cols.extend([
        col for col in existing
        if col.startswith(("time_lag_", "segment_lag_", "segment_shift_", "shift_"))
        and col not in cols
    ])
    if "livetime" not in cols:
        return {
            "source": progress_path,
            "livetime": _livetime_dict(None),
            "n_jobs": None,
            "n_rows": 0,
            "warning": "Progress file does not contain a livetime column.",
        }

    try:
        progress = pd.read_parquet(progress_path, columns=cols)
    except Exception as exc:
        logger.warning("Could not read zero-lag progress %s: %s", progress_path, exc)
        return {
            "source": progress_path,
            "livetime": _livetime_dict(None),
            "n_jobs": None,
            "n_rows": 0,
            "warning": f"Could not read zero-lag progress: {exc}",
        }

    unshifted_job_ids = None
    if catalog_path and os.path.exists(catalog_path):
        try:
            unshifted_job_ids = try_unshifted_job_ids_from_catalog(catalog_path)
        except Exception as exc:
            logger.warning("Could not read unshifted jobs for zero-lag live time: %s", exc)

    zero = progress[zero_lag_mask(progress, unshifted_job_ids=unshifted_job_ids)].copy()
    if "status" in zero.columns:
        zero = zero[zero["status"] == "completed"]
    zero["livetime"] = pd.to_numeric(zero["livetime"], errors="coerce").fillna(0.0)
    seconds = float(zero["livetime"].sum())
    has_shift_columns = any(
        col in progress.columns
        or any(existing_col.startswith(f"{col}_") for existing_col in progress.columns)
        for col in ("segment_lag", "segment_shift", "shift")
    )
    warning = ""
    if unshifted_job_ids is None and not has_shift_columns:
        warning = (
            "Zero-lag live time was selected without catalog job metadata or "
            "segment-shift columns, so shifted jobs may be included."
        )
    return {
        "source": os.path.basename(progress_path),
        "livetime": _livetime_dict(seconds),
        "n_jobs": int(zero["job_id"].nunique()) if "job_id" in zero.columns else None,
        "n_rows": int(len(zero)),
        "warning": warning,
    }


def _interval_summary(path: Optional[str]) -> dict[str, Any]:
    if not path or not os.path.exists(path):
        return {"info": {}, "table": _empty_table()}
    info = _parquet_file_info(path)
    existing = _parquet_columns(path)
    cols = [col for col in ["shift_key", "lag_idx", "livetime", "n_rows", "n_jobs", "shift_0", "shift_1"] if col in existing]
    if not cols:
        return {"info": info, "table": _empty_table()}
    df = pd.read_parquet(path, columns=cols)
    table = _dataframe_table(df.sort_values(cols[:1]), limit=100)
    livetime = pd.to_numeric(df["livetime"], errors="coerce").sum() if "livetime" in df.columns else None
    return {
        "info": info,
        "total_livetime": _livetime_dict(float(livetime) if livetime is not None else None),
        "table": table,
    }


def _progress_figures(progress_summary: dict[str, Any]) -> list[dict[str, Any]]:
    table = progress_summary.get("by_lag") or {}
    rows = table.get("rows") or []
    if not rows:
        return []
    try:
        x = np.asarray([float(row["lag_idx"]) for row in rows], dtype=float)
        y = np.asarray([float(row["livetime"]) for row in rows], dtype=float)
    except Exception:
        return []
    return [
        _xy_figure(
            "livetime_by_lag",
            "Live time by lag",
            x,
            y,
            "Lag index",
            "Live time [s]",
            mode="markers",
            chart_type="bar",
        )
    ]


def _catalog_numeric_summary(
    path: Optional[str],
    numeric_columns: list[str],
    max_bins: int = DEFAULT_MAX_BINS,
) -> dict[str, Any]:
    if not path or not os.path.exists(path):
        return {"info": {}, "numeric": [], "histograms": []}
    info = _parquet_file_info(path)
    existing = _parquet_columns(path)
    cols = [col for col in numeric_columns if col in existing]
    if not cols:
        return {"info": info, "numeric": [], "histograms": []}
    df = pd.read_parquet(path, columns=cols)
    numeric = []
    histograms = []
    for col in cols:
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        if values.empty:
            continue
        numeric.append({
            "column": col,
            "count": int(values.size),
            "min": _format_cell(values.min()),
            "median": _format_cell(values.median()),
            "max": _format_cell(values.max()),
        })
        if col in {"rho", "xgb_prob"}:
            histograms.append(_histogram_figure(values.to_numpy(), col, max_bins))
    return {
        "info": info,
        "numeric": numeric,
        "histograms": [fig for fig in histograms if fig],
    }


def _matched_sim_summary(path: Optional[str]) -> dict[str, Any]:
    if not path or not os.path.exists(path):
        return {"info": {}, "table": _empty_table(), "metrics": []}
    info = _parquet_file_info(path)
    existing = _parquet_columns(path)
    cols = [
        col for col in [
            "sim_sim_idx", "sim_name", "sim_hrss", "id", "rho", "xgb_prob",
            "sim_vetoed_cat0", "sim_vetoed_cat1", "sim_vetoed_cat2",
            "sim_across_segments",
        ] if col in existing
    ]
    if not cols:
        return {"info": info, "table": _empty_table(), "metrics": []}
    df = pd.read_parquet(path, columns=cols)
    metrics = []
    if "sim_sim_idx" in df.columns:
        metrics.append({"label": "Unique simulations", "value": _format_int(df["sim_sim_idx"].nunique())})
    if "id" in df.columns:
        metrics.append({"label": "Recovered rows", "value": _format_int(df["id"].notna().sum())})
    if "sim_name" in df.columns:
        metrics.append({"label": "Waveforms", "value": _format_int(df["sim_name"].nunique())})
    if "sim_hrss" in df.columns:
        hrss = pd.to_numeric(df["sim_hrss"], errors="coerce").dropna()
        if not hrss.empty:
            metrics.append({"label": "hrss range", "value": f"{_format_cell(hrss.min())} to {_format_cell(hrss.max())}"})
    table = _dataframe_table(df.head(100), limit=100)
    return {"info": info, "metrics": metrics, "table": table}


def _read_fit_parameter_table(path: Optional[str], table_limit: int) -> dict[str, Any]:
    preferred = [
        "ifar", "waveform", "status", "fit_status", "hrss10", "hrss50",
        "hrss90", "chi2", "hrssEr", "sigma", "betam", "betap", "flag",
    ]
    return _read_csv_table(
        path,
        table_limit=table_limit,
        preferred_columns=preferred,
        sort_columns=["ifar", "waveform"],
        ascending=True,
    )


def _read_csv_table(
    path: Optional[str],
    table_limit: int,
    preferred_columns: Optional[list[str]] = None,
    sort_columns: Optional[list[str]] = None,
    ascending: bool = False,
) -> dict[str, Any]:
    if not path or not os.path.exists(path):
        return _empty_table()
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        logger.warning("Could not read CSV table %s: %s", path, exc)
        return _empty_table()
    if sort_columns:
        for col in sort_columns:
            if col in df.columns:
                df = df.sort_values(col, ascending=ascending)
                break
    return _dataframe_table(df, preferred_columns=preferred_columns, limit=table_limit)


def _load_workflow_yaml(path: Optional[str]) -> tuple[str, dict[str, Any]]:
    if not path or not os.path.exists(path):
        return "", {}
    text = _read_text_preview(path, limit=None)
    try:
        return text, yaml.safe_load(text) or {}
    except Exception as exc:
        logger.warning("Could not parse workflow YAML %s: %s", path, exc)
        return text, {}


def _workflow_steps_by_action(workflow_data: dict[str, Any], action_fragments: list[str]) -> list[dict[str, Any]]:
    steps = []
    for idx, step in enumerate(workflow_data.get("steps", []) or []):
        action = str(step.get("action") or "")
        if any(fragment in action for fragment in action_fragments):
            steps.append({
                "index": idx + 1,
                "id": step.get("id") or "",
                "name": step.get("name") or step.get("id") or action,
                "action": action,
                "inputs": _compact_mapping(step.get("inputs") or {}, limit=40),
                "args": _compact_mapping(step.get("args") or {}, limit=40),
                "outputs": _compact_mapping(step.get("outputs") or {}, limit=40),
            })
    return steps


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------


def _xy_figure(
    figure_id: str,
    title: str,
    x: Any,
    y: Any,
    x_label: str,
    y_label: str,
    log_x: bool = False,
    log_y: bool = False,
    mode: str = "lines+markers",
    chart_type: str = "scatter",
) -> dict[str, Any]:
    return {
        "id": _safe_id(figure_id),
        "title": title,
        "traces": [{
            "x": _array_to_list(x),
            "y": _array_to_list(y),
            "type": chart_type,
            "mode": mode,
            "marker": {"size": 6, "color": "#2563eb"},
            "line": {"color": "#2563eb", "width": 1.5},
        }],
        "layout": {
            "margin": {"l": 60, "r": 20, "t": 34, "b": 52},
            "xaxis": {"title": x_label, "type": "log" if log_x else "linear"},
            "yaxis": {"title": y_label, "type": "log" if log_y else "linear"},
            "showlegend": False,
        },
    }


def _binned_figure(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    bins: int,
    figure_id: str,
    title: str,
    x_label: str,
    y_label: str,
    agg: str = "max",
    log_y: bool = False,
) -> Optional[dict[str, Any]]:
    if x_col not in df.columns or y_col not in df.columns:
        return None
    values = df[[x_col, y_col]].copy()
    values[x_col] = pd.to_numeric(values[x_col], errors="coerce")
    values[y_col] = pd.to_numeric(values[y_col], errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return None
    xmin = float(values[x_col].min())
    xmax = float(values[x_col].max())
    if not np.isfinite(xmin) or not np.isfinite(xmax):
        return None
    if xmin == xmax:
        grouped = pd.DataFrame({"x": [xmin], "y": [float(values[y_col].iloc[0])], "count": [len(values)]})
    else:
        nbins = max(1, min(int(bins), len(values)))
        edges = np.linspace(xmin, xmax, nbins + 1)
        values["_bin"] = pd.cut(values[x_col], edges, labels=False, include_lowest=True)
        grouped_raw = values.dropna(subset=["_bin"]).groupby("_bin")
        if agg == "min":
            y = grouped_raw[y_col].min()
        else:
            y = grouped_raw[y_col].max()
        count = grouped_raw[y_col].size()
        centers = (edges[:-1] + edges[1:]) / 2.0
        grouped = pd.DataFrame({
            "x": [float(centers[int(i)]) for i in y.index],
            "y": [float(v) for v in y.values],
            "count": [int(count.loc[i]) for i in y.index],
        })
    return _xy_figure(
        figure_id,
        title,
        grouped["x"].to_numpy(),
        grouped["y"].to_numpy(),
        x_label,
        y_label,
        log_y=log_y,
        mode="lines+markers",
    )


def _histogram_figure(values: np.ndarray, column: str, max_bins: int) -> Optional[dict[str, Any]]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    hist, edges = np.histogram(values, bins=min(max_bins, max(1, values.size)))
    centers = (edges[:-1] + edges[1:]) / 2.0
    return _xy_figure(
        f"hist_{column}",
        f"{column} distribution",
        centers,
        hist,
        column,
        "Count",
        chart_type="bar",
        mode="markers",
    )
