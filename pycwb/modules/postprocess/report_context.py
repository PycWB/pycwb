"""Shared context, formatting, and low-level helpers for the postproduction
report builder.

This module is the foundation layer for the report stack: it holds the
:class:`ReportContext` (path resolution and artifact tracking) plus the
formatting primitives, table reducers, and parquet/CSV/JSON utilities used by
the section and summary builders.  It must not import the other ``report_*``
modules to keep the dependency graph acyclic.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

SECONDS_PER_YEAR = 31557600.0
DEFAULT_MAX_POINTS = 2000
DEFAULT_MAX_BINS = 80
DEFAULT_TABLE_LIMIT = 50
JSON_PARSE_SIZE_LIMIT = 25 * 1024 * 1024
TEXT_PREVIEW_LIMIT = 20000


@dataclass
class ReportContext:
    """Shared state for path resolution and artifact tracking."""

    work_dir: str
    output_dir: str
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    missing_artifacts: list[dict[str, Any]] = field(default_factory=list)

    def resolve(self, path: Optional[str]) -> Optional[str]:
        if path is None:
            return None
        text = str(path)
        if os.path.isabs(text):
            return text
        return os.path.join(self.work_dir, text)

    def display_path(self, path: Optional[str]) -> str:
        if path is None:
            return ""
        abs_path = self.resolve(path) or ""
        try:
            return os.path.relpath(abs_path, self.output_dir).replace(os.sep, "/")
        except ValueError:
            return abs_path

    def href(self, path: Optional[str]) -> str:
        abs_path = self.resolve(path)
        if not abs_path:
            return ""
        return os.path.relpath(abs_path, self.output_dir).replace(os.sep, "/")

    def register_artifact(
        self,
        path: Optional[str],
        label: Optional[str] = None,
        kind: str = "file",
        required: bool = False,
    ) -> Optional[dict[str, Any]]:
        if not path:
            return None

        abs_path = self.resolve(path)
        assert abs_path is not None
        info: dict[str, Any] = {
            "label": label or os.path.basename(str(path)),
            "kind": kind,
            "path": self.display_path(path),
            "href": self.href(path),
            "exists": os.path.exists(abs_path),
            "required": bool(required),
            "size": "",
            "size_bytes": 0,
        }
        if info["exists"]:
            try:
                info["size_bytes"] = os.path.getsize(abs_path)
                info["size"] = _format_bytes(info["size_bytes"])
            except OSError:
                pass
            if kind == "parquet":
                info.update(_parquet_file_info(abs_path))
            elif kind == "csv":
                info["rows"] = _count_csv_rows(abs_path)
        else:
            missing = {
                "label": info["label"],
                "path": info["path"],
                "kind": kind,
                "required": bool(required),
            }
            self.missing_artifacts.append(missing)

        self.artifacts.append(info)
        return info


def _plot_card(ctx: ReportContext, entry: Any) -> dict[str, Any]:
    if isinstance(entry, dict):
        path = entry.get("path") or entry.get("file") or entry.get("href")
        label = entry.get("label") or entry.get("title")
    else:
        path = str(entry)
        label = None
    artifact = ctx.register_artifact(path, label=label, kind="image")
    return artifact or {
        "label": label or "",
        "path": "",
        "href": "",
        "exists": False,
        "size": "",
    }


def _dataframe_table(
    df: pd.DataFrame,
    preferred_columns: Optional[list[str]] = None,
    limit: int = DEFAULT_TABLE_LIMIT,
) -> dict[str, Any]:
    if df is None or df.empty:
        return _empty_table()
    if preferred_columns:
        columns = [col for col in preferred_columns if col in df.columns]
        if not columns:
            columns = list(df.columns)
    else:
        columns = list(df.columns)
    capped = df.loc[:, columns].head(limit).copy()
    rows = []
    for _, row in capped.iterrows():
        rows.append({col: _format_cell(row[col]) for col in columns})
    return {
        "columns": columns,
        "rows": rows,
        "n_rows": int(len(df)),
        "displayed_rows": int(len(capped)),
    }


def _empty_table() -> dict[str, Any]:
    return {"columns": [], "rows": [], "n_rows": 0, "displayed_rows": 0}


def _format_cell(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        val = float(value)
        if not math.isfinite(val):
            return ""
        if abs(val) >= 1e5 or (0 < abs(val) < 1e-3):
            return f"{val:.4e}"
        return f"{val:.6g}"
    return str(value)


def _format_bytes(size: Any) -> str:
    try:
        value = float(size)
    except (TypeError, ValueError):
        return ""
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return ""


def _format_int(value: Any) -> str:
    try:
        if value is None:
            return "0"
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "0"


def _format_years(value: Any) -> str:
    val = _to_float(value)
    if val is None:
        return "unknown"
    if abs(val) >= 1:
        return f"{val:.3g} yr"
    return f"{val:.3e} yr"


def _livetime_dict(seconds: Optional[float]) -> dict[str, Any]:
    if seconds is None:
        return {
            "seconds": None,
            "days": None,
            "years": None,
            "label": "unknown",
            "seconds_label": "unknown",
            "days_label": "unknown",
            "years_label": "unknown",
            "compact_label": "unknown",
            "bkg_compact_label": "unknown",
            "zero_lag_compact_label": "unknown",
        }
    seconds = float(seconds)
    days = seconds / 86400.0
    years = seconds / SECONDS_PER_YEAR
    seconds_label = f"{seconds:.0f} s"
    days_label = f"{days:.3f} d"
    years_label = f"{years:.3f} yr"
    return {
        "seconds": seconds,
        "days": days,
        "years": years,
        "label": years_label,
        "seconds_label": seconds_label,
        "days_label": days_label,
        "years_label": years_label,
        "compact_label": f"{years_label} / {days_label} / {seconds_label}",
        "bkg_compact_label": f"{years_label} / {seconds_label}",
        "zero_lag_compact_label": f"{years_label} / {days_label} / {seconds_label}",
    }


def _config_summary(config: dict[str, Any]) -> list[dict[str, str]]:
    keys = [
        "ifo", "nIFO", "cfg_search", "search", "fLow", "fHigh", "inRate",
        "fResample", "rateANA", "levelR", "l_low", "l_high", "nRES",
        "lagStep", "segEdge", "segMLS", "xgb_rho_mode",
    ]
    summary = []
    for key in keys:
        if key in config:
            value = config[key]
            if isinstance(value, list):
                value = ", ".join(str(item) for item in value)
            summary.append({"key": key, "value": _format_cell(value)})
    if "DQF" in config and isinstance(config["DQF"], list):
        summary.append({"key": "DQF entries", "value": str(len(config["DQF"]))})
    if "dq_files" in config and isinstance(config["dq_files"], list):
        summary.append({"key": "DQ files", "value": str(len(config["dq_files"]))})
    return summary


def _compact_mapping(value: Any, limit: int = 80) -> Any:
    if isinstance(value, dict):
        result = {}
        for idx, (key, item) in enumerate(value.items()):
            if idx >= limit:
                result["..."] = f"{len(value) - limit} more"
                break
            result[key] = _compact_mapping(item, limit=limit)
        return result
    if isinstance(value, list):
        capped = [_compact_mapping(item, limit=limit) for item in value[:limit]]
        if len(value) > limit:
            capped.append(f"... {len(value) - limit} more")
        return capped
    return value


# ---------------------------------------------------------------------------
# Low-level utilities
# ---------------------------------------------------------------------------


def _resolve_path(work_dir: str, path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(work_dir, path)


def _catalog_progress_path(catalog_path: Optional[str]) -> Optional[str]:
    if not catalog_path:
        return None
    dirname = os.path.dirname(catalog_path)
    basename = os.path.basename(catalog_path).replace("catalog", "progress", 1)
    return os.path.join(dirname, basename)


def _parquet_file_info(path: Optional[str]) -> dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    try:
        parquet = pq.ParquetFile(path)
        rows = int(parquet.metadata.num_rows)
        return {
            "rows": rows,
            "rows_label": _format_int(rows),
            "columns": int(len(parquet.schema_arrow.names)),
            "column_names": list(parquet.schema_arrow.names),
        }
    except Exception as exc:
        logger.warning("Could not inspect parquet %s: %s", path, exc)
        return {}


def _parquet_columns(path: str) -> list[str]:
    try:
        return list(pq.ParquetFile(path).schema_arrow.names)
    except Exception:
        return []


def _count_csv_rows(path: str) -> int:
    try:
        with open(path) as f:
            n_lines = sum(1 for _ in f)
        return max(0, n_lines - 1)
    except OSError:
        return 0


def _read_text_preview(path: Optional[str], limit: Optional[int] = TEXT_PREVIEW_LIMIT) -> str:
    if not path or not os.path.exists(path):
        return ""
    try:
        with open(path) as f:
            text = f.read() if limit is None else f.read(limit + 1)
        if limit is not None and len(text) > limit:
            return text[:limit] + "\n..."
        return text
    except OSError as exc:
        logger.warning("Could not read text file %s: %s", path, exc)
        return ""


def _load_json_if_reasonable(path: str, size_limit: int = JSON_PARSE_SIZE_LIMIT) -> Any:
    try:
        if os.path.getsize(path) > size_limit:
            logger.info("Skipping large JSON artifact for report data: %s", path)
            return None
        with open(path) as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Could not read JSON %s: %s", path, exc)
        return None


def _frequency_series(df: pd.DataFrame) -> Optional[pd.Series]:
    if "central_freq" in df.columns:
        return pd.to_numeric(df["central_freq"], errors="coerce")
    if "frequency" in df.columns:
        return pd.to_numeric(df["frequency"], errors="coerce")
    freq_cols = [col for col in df.columns if col.startswith("central_freq_")]
    if not freq_cols:
        return None
    freq_df = df[freq_cols].apply(pd.to_numeric, errors="coerce")
    return freq_df.mean(axis=1)


def _aligned_finite_arrays(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    if not arrays:
        return tuple()
    min_len = min(len(arr) for arr in arrays)
    trimmed = [np.asarray(arr[:min_len], dtype=float) for arr in arrays]
    mask = np.ones(min_len, dtype=bool)
    for arr in trimmed:
        mask &= np.isfinite(arr)
    return tuple(arr[mask] for arr in trimmed)


def _downsample_arrays(max_points: int, *arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    if not arrays:
        return tuple()
    n = len(arrays[0])
    if n <= max_points:
        return arrays
    idx = np.linspace(0, n - 1, max_points).round().astype(int)
    return tuple(np.asarray(arr)[idx] for arr in arrays)


def _array_to_list(values: Any) -> list[Any]:
    arr = np.asarray(values)
    return [_jsonable(v) for v in arr.tolist()]


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    return val if math.isfinite(val) else None


def _nested_get(data: Any, path: list[str]) -> Any:
    cur = data
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _safe_id(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value)).strip("_") or "section"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        val = float(value)
        return val if math.isfinite(val) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (np.ndarray,)):
        return _jsonable(value.tolist())
    return value
