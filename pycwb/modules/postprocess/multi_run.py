"""Generic readers for postproduction studies spanning multiple runs."""

from __future__ import annotations

import json
import os
from typing import Any, Optional

import pandas as pd

from pycwb.modules.catalog.catalog import Catalog
from pycwb.post_production.action_spec import action_spec


def _resolve(work_dir: str, path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(work_dir, path)


def _output_path(work_dir: str, path: str) -> str:
    resolved = _resolve(work_dir, path)
    os.makedirs(os.path.dirname(resolved) or ".", exist_ok=True)
    return resolved


def _normalise_run(entry: Any, index: int) -> dict[str, Any]:
    if isinstance(entry, str):
        return {
            "run_index": index,
            "name": f"run {index}",
            "label": f"run {index}",
            "catalog_file": entry,
            "metadata": {},
        }
    if not isinstance(entry, dict):
        raise TypeError("Each run must be a catalog path or a mapping")
    catalog_file = (
        entry.get("catalog_file") or entry.get("catalog") or entry.get("path")
    )
    if not catalog_file:
        raise ValueError(f"Run {index} does not define catalog_file")
    name = str(entry.get("name") or entry.get("label") or f"run {index}")
    metadata = {
        key: value
        for key, value in entry.items()
        if key not in {"catalog_file", "catalog", "path", "label", "name"}
    }
    return {
        "run_index": index,
        "name": name,
        # Compatibility alias for manifests and consumers created before
        # ``name`` became the canonical generic identifier.
        "label": name,
        "catalog_file": str(catalog_file),
        "metadata": metadata,
    }


def _attach_run_columns(
    frame: pd.DataFrame,
    run: dict[str, Any],
    *,
    row_name: str,
) -> pd.DataFrame:
    source_index = frame.index.to_numpy(copy=True)
    result = frame.reset_index(drop=True)
    if "source_row_index" in result.columns:
        result = result.rename(
            columns={"source_row_index": "upstream_source_row_index"}
        )
    result.insert(0, "source_row_index", source_index)
    result.insert(0, row_name, range(len(result)))
    result.insert(0, "run_label", run["name"])
    result.insert(0, "run_name", run["name"])
    result.insert(0, "run_index", int(run["run_index"]))
    result["run_catalog_file"] = run["catalog_file"]
    result["run_metadata"] = json.dumps(run["metadata"], sort_keys=True)
    return result


def _catalog_injections(catalog: Catalog) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for job in catalog.jobs:
        job_id = job.get("index") if isinstance(job, dict) else None
        injections = job.get("injections", []) if isinstance(job, dict) else []
        for injection in injections or []:
            row = dict(injection)
            row.setdefault("job_id", job_id)
            rows.append(row)
    return pd.DataFrame(rows)


_EMPTY_INJECTION_COLUMNS = [
    "ra",
    "dec",
    "gps_time",
    "parameters",
    "job_id",
]


@action_spec(
    outputs=["output_file", "injections_output_file", "manifest_file"],
    inputs=["runs"],
    display_name="Read catalog runs",
    description="Read, combine, and reindex an array of pycWB catalogs",
    help=(
        "Each run may provide a name, catalog_file, and arbitrary metadata. "
        "The action writes combined trigger and scheduled-injection parquet "
        "tables with run_index/run_name columns plus a JSON manifest."
    ),
)
def read_catalog_runs(
    work_dir: str,
    runs: list[Any],
    output_file: str = "tmp/postprod/catalog_runs.parquet",
    injections_output_file: Optional[str] = None,
    manifest_file: Optional[str] = None,
    columns: Optional[list[str]] = None,
    require_injections: bool = False,
    **kwargs,
) -> dict[str, Any]:
    """Combine multiple catalogs while retaining their run identity.

    The output trigger table has a fresh global RangeIndex and explicit
    ``run_index``, ``run_name``, ``run_row_index`` and ``source_row_index``
    columns. ``run_label`` is retained as a compatibility alias for
    ``run_name``. Scheduled injections are read from each catalog's job
    metadata and written to a second table using the same run identifiers.
    This preserves injections that were not recovered and therefore have no
    trigger row.
    """
    if not runs:
        raise ValueError("runs must contain at least one catalog")

    work_dir = os.path.abspath(str(work_dir))
    normalised = [_normalise_run(entry, index) for index, entry in enumerate(runs)]
    trigger_frames: list[pd.DataFrame] = []
    injection_frames: list[pd.DataFrame] = []
    manifest_runs: list[dict[str, Any]] = []

    for run in normalised:
        catalog_path = _resolve(work_dir, run["catalog_file"])
        if not os.path.exists(catalog_path):
            raise FileNotFoundError(f"Catalog not found: {catalog_path}")
        catalog = Catalog.open(catalog_path)
        frame = pd.read_parquet(catalog_path, columns=columns)
        trigger_frames.append(
            _attach_run_columns(frame, run, row_name="run_row_index")
        )

        injections = _catalog_injections(catalog)
        if not injections.empty:
            injection_frames.append(
                _attach_run_columns(injections, run, row_name="run_injection_index")
            )
        elif require_injections:
            raise ValueError(
                f"Catalog {run['catalog_file']} has no scheduled injections in job metadata"
            )

        manifest_runs.append(
            {
                **run,
                "catalog_file": os.path.abspath(catalog_path),
                "n_triggers": int(len(frame)),
                "n_injections": int(len(injections)),
                "pycwb_version": catalog.version,
            }
        )

    combined = pd.concat(trigger_frames, ignore_index=True, sort=False)
    combined.index = pd.RangeIndex(len(combined), name="combined_index")
    combined_injections = (
        pd.concat(injection_frames, ignore_index=True, sort=False)
        if injection_frames
        else pd.DataFrame(
            columns=[
                "run_index",
                "run_name",
                "run_label",
                "run_injection_index",
                "source_row_index",
                *_EMPTY_INJECTION_COLUMNS,
                "run_catalog_file",
                "run_metadata",
            ]
        )
    )
    combined_injections.index = pd.RangeIndex(
        len(combined_injections), name="combined_injection_index"
    )

    triggers_path = _output_path(work_dir, output_file)
    injections_output_file = (
        injections_output_file
        or os.path.splitext(output_file)[0] + "_injections.parquet"
    )
    manifest_file = (
        manifest_file or os.path.splitext(output_file)[0] + "_manifest.json"
    )
    injections_path = _output_path(work_dir, injections_output_file)
    manifest_path = _output_path(work_dir, manifest_file)

    combined.to_parquet(triggers_path, index=True)
    combined_injections.to_parquet(injections_path, index=True)
    manifest = {
        "runs": manifest_runs,
        "n_runs": len(manifest_runs),
        "n_triggers": int(len(combined)),
        "n_injections": int(len(combined_injections)),
        "triggers_file": os.path.abspath(triggers_path),
        "injections_file": os.path.abspath(injections_path),
    }
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    return {
        "output_file": triggers_path,
        "injections_output_file": injections_path,
        "triggers_file": triggers_path,
        "injections_file": injections_path,
        "manifest_file": manifest_path,
        "runs": manifest_runs,
        "n_runs": len(manifest_runs),
        "n_triggers": int(len(combined)),
        "n_injections": int(len(combined_injections)),
    }


__all__ = ["read_catalog_runs"]
