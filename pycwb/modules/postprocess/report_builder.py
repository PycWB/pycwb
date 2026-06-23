"""Multi-tab postproduction HTML report builder.

The action in this module assembles a lightweight scientific review page from
existing workflow products.  Large parquet catalogs are summarized using small
column subsets and binned data; full catalogs are never embedded in the HTML.
"""

from __future__ import annotations

import getpass
import json
import logging
import math
import os
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from jinja2 import Environment, FileSystemLoader, select_autoescape

import pycwb
from pycwb.modules.catalog.catalog import Catalog
from pycwb.modules.postprocess.lag_filters import (
    try_unshifted_job_ids_from_catalog,
    zero_lag_mask,
)
from pycwb.post_production.action_spec import action_spec

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


@action_spec(
    outputs=["output_file", "data_file"],
    inputs=["workflow_file", "production_catalog_file"],
    display_name="Postproduction web report",
    description="Build a multi-tab postproduction HTML report from workflow outputs",
    help=(
        "This final workflow action reads existing postproduction artifacts, "
        "catalog metadata, and reduced parquet/CSV summaries to produce a "
        "review-ready HTML page and companion JSON data file."
    ),
)
def postproduction_report(
    work_dir: str,
    workflow_file: str,
    production_catalog_file: str,
    title: str = "pycWB postproduction report",
    output_file: str = "public/postproduction_report/index.html",
    data_file: Optional[str] = None,
    bkg: Optional[dict[str, Any]] = None,
    training: Optional[dict[str, Any]] = None,
    simulation_runs: Optional[list[dict[str, Any]]] = None,
    max_plot_points: int = DEFAULT_MAX_POINTS,
    max_bins: int = DEFAULT_MAX_BINS,
    table_limit: int = DEFAULT_TABLE_LIMIT,
    **kwargs,
) -> dict:
    """Build a multi-tab postproduction report.

    Parameters are intentionally workflow-friendly: nested ``bkg``,
    ``training`` and ``simulation_runs`` dictionaries point at existing
    artifacts, and missing optional files are recorded rather than fatal.
    ``bkg.zero_lag_reference_catalog_file`` may be provided when zero-lag
    progress belongs to a different original catalog than
    ``production_catalog_file``.
    """
    work_dir = os.path.abspath(str(work_dir))
    output_path = _resolve_path(work_dir, output_file)
    output_dir = os.path.dirname(output_path) or "."
    data_path = _resolve_path(
        work_dir,
        data_file or os.path.join(os.path.dirname(output_file) or ".", "report_data.json"),
    )
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(data_path) or ".", exist_ok=True)

    ctx = ReportContext(work_dir=work_dir, output_dir=output_dir)
    workflow_path = ctx.resolve(workflow_file)
    workflow_copy_path = _copy_into_output_dir(
        ctx,
        workflow_path,
        os.path.basename(workflow_path or str(workflow_file)),
    )
    production_artifact = ctx.register_artifact(
        production_catalog_file,
        label="Production catalog",
        kind="parquet",
        required=True,
    )
    workflow_artifact = ctx.register_artifact(
        workflow_copy_path,
        label="Workflow YAML",
        kind="yaml",
        required=True,
    )

    workflow_text, workflow_data = _load_workflow_yaml(workflow_path)
    metadata = _build_metadata(
        ctx=ctx,
        title=title,
        workflow_file=workflow_copy_path,
        production_catalog_file=production_catalog_file,
    )
    bkg_data = _build_bkg_section(
        ctx=ctx,
        bkg=bkg or {},
        production_catalog_file=production_catalog_file,
        max_bins=max_bins,
        max_plot_points=max_plot_points,
        table_limit=table_limit,
    )
    training_data = _build_training_section(
        ctx=ctx,
        training=training or {},
        workflow_data=workflow_data,
        max_bins=max_bins,
        table_limit=table_limit,
    )
    simulation_data = _build_simulation_sections(
        ctx=ctx,
        simulation_runs=simulation_runs or [],
        table_limit=table_limit,
    )
    workflow_section = _build_workflow_section(
        ctx=ctx,
        workflow_file=workflow_file,
        workflow_text=workflow_text,
        workflow_artifact=workflow_artifact,
    )
    basic_info = _build_basic_information(
        metadata=metadata,
        bkg_data=bkg_data,
        training_data=training_data,
        simulation_data=simulation_data,
        workflow_data=workflow_data,
        production_artifact=production_artifact,
        workflow_artifact=workflow_artifact,
    )

    data: dict[str, Any] = {
        "title": title,
        "metadata": metadata,
        "basic_info": basic_info,
        "summary": _build_summary(metadata, bkg_data, training_data, simulation_data),
        "bkg": bkg_data,
        "training": training_data,
        "simulation_runs": simulation_data,
        "workflow": workflow_section,
        "artifacts": ctx.artifacts,
        "missing_artifacts": ctx.missing_artifacts,
        "production_artifact": production_artifact,
        "generated_files": {
            "html": ctx.display_path(output_path),
            "data": ctx.display_path(data_path),
        },
    }
    data = _jsonable(data)

    with open(data_path, "w") as f:
        json.dump(data, f, indent=2)

    html = _render_report_html(data)
    with open(output_path, "w") as f:
        f.write(html)

    n_tabs = 4 + len(simulation_data)
    logger.info("Postproduction report written to %s", output_path)
    logger.info("Postproduction report data written to %s", data_path)

    return {
        "output_file": output_file,
        "data_file": ctx.display_path(data_path),
        "n_tabs": n_tabs,
        "missing_artifacts": ctx.missing_artifacts,
    }


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------


def _build_metadata(
    ctx: ReportContext,
    title: str,
    workflow_file: str,
    production_catalog_file: str,
) -> dict[str, Any]:
    catalog_path = ctx.resolve(production_catalog_file)
    catalog_meta = _read_catalog_metadata(catalog_path)
    config = catalog_meta.get("config") or {}
    jobs = catalog_meta.get("jobs") or []
    created_utc = datetime.now(timezone.utc)

    return {
        "title": title,
        "creator": getpass.getuser(),
        "created_utc": created_utc.isoformat(timespec="seconds"),
        "created_local": created_utc.astimezone().isoformat(timespec="seconds"),
        "production_pycwb_version": catalog_meta.get("version") or "",
        "postproduction_pycwb_version": getattr(pycwb, "__version__", ""),
        "workflow_file": ctx.display_path(workflow_file),
        "production_catalog_file": ctx.display_path(production_catalog_file),
        "config_summary": _config_summary(config),
        "config_metadata": _compact_mapping(config, limit=80),
        "n_catalog_jobs": len(jobs) if isinstance(jobs, list) else None,
    }


def _build_bkg_section(
    ctx: ReportContext,
    bkg: dict[str, Any],
    production_catalog_file: str,
    max_bins: int,
    max_plot_points: int,
    table_limit: int,
) -> dict[str, Any]:
    ranking_par = str(bkg.get("ranking_par") or "rho")
    livetime = _to_float(bkg.get("livetime"))

    scored_artifact = ctx.register_artifact(
        bkg.get("scored_catalog"),
        label="BKG scored catalog",
        kind="parquet",
    )
    ctx.register_artifact(bkg.get("far_json"), label="Full FAR/rho JSON", kind="json")
    ctx.register_artifact(bkg.get("binned_far_json"), label="Binned FAR/rho JSON", kind="json")
    progress_artifact = ctx.register_artifact(
        bkg.get("progress_file"),
        label="BKG progress",
        kind="parquet",
    )
    intervals_artifact = ctx.register_artifact(
        bkg.get("intervals_file"),
        label="BKG live-time intervals",
        kind="parquet",
    )
    zero_lag_artifact = ctx.register_artifact(
        bkg.get("zero_lag_csv"),
        label="Zero-lag trigger table",
        kind="csv",
    )
    zero_lag_progress_file = bkg.get("zero_lag_progress_file") or _catalog_progress_path(
        ctx.resolve(bkg.get("zero_lag_catalog_file") or production_catalog_file)
    )
    zero_lag_reference_catalog_file = (
        bkg.get("zero_lag_reference_catalog_file")
        or production_catalog_file
    )
    zero_lag_progress_artifact = ctx.register_artifact(
        zero_lag_progress_file,
        label="Zero-lag progress",
        kind="parquet",
    )
    plot_cards = [
        _plot_card(ctx, entry)
        for entry in bkg.get("plots", []) or []
    ]

    far_data = _load_far_curve_data(
        ctx=ctx,
        binned_far_json=bkg.get("binned_far_json"),
        far_json=bkg.get("far_json"),
        max_plot_points=max_plot_points,
    )
    progress_summary = _progress_summary(ctx.resolve(bkg.get("progress_file")))
    interval_summary = _interval_summary(ctx.resolve(bkg.get("intervals_file")))
    zero_lag_livetime = _zero_lag_livetime_summary(
        progress_path=ctx.resolve(zero_lag_progress_file),
        catalog_path=ctx.resolve(zero_lag_reference_catalog_file),
    )
    livetime_choice = _select_bkg_livetime(
        explicit_livetime=livetime,
        progress_summary=progress_summary,
        interval_summary=interval_summary,
        far_data=far_data,
    )
    effective_livetime = livetime_choice["seconds"]
    scored_summary = _build_scored_bkg_summary(
        ctx=ctx,
        catalog_file=bkg.get("scored_catalog"),
        ranking_par=ranking_par,
        livetime=effective_livetime,
        max_bins=max_bins,
        table_limit=table_limit,
    )
    zero_lag_table = _read_csv_table(
        ctx.resolve(bkg.get("zero_lag_csv")),
        table_limit=table_limit,
        preferred_columns=[
            "id", "job_id", "lag_idx", ranking_par, "xgb_prob", "ifar",
            "ifar_years", "far_attached", "significance", "p_value",
            "gps_time", "net_cc", "likelihood", "coherent_energy",
        ],
        sort_columns=["significance", ranking_par, "ifar_years"],
    )

    figures = []
    figures.extend(far_data.get("figures", []))
    figures.extend(scored_summary.get("figures", []))
    figures.extend(_progress_figures(progress_summary))

    return {
        "ranking_par": ranking_par,
        "artifacts": [
            item for item in [
                scored_artifact,
                progress_artifact,
                intervals_artifact,
                zero_lag_artifact,
                zero_lag_progress_artifact,
            ] if item
        ],
        "plots": plot_cards,
        "far_curve": far_data,
        "scored_catalog": scored_summary,
        "progress": progress_summary,
        "intervals": interval_summary,
        "zero_lag_table": zero_lag_table,
        "zero_lag_livetime": zero_lag_livetime,
        "figures": figures,
        "livetime": _livetime_dict(effective_livetime),
        "livetime_source": livetime_choice["source"],
        "livetime_warning": livetime_choice.get("warning", ""),
    }


def _build_training_section(
    ctx: ReportContext,
    training: dict[str, Any],
    workflow_data: dict[str, Any],
    max_bins: int,
    table_limit: int,
) -> dict[str, Any]:
    bkg_artifact = ctx.register_artifact(
        training.get("bkg_catalog"),
        label="Training BKG catalog",
        kind="parquet",
    )
    bkg_progress_artifact = ctx.register_artifact(
        training.get("bkg_progress_file"),
        label="Training BKG progress",
        kind="parquet",
    )
    bkg_intervals_artifact = ctx.register_artifact(
        training.get("bkg_intervals_file"),
        label="Training BKG intervals",
        kind="parquet",
    )
    sim_artifact = ctx.register_artifact(
        training.get("sim_catalog"),
        label="Training SIM catalog",
        kind="parquet",
    )
    model_artifact = ctx.register_artifact(
        training.get("model_file"),
        label="XGBoost model",
        kind="model",
    )
    config_artifact = ctx.register_artifact(
        training.get("config_file"),
        label="XGBoost config",
        kind="python",
    )
    settings_artifact = ctx.register_artifact(
        training.get("training_settings_file"),
        label="XGBoost training settings",
        kind="text",
    )
    output_artifact = ctx.register_artifact(
        training.get("training_output_file"),
        label="XGBoost training output",
        kind="text",
    )

    bkg_stats = _catalog_numeric_summary(
        ctx.resolve(training.get("bkg_catalog")),
        numeric_columns=["rho", "xgb_prob", "net_cc", "likelihood", "coherent_energy"],
        max_bins=max_bins,
    )
    sim_stats = _catalog_numeric_summary(
        ctx.resolve(training.get("sim_catalog")),
        numeric_columns=["rho", "xgb_prob", "net_cc", "likelihood", "coherent_energy", "sim_hrss"],
        max_bins=max_bins,
    )
    workflow_steps = _workflow_steps_by_action(
        workflow_data,
        action_fragments=[
            "postprocess.selection.trigger_selection",
            "postprocess.selection.filter_real_simulation",
            "postprocess.train_xgboost.train_xgboost",
        ],
    )
    config_text = _read_text_preview(ctx.resolve(training.get("config_file")), TEXT_PREVIEW_LIMIT)

    return {
        "artifacts": [
            item for item in [
                bkg_artifact,
                bkg_progress_artifact,
                bkg_intervals_artifact,
                sim_artifact,
                model_artifact,
                config_artifact,
                settings_artifact,
                output_artifact,
            ] if item
        ],
        "bkg_stats": bkg_stats,
        "sim_stats": sim_stats,
        "bkg_progress": _progress_summary(ctx.resolve(training.get("bkg_progress_file"))),
        "bkg_intervals": _interval_summary(ctx.resolve(training.get("bkg_intervals_file"))),
        "workflow_steps": workflow_steps,
        "config_text": config_text,
        "placeholders": [
            "Training curves will appear here once train_xgboost persists evaluation history.",
            "Feature importance will appear here once the model diagnostic artifact is available.",
        ],
    }


def _build_simulation_sections(
    ctx: ReportContext,
    simulation_runs: list[dict[str, Any]],
    table_limit: int,
) -> list[dict[str, Any]]:
    sections = []
    for idx, run in enumerate(simulation_runs):
        label = str(run.get("label") or f"Simulation {idx + 1}")
        scored_artifact = ctx.register_artifact(
            run.get("scored_catalog"),
            label=f"{label} scored catalog",
            kind="parquet",
        )
        matched_artifact = ctx.register_artifact(
            run.get("matched_file"),
            label=f"{label} matched simulations",
            kind="parquet",
        )
        plot_cards = [_plot_card(ctx, entry) for entry in run.get("plots", []) or []]
        fit_tables = []
        for fit_file in run.get("fit_parameter_files", []) or []:
            artifact = ctx.register_artifact(
                fit_file,
                label=f"{label} fit parameters",
                kind="csv",
            )
            table = _read_fit_parameter_table(ctx.resolve(fit_file), table_limit=table_limit)
            table["artifact"] = artifact
            fit_tables.append(table)

        sections.append({
            "id": _safe_id(label),
            "label": label,
            "artifacts": [
                item for item in [scored_artifact, matched_artifact] if item
            ],
            "plots": plot_cards,
            "scored_stats": _catalog_numeric_summary(
                ctx.resolve(run.get("scored_catalog")),
                numeric_columns=["rho", "xgb_prob", "net_cc", "likelihood", "coherent_energy"],
            ),
            "matched_summary": _matched_sim_summary(ctx.resolve(run.get("matched_file"))),
            "fit_tables": fit_tables,
            "placeholders": [
                "Additional hrss10/hrss50/hrss90 rows will appear when more IFAR fit CSVs are provided.",
            ],
        })
    return sections


def _build_workflow_section(
    ctx: ReportContext,
    workflow_file: str,
    workflow_text: str,
    workflow_artifact: Optional[dict[str, Any]],
) -> dict[str, Any]:
    workflow_path = ctx.resolve(workflow_file)
    workflow_dir = os.path.dirname(workflow_path or ctx.work_dir)
    candidates = [
        os.path.join(ctx.work_dir, "workflow_diagram.png"),
        os.path.join(workflow_dir, "workflow_diagram.png"),
    ]
    png_path = next((path for path in candidates if os.path.exists(path)), candidates[0])
    html_candidates = [
        os.path.join(ctx.work_dir, "workflow_diagram.html"),
        os.path.join(workflow_dir, "workflow_diagram.html"),
    ]
    html_path = next((path for path in html_candidates if os.path.exists(path)), html_candidates[0])
    png_path = _copy_into_output_dir(ctx, png_path, "workflow_diagram.png")
    html_path = _copy_into_output_dir(ctx, html_path, "workflow_diagram.html")

    diagram_png = ctx.register_artifact(
        png_path,
        label="Workflow diagram PNG",
        kind="image",
    )
    diagram_html = ctx.register_artifact(
        html_path,
        label="Workflow diagram HTML",
        kind="html",
    )

    return {
        "workflow_artifact": workflow_artifact,
        "workflow_text": workflow_text,
        "diagram_png": diagram_png,
        "diagram_html": diagram_html,
    }


def _copy_into_output_dir(
    ctx: ReportContext,
    source_path: Optional[str],
    output_name: str,
) -> str:
    destination = os.path.join(ctx.output_dir, output_name)
    if source_path and os.path.exists(source_path):
        try:
            if os.path.abspath(source_path) != os.path.abspath(destination):
                shutil.copy2(source_path, destination)
        except OSError as exc:
            logger.warning("Could not copy %s to report output: %s", source_path, exc)
    return destination


def _build_basic_information(
    metadata: dict[str, Any],
    bkg_data: dict[str, Any],
    training_data: dict[str, Any],
    simulation_data: list[dict[str, Any]],
    workflow_data: dict[str, Any],
    production_artifact: Optional[dict[str, Any]],
    workflow_artifact: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """Build a compact cWB-style basic-information summary."""
    config = metadata.get("config_metadata") or {}
    bkg_split_step = _workflow_step_by_id(workflow_data, "bkg_split")
    far_step = _workflow_step_by_id(workflow_data, "far_rho")
    report_step = _workflow_step_by_id(workflow_data, "background_report")
    model_step = _workflow_step_by_id(workflow_data, "model")

    ranking_par = (
        _nested_get(far_step, ["args", "ranking_par"])
        or _nested_get(report_step, ["args", "ranking_par"])
        or bkg_data.get("ranking_par")
        or "rho"
    )

    run_parameters = [
        _info_item("IFOs", _join_value(_config_value(config, "ifo") or _config_value(config, "nIFO")), accent=True),
        _info_item("Search", _join_nonempty([
            _config_value(config, "cfg_search"),
            _config_value(config, "Search"),
        ], sep=" / "), accent=True),
        _info_item("Frequency band", _format_band(_config_value(config, "fLow"), _config_value(config, "fHigh"), "Hz"), accent=True),
        _info_item("Analysis rate", _format_rate_block(config), accent=True),
        _info_item("Lag plan", _format_lag_plan(config), accent=True),
        _info_item("Superlags", _format_superlag_plan(config), accent=True),
        _info_item("Pixel selection", _format_thresholds(config, ["bpp", "netRHO", "netCC"]), accent=True),
        _info_item("Subnet cuts", _format_thresholds(config, ["subnet", "subcut", "subnorm"])),
        _info_item("Cluster gaps", _format_thresholds(config, ["Tgap", "Fgap", "LOUD"])),
        _info_item("Sky map", _format_skymap(config)),
        _info_item("Segmenting", _format_thresholds(config, ["segEdge", "segMLS", "segTHR"])),
        _info_item("DQ definitions", _format_dq_summary(config)),
    ]

    workflow_vars = workflow_data.get("vars", {}) if isinstance(workflow_data, dict) else {}
    split = _resolve_workflow_refs(_nested_get(bkg_split_step, ["args", "split"]) or {}, workflow_vars)
    fractions = split.get("fractions") if isinstance(split, dict) else {}
    selection_cuts = [
        _info_item("Ranking statistic", ranking_par, accent=True),
        _info_item("BKG split", _format_split(split, fractions), accent=True),
        _info_item("Training action", _nested_get(model_step, ["action"]) or ""),
        _info_item(
            "Report output",
            _resolve_workflow_refs(_nested_get(report_step, ["args", "output_dir"]) or "", workflow_vars),
        ),
        _info_item("Zero lag", _enabled_label(_nested_get(report_step, ["args", "include_zero_lag"]))),
        _info_item("Fake openbox", _format_fake_openbox(report_step)),
        _info_item("Public alert window", _format_public_alert_window(report_step)),
        _info_item("Simulation runs", _format_int(len(simulation_data))),
    ]

    bkg_livetime = bkg_data.get("livetime") or _livetime_dict(None)
    zero_lag = _nested_get(bkg_data, ["zero_lag_livetime", "livetime"]) or _livetime_dict(None)
    interval_livetime = _nested_get(bkg_data, ["intervals", "total_livetime"]) or _livetime_dict(None)
    progress_livetime = _nested_get(bkg_data, ["progress", "total_livetime"]) or _livetime_dict(None)
    livetimes = [
        _info_item("Zero lag", _livetime_detail(zero_lag), accent=True),
        _info_item("FAR non-zero lag", _livetime_detail(bkg_livetime), accent=True),
        _info_item("Progress total", _livetime_detail(progress_livetime)),
        _info_item("Interval total", _livetime_detail(interval_livetime)),
        _info_item("Live-time source", bkg_data.get("livetime_source") or ""),
        _info_item("Completed progress jobs", _format_int(_nested_get(bkg_data, ["progress", "n_jobs"]))),
        _info_item("Live-time intervals", _format_int(_nested_get(bkg_data, ["intervals", "table", "n_rows"]))),
        _info_item("Zero-lag rows", _format_int(_nested_get(bkg_data, ["zero_lag_livetime", "n_rows"]))),
    ]

    review_links = [
        _artifact_link("Production catalog", production_artifact),
        _artifact_link("Workflow YAML", workflow_artifact),
    ]
    for artifact in training_data.get("artifacts", []) or []:
        if artifact.get("label") in {
            "XGBoost config",
            "XGBoost model",
            "XGBoost training settings",
            "XGBoost training output",
        }:
            review_links.append(_artifact_link(artifact.get("label"), artifact))
    for artifact in bkg_data.get("artifacts", []) or []:
        if artifact.get("label") in {"BKG scored catalog", "Zero-lag trigger table", "BKG progress"}:
            review_links.append(_artifact_link(artifact.get("label"), artifact))

    return {
        "run_parameters": [item for item in run_parameters if item["value"]],
        "selection_cuts": [item for item in selection_cuts if item["value"]],
        "livetimes": [item for item in livetimes if item["value"]],
        "review_links": [item for item in review_links if item],
    }


def _workflow_step_by_id(workflow_data: dict[str, Any], step_id: str) -> dict[str, Any]:
    for step in workflow_data.get("steps", []) or []:
        if str(step.get("id") or "") == step_id:
            return step
    return {}


def _info_item(label: str, value: Any, accent: bool = False) -> dict[str, Any]:
    return {
        "label": label,
        "value": _join_value(value),
        "accent": bool(accent),
    }


def _artifact_link(label: str, artifact: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if not artifact:
        return None
    return {
        "label": label,
        "path": artifact.get("path") or "",
        "href": artifact.get("href") or "",
        "exists": bool(artifact.get("exists")),
        "kind": artifact.get("kind") or "file",
    }


def _config_value(config: dict[str, Any], key: str) -> Any:
    return config.get(key) if isinstance(config, dict) else None


def _join_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return ", ".join(_join_value(item) for item in value if _join_value(item))
    if isinstance(value, dict):
        return ", ".join(f"{key}: {_join_value(item)}" for key, item in value.items())
    return _format_cell(value)


def _join_nonempty(values: list[Any], sep: str = " - ") -> str:
    parts = [_join_value(value) for value in values]
    return sep.join(part for part in parts if part)


def _format_band(low: Any, high: Any, unit: str) -> str:
    if low is None and high is None:
        return ""
    return f"{_format_cell(low)} - {_format_cell(high)} {unit}".strip()


def _format_rate_block(config: dict[str, Any]) -> str:
    parts = []
    for key, label in [
        ("inRate", "input"),
        ("fResample", "resample"),
        ("rateANA", "analysis"),
    ]:
        value = _config_value(config, key)
        if value is not None:
            parts.append(f"{label}: {_format_cell(value)} Hz")
    return " / ".join(parts)


def _format_lag_plan(config: dict[str, Any]) -> str:
    parts = []
    for key in ["lagSize", "lagStep", "lagOff", "lagMax"]:
        value = _config_value(config, key)
        if value is not None:
            suffix = " s" if key == "lagStep" else ""
            parts.append(f"{key}: {_format_cell(value)}{suffix}")
    return " / ".join(parts)


def _format_superlag_plan(config: dict[str, Any]) -> str:
    parts = []
    for key in ["slagSize", "slagMin", "slagMax", "slagOff"]:
        value = _config_value(config, key)
        if value is not None:
            parts.append(f"{key}: {_format_cell(value)}")
    return " / ".join(parts)


def _format_thresholds(config: dict[str, Any], keys: list[str]) -> str:
    parts = []
    for key in keys:
        value = _config_value(config, key)
        if value is not None:
            parts.append(f"{key}: {_format_cell(value)}")
    return " / ".join(parts)


def _format_skymap(config: dict[str, Any]) -> str:
    healpix = _to_float(_config_value(config, "healpix"))
    if healpix is None:
        return ""
    order = int(healpix)
    nside = 2 ** order
    pixels = 12 * nside * nside
    pixel_area = 41252.96124941927 / pixels
    return f"healpix order {order} / {pixels:,} pixels / {pixel_area:.4g} deg^2 per pixel"


def _format_dq_summary(config: dict[str, Any]) -> str:
    dqf = _config_value(config, "DQF")
    if not isinstance(dqf, list):
        return ""
    categories = []
    ifos = []
    for row in dqf:
        if isinstance(row, (list, tuple)) and len(row) >= 3:
            ifos.append(str(row[0]))
            categories.append(str(row[2]))
    category_label = ", ".join(sorted(set(categories)))
    ifo_label = ", ".join(sorted(set(ifos)))
    return f"{len(dqf)} entries / {ifo_label} / {category_label}".strip(" /")


def _format_split(split: Any, fractions: Any) -> str:
    if not isinstance(split, dict):
        return ""
    by = split.get("by")
    seed = split.get("seed")
    fraction_text = _join_value(fractions) if fractions else ""
    return _join_nonempty([
        f"by {by}" if by else "",
        f"fractions {fraction_text}" if fraction_text else "",
        f"seed {seed}" if seed is not None else "",
    ], sep=" / ")


def _resolve_workflow_refs(value: Any, variables: dict[str, Any]) -> Any:
    """Resolve simple workflow ${var} and ${nested.var} references for display."""
    if isinstance(value, dict):
        return {key: _resolve_workflow_refs(item, variables) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_workflow_refs(item, variables) for item in value]
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text.startswith("${") and text.endswith("}"):
        key = text[2:-1]
        found = _lookup_workflow_var(variables, key)
        return found if found is not None else value

    def replace(match: re.Match[str]) -> str:
        found = _lookup_workflow_var(variables, match.group(1))
        return str(found) if found is not None else match.group(0)

    return re.sub(r"\$\{([^}]+)\}", replace, value)


def _lookup_workflow_var(variables: dict[str, Any], key: str) -> Any:
    current: Any = variables
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _enabled_label(value: Any) -> str:
    if value is None:
        return ""
    return "enabled" if bool(value) else "disabled"


def _format_fake_openbox(report_step: dict[str, Any]) -> str:
    enabled = _nested_get(report_step, ["args", "include_fake_openbox"])
    if enabled is None:
        return ""
    parts = [_enabled_label(enabled)]
    n = _nested_get(report_step, ["args", "fake_openbox_n"])
    seed = _nested_get(report_step, ["args", "fake_openbox_seed"])
    if n is not None:
        parts.append(f"n={n}")
    if seed is not None:
        parts.append(f"seed={seed}")
    return " / ".join(parts)


def _format_public_alert_window(report_step: dict[str, Any]) -> str:
    window = _nested_get(report_step, ["args", "public_alert_time_window"])
    if window is None:
        return ""
    return f"{_format_cell(window)} s"


def _livetime_detail(livetime: dict[str, Any]) -> str:
    seconds = livetime.get("seconds") if isinstance(livetime, dict) else None
    if seconds is None:
        return ""
    return (
        f"{livetime.get('seconds_label', '')} = "
        f"{livetime.get('days_label', '')} = "
        f"{livetime.get('years_label', '')}"
    )


def _build_summary(
    metadata: dict[str, Any],
    bkg_data: dict[str, Any],
    training_data: dict[str, Any],
    simulation_data: list[dict[str, Any]],
) -> dict[str, Any]:
    bkg_rows = _nested_get(bkg_data, ["scored_catalog", "info", "rows"])
    bkg_livetime = bkg_data.get("livetime") or _livetime_dict(None)
    zero_lag_livetime = _nested_get(bkg_data, ["zero_lag_livetime", "livetime"]) or _livetime_dict(None)
    train_bkg_rows = _nested_get(training_data, ["bkg_stats", "info", "rows"])
    train_sim_rows = _nested_get(training_data, ["sim_stats", "info", "rows"])
    sim_rows = sum(
        int(_nested_get(run, ["scored_stats", "info", "rows"]) or 0)
        for run in simulation_data
    )
    compact_items = [
        {"label": "Production pycWB", "value": metadata.get("production_pycwb_version") or "unknown"},
        {"label": "Postproduction pycWB", "value": metadata.get("postproduction_pycwb_version") or "unknown"},
        {"label": "Creator", "value": metadata.get("creator") or ""},
        {"label": "Created", "value": metadata.get("created_local") or ""},
        {"label": "BKG triggers", "value": _format_int(bkg_rows)},
        {"label": "BKG live time", "value": bkg_livetime.get("bkg_compact_label", "unknown")},
        {"label": "Zero-lag live time", "value": zero_lag_livetime.get("zero_lag_compact_label", "unknown")},
        {"label": "Training BKG / SIM", "value": f"{_format_int(train_bkg_rows)} / {_format_int(train_sim_rows)}"},
        {"label": "SIM evaluated", "value": _format_int(sim_rows)},
    ]
    return {
        "items": compact_items,
        "cards": [
            {"label": "Production pycWB", "value": metadata.get("production_pycwb_version") or "unknown"},
            {"label": "Postproduction pycWB", "value": metadata.get("postproduction_pycwb_version") or "unknown"},
            {"label": "BKG triggers", "value": _format_int(bkg_rows)},
            {"label": "BKG live time", "value": bkg_livetime.get("years_label", "unknown")},
            {"label": "Zero-lag live time", "value": zero_lag_livetime.get("days_label", "unknown")},
            {"label": "Training BKG", "value": _format_int(train_bkg_rows)},
            {"label": "Training SIM", "value": _format_int(train_sim_rows)},
            {"label": "SIM evaluated", "value": _format_int(sim_rows)},
            {"label": "Creator", "value": metadata.get("creator") or ""},
        ],
    }


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


# ---------------------------------------------------------------------------
# Rendering and formatting
# ---------------------------------------------------------------------------


def _render_report_html(data: dict[str, Any]) -> str:
    template_dir = Path(__file__).with_name("templates")
    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template("postproduction_report.html.j2")
    return template.render(
        report=data,
        embedded_data_json=json.dumps(data, ensure_ascii=True),
    )


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
