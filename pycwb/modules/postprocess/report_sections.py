"""Section builders and basic-information formatting for the postproduction
report.

Each ``_build_*`` function assembles one tab/section of the report from
artifacts and the reduced summaries produced by
:mod:`pycwb.modules.postprocess.report_summaries`.  The many small ``_format_*``
helpers shape config and workflow values into compact, review-friendly strings.
"""

from __future__ import annotations

import getpass
import logging
import os
import re
import shutil
from datetime import datetime, timezone
from typing import Any, Optional

import pycwb
from pycwb.modules.postprocess.report_context import (
    TEXT_PREVIEW_LIMIT,
    ReportContext,
    _catalog_progress_path,
    _compact_mapping,
    _config_summary,
    _format_cell,
    _format_int,
    _livetime_dict,
    _nested_get,
    _plot_card,
    _read_text_preview,
    _safe_id,
    _to_float,
)
from pycwb.modules.postprocess.report_summaries import (
    _build_scored_bkg_summary,
    _catalog_numeric_summary,
    _interval_summary,
    _load_far_curve_data,
    _matched_sim_summary,
    _progress_figures,
    _progress_summary,
    _read_catalog_metadata,
    _read_csv_table,
    _read_fit_parameter_table,
    _select_bkg_livetime,
    _workflow_steps_by_action,
    _zero_lag_livetime_summary,
)

logger = logging.getLogger(__name__)


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
