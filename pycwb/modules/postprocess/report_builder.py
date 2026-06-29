"""Multi-tab postproduction HTML report builder.

The action in this module assembles a lightweight scientific review page from
existing workflow products.  Large parquet catalogs are summarized using small
column subsets and binned data; full catalogs are never embedded in the HTML.

The implementation is split across helper modules:

* :mod:`pycwb.modules.postprocess.report_context` — :class:`ReportContext`,
  formatting primitives, and parquet/CSV/JSON utilities.
* :mod:`pycwb.modules.postprocess.report_summaries` — data readers, reducers,
  and figure builders.
* :mod:`pycwb.modules.postprocess.report_sections` — per-tab section builders.
* :mod:`pycwb.modules.postprocess.report_render` — Jinja rendering glue.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

from pycwb.post_production.action_spec import action_spec
from pycwb.modules.postprocess.report_context import (
    DEFAULT_MAX_BINS,
    DEFAULT_MAX_POINTS,
    DEFAULT_TABLE_LIMIT,
    ReportContext,
    _jsonable,
    _livetime_dict,
    _parquet_file_info,
    _resolve_path,
)
from pycwb.modules.postprocess.report_render import _render_report_html
from pycwb.modules.postprocess.report_sections import (
    _build_basic_information,
    _build_bkg_section,
    _build_metadata,
    _build_simulation_sections,
    _build_summary,
    _build_training_section,
    _build_workflow_section,
    _copy_into_output_dir,
)
from pycwb.modules.postprocess.report_summaries import (
    _load_workflow_yaml,
    _zero_lag_livetime_summary,
)

logger = logging.getLogger(__name__)

__all__ = [
    "postproduction_report",
    "ReportContext",
    "_livetime_dict",
    "_parquet_file_info",
    "_zero_lag_livetime_summary",
]


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
