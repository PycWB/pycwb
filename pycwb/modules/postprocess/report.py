"""Composite postprocess report actions."""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

from pycwb.post_production.action_spec import action_spec
from pycwb.modules.postprocess import fake_openbox
from pycwb.modules.postprocess import far
from pycwb.modules.postprocess import report_plots
from pycwb.modules.postprocess import zero_lag

logger = logging.getLogger(__name__)


@action_spec(
    outputs=[],
    inputs=[
        "catalog_file",
        "progress_file",
        "job_ids_file",
        "livetime",
        "zero_lag_catalog_file",
        "zero_lag_job_ids_file",
        "fake_openbox_intervals_file",
    ],
    display_name="Background report",
    description="Generate the standard background FAR and zero-lag report",
    help=(
        "Composite action for common background-report production. It calls "
        "FAR/rho first, then zero-lag and optional fake-openbox reports with "
        "the resulting FAR data."
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
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    far_rho_file: Optional[str] = None,
    output_dir: str = "public",
    include_zero_lag: bool = True,
    zero_lag_catalog_file: Optional[str] = None,
    zero_lag_job_ids_file: Optional[str] = None,
    include_fake_openbox: bool = False,
    fake_openbox_intervals_file: Optional[str] = None,
    ifo_order: Optional[list[str]] = None,
    fake_openbox_n: int = 3,
    fake_openbox_seed: int = 150914,
    public_alerts_file: Optional[str] = None,
    public_alert_time_window: float = 1.0,
    **kwargs,
) -> dict:
    """Generate standard background report artifacts."""
    def _resolve(p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(work_dir, p)

    if far_rho_file:
        far_path = _resolve(far_rho_file)
        with open(far_path) as f:
            binned = json.load(f)
        if "bins" not in binned and "far_rho" in binned:
            binned = binned["far_rho"]
        required = {"bins", "far", "cum_events", "livetime", "ranking_par"}
        missing = required - set(binned.keys())
        if missing:
            raise KeyError(f"far_rho_file missing required keys: {sorted(missing)}")
        out_dir = _resolve(output_dir)
        os.makedirs(out_dir, exist_ok=True)
        report_plots.plot_far_rho(binned, out_dir)
        report_plots.plot_n_events(binned, out_dir)
        json_path = os.path.join(out_dir, "far_rho.json")
        with open(json_path, "w") as f:
            json.dump(binned, f, indent=2)
        logger.info("Loaded binned FAR from %s → %s", far_path, json_path)
        far_result = {"far_rho": binned}
    else:
        far_result = far.far_rho_plot(
            work_dir=work_dir,
            catalog_file=catalog_file,
            progress_file=progress_file,
            job_ids_file=job_ids_file,
            livetime=livetime,
            ranking_par=ranking_par,
            exclude_zero_lag=exclude_zero_lag,
            bin_size=bin_size,
            vmin=vmin,
            vmax=vmax,
            output_dir=output_dir,
            **kwargs,
        )

    result = {"far_rho": far_result}
    if include_zero_lag:
        zero_lag_jobs = zero_lag_job_ids_file
        if zero_lag_catalog_file is None and zero_lag_jobs is None:
            zero_lag_jobs = job_ids_file
        result["zero_lag"] = zero_lag.zero_lag_report(
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
        result["fake_openbox"] = fake_openbox.fake_openbox_report(
            work_dir=work_dir,
            catalog_file=catalog_file,
            intervals_file=fake_openbox_intervals_file,
            far_rho_data=far_result,
            ranking_par=ranking_par,
            output_dir=output_dir,
            ifo_order=ifo_order,
            fake_openbox_n=fake_openbox_n,
            fake_openbox_seed=fake_openbox_seed,
            exclude_zero_lag=exclude_zero_lag,
            **kwargs,
        )
    return result


__all__ = ["standard_background_report"]

