"""Zero-lag postprocess report action."""

from __future__ import annotations

import logging
import os
from typing import Optional

import pandas as pd

from pycwb.post_production.action_spec import action_spec
from pycwb.modules.postprocess import far
from pycwb.modules.postprocess import report_plots
from pycwb.modules.postprocess.lag_filters import (
    try_unshifted_job_ids_from_catalog,
    zero_lag_mask,
)

logger = logging.getLogger(__name__)


@action_spec(
    outputs=[],
    inputs=["catalog_file", "progress_file", "job_ids_file"],
    description="Compute zero-lag significance and plot triggers with FAR",
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
    """Plot zero-lag triggers with FAR values and Poisson significance."""
    def _resolve(p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(work_dir, p)

    cat_path = _resolve(catalog_file)
    prog_path = _resolve(progress_file)
    jobs_path = _resolve(job_ids_file) if job_ids_file else None
    out_dir = _resolve(output_dir)
    trigger_unshifted_jobs = try_unshifted_job_ids_from_catalog(cat_path)
    progress_unshifted_jobs = far.unshifted_jobs_for_progress(prog_path)

    far_rho_data = far.resolve_far_rho_data(far_rho_data, out_dir, kwargs)

    job_ids = far.read_job_ids(jobs_path)
    columns = far.trigger_read_columns(cat_path, ranking_par)
    frames = []
    for chunk in far.iter_parquet_row_groups(cat_path, columns):
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

    zl_livetime = far.sum_progress_livetime(
        prog_path,
        zero_lag=True,
        job_ids=job_ids,
        unshifted_job_ids=progress_unshifted_jobs,
    )
    zl_livetime_years = zl_livetime / 86400.0 / 365.25
    logger.info("Zero-lag live time: %.0f s = %.4f yr", zl_livetime, zl_livetime_years)

    df = far.attach_far_and_significance(df, far_rho_data, ranking_par, zl_livetime)

    os.makedirs(out_dir, exist_ok=True)
    known_candidates = {}
    if public_alerts_file:
        known_candidates = report_plots.load_public_alert_candidates(
            df,
            _resolve(public_alerts_file),
            public_alert_time_window,
        )
    report_plots.plot_zero_lag(
        df,
        ranking_par,
        zl_livetime_years,
        out_dir,
        known_candidates=known_candidates,
        far_rho_data=far_rho_data,
    )

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

