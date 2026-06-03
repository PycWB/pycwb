"""Job-based selection and catalog filtering — workflow-compatible.

Selects whole jobs (by ``job_id``) so that live time can be correctly
accumulated from the progress file.  This is essential for FAR computation
where the live time must match the selected data.

Workflow actions
----------------
``postprocess.job_selector.select_jobs_by_livetime``
    Read a progress file, randomly select job_ids whose summed live time
    approximates *fraction* of the total (non-zero-lag) live time.

``postprocess.job_selector.filter_catalog_by_jobs``
    Read a catalog parquet and keep only rows whose ``job_id`` is in a
    previously-saved job list.

``postprocess.job_selector.compute_livetime``
    Sum the live time from a progress file for a given set of job_ids.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd

from pycwb.post_production.action_spec import action_spec

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# select_jobs_by_livetime
# ---------------------------------------------------------------------------

@action_spec(
    outputs=['output_file'],
    inputs=['progress_file'],
    description='Select random subset of jobs by live time fraction',
)
def select_jobs_by_livetime(
    work_dir: str,
    progress_file: str,
    output_file: str,
    fraction: float = 0.10,
    exclude_zero_lag: bool = True,
    seed: int = 150914,
    **kwargs,
) -> dict:
    """Select a random subset of job_ids representing ~*fraction* of live time.

    Parameters
    ----------
    work_dir : str
        Base directory for relative path resolution.
    progress_file : str
        Path to ``progress.M1.parquet`` (relative to *work_dir*).
    output_file : str
        Path where the newline-separated job_id list is written.
    fraction : float
        Target fraction of total live time (default 0.10 = 10%).
    exclude_zero_lag : bool
        If True, exclude ``lag_idx == 0`` rows from live-time calculation.
    seed : int
        Random seed.

    Returns
    -------
    dict
        ``job_ids_file``, ``n_jobs_selected``, ``n_jobs_total``,
        ``livetime_selected``, ``livetime_total``.
    """
    def _resolve(p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(work_dir, p)

    prog_path = _resolve(progress_file)
    out_path = _resolve(output_file)

    df = pd.read_parquet(prog_path)
    if exclude_zero_lag and "lag_idx" in df.columns:
        df = df[df["lag_idx"] != 0]

    # Livetime per job
    job_lt = df.groupby("job_id")["livetime"].sum()
    total_lt = job_lt.sum()
    n_total = len(job_lt)

    # Randomly select jobs until cumulative live time >= fraction * total
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(job_lt.index)
    cumsum = 0.0
    selected = []
    target = fraction * total_lt
    for jid in shuffled:
        selected.append(jid)
        cumsum += job_lt.loc[jid]
        if cumsum >= target:
            break

    n_sel = len(selected)
    sel_lt = job_lt.loc[selected].sum()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        for jid in sorted(selected):
            f.write(f"{jid}\n")

    logger.info(
        "Selected %d / %d jobs (%.1f%% of live time: %.0f / %.0f s)",
        n_sel, n_total, sel_lt / max(total_lt, 1) * 100, sel_lt, total_lt,
    )
    logger.info("Job list written to %s", out_path)

    return {
        "job_ids_file": output_file,
        "n_jobs_selected": n_sel,
        "n_jobs_total": n_total,
        "livetime_selected": float(sel_lt),
        "livetime_total": float(total_lt),
    }


# ---------------------------------------------------------------------------
# filter_catalog_by_jobs
# ---------------------------------------------------------------------------

@action_spec(
    outputs=['output_file'],
    inputs=['input_file', 'job_ids_file'],
    description='Filter catalog parquet by job ID list',
)
def filter_catalog_by_jobs(
    work_dir: str,
    input_file: str,
    job_ids_file: str,
    output_file: str,
    **kwargs,
) -> dict:
    """Filter a catalog parquet, keeping only rows with selected job_ids.

    Parameters
    ----------
    work_dir : str
        Base directory.
    input_file : str
        Path to input catalog parquet.
    job_ids_file : str
        Path to the job list file (one job_id per line).
    output_file : str
        Path for the filtered output parquet.

    Returns
    -------
    dict
        ``filtered_file``, ``n_before``, ``n_after``.
    """
    def _resolve(p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(work_dir, p)

    in_path = _resolve(input_file)
    jobs_path = _resolve(job_ids_file)
    out_path = _resolve(output_file)

    with open(jobs_path) as f:
        job_ids = {int(line.strip()) for line in f if line.strip()}
    logger.info("Loaded %d job_ids from %s", len(job_ids), jobs_path)

    df = pd.read_parquet(in_path)
    n_before = len(df)
    df = df[df["job_id"].isin(job_ids)].reset_index(drop=True)
    n_after = len(df)
    logger.info("Filtered %d → %d rows", n_before, n_after)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_parquet(out_path, index=False)
    logger.info("Wrote %s", out_path)

    return {
        "filtered_file": output_file,
        "n_before": n_before,
        "n_after": n_after,
    }


# ---------------------------------------------------------------------------
# compute_livetime
# ---------------------------------------------------------------------------

@action_spec(
    outputs=[],
    inputs=['progress_file', 'job_ids_file'],
    description='Compute total live time for selected jobs (result spread into global context)',
)
def compute_livetime(
    work_dir: str,
    progress_file: str,
    job_ids_file: str,
    exclude_zero_lag: bool = True,
    **kwargs,
) -> dict:
    """Sum live time from a progress file for the selected jobs.

    Parameters
    ----------
    work_dir : str
        Base directory.
    progress_file : str
        Path to progress parquet.
    job_ids_file : str
        Path to job list file.
    exclude_zero_lag : bool
        Exclude ``lag_idx == 0`` from the sum.

    Returns
    -------
    dict
        ``livetime`` (float, seconds), ``livetime_days`` (float).
    """
    def _resolve(p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(work_dir, p)

    prog_path = _resolve(progress_file)
    jobs_path = _resolve(job_ids_file)

    with open(jobs_path) as f:
        job_ids = {int(line.strip()) for line in f if line.strip()}

    df = pd.read_parquet(prog_path)
    df = df[df["job_id"].isin(job_ids)]
    if exclude_zero_lag and "lag_idx" in df.columns:
        df = df[df["lag_idx"] != 0]

    lt = df["livetime"].sum()
    logger.info("Live time: %.0f s = %.2f days", lt, lt / 86400.0)

    return {
        "livetime": float(lt),
        "livetime_days": float(lt / 86400.0),
    }
