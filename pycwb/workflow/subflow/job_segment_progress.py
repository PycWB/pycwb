"""Progress and catalog helpers for the native job-segment workflow."""

import os

import numpy as np

from pycwb.config import Config
from pycwb.types.job import WaveSegment


def _catalog_path(working_dir: str, config: Config, catalog_file: str | None) -> str | None:
    if not catalog_file:
        return None
    if os.path.isabs(catalog_file):
        return catalog_file
    return os.path.join(working_dir, config.catalog_dir, catalog_file)


def _record_lag_progress(
    working_dir: str,
    config: Config,
    catalog_file: str | None,
    queue,
    progress_record: dict,
) -> None:
    if queue is not None:
        queue.put({"type": "progress", **progress_record})
        return
    catalog_path = _catalog_path(working_dir, config, catalog_file)
    if catalog_path:
        from pycwb.modules.catalog.catalog import Catalog

        Catalog.open(catalog_path).add_lag_progress(**progress_record)


def _lag_metadata(sub_job_seg: WaveSegment, lag: int) -> tuple[list[float], list[float], np.ndarray]:
    lag_shifts = sub_job_seg.lag_shifts[lag]
    time_lag = [float(v) for v in lag_shifts]
    segment_lag = (
        [float(v) for v in sub_job_seg.shift]
        if sub_job_seg.shift is not None
        else [0.0 for _ in sub_job_seg.ifos]
    )
    return time_lag, segment_lag, lag_shifts


def _lag_progress_record(
    context,
    lag: int,
    n_triggers: int,
    livetime: float,
    status: str,
) -> dict:
    return dict(
        job_id=context.sub_job_seg.index,
        trial_idx=context.trial_idx,
        lag_idx=lag,
        n_triggers=n_triggers,
        livetime=livetime,
        status=status,
    )
