"""Legacy config-driven adapter; new callers pass explicit scheduling options."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pycwb.config import Config
    from pycwb.types.job import WaveSegment
    from pycwb.types.time_series import TimeSeries

from pycwb.constants.gpu_options import gpu_options
from pycwb.modules.read_data.parallel import read_from_job_segment as _run


def read_from_job_segment(config: Config, job_seg: WaveSegment) -> list[TimeSeries]:
    """Adapt the historical GPU YAML settings to the CPU scheduling API."""
    options = gpu_options(config)
    return _run(
        config,
        job_seg,
        workers=options.read_workers,
        processes=options.read_processes,
        validate=options.validate_read,
    )
