"""
pycwb.modules.read_data — Gravitational-wave strain data reader.

Reads GW strain from frame files (GWF) or online data servers, constructs
simulated job-segment data, writes GWF output, and checks data quality.
"""

from pycwb.modules.injection.strain import (
    generate_strain_from_injection,
    project_to_detector,
)

from .data_check import check_and_resample, check_and_resample_py, data_check
from .read_data import (
    merge_frames,
    read_from_catalog,
    read_from_gwf,
    read_from_job_segment,
    read_from_online,
    read_single_frame_from_job_segment,
)
from .simulations import (
    generate_injection,
    generate_injections,
    generate_noise_for_job_seg,
)
from .write_data import save_to_gwf

__all__ = [
    "read_from_gwf",
    "read_from_online",
    "read_from_catalog",
    "read_from_job_segment",
    "merge_frames",
    "read_single_frame_from_job_segment",
    "data_check",
    "check_and_resample",
    "check_and_resample_py",
    "generate_noise_for_job_seg",
    "generate_injections",
    "generate_injection",
    "generate_strain_from_injection",
    "project_to_detector",
    "save_to_gwf",
]
