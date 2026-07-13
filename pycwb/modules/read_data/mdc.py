"""Deprecated compatibility imports for the former MDC module."""

import warnings

warnings.warn(
    "pycwb.modules.read_data.mdc is deprecated and will be removed after one "
    "release; use "
    "pycwb.modules.read_data.simulations, pycwb.modules.injection.strain, "
    "and pycwb.modules.read_data.write_data instead",
    DeprecationWarning,
    stacklevel=2,
)

from pycwb.modules.injection.strain import (  # noqa: E402
    generate_strain_from_injection,
    project_to_detector,
)

from .simulations import (  # noqa: E402
    generate_injection,
    generate_injections,
    generate_noise_for_job_seg,
)
from .write_data import save_to_gwf  # noqa: E402

__all__ = [
    "generate_noise_for_job_seg",
    "project_to_detector",
    "save_to_gwf",
    "generate_strain_from_injection",
    "generate_injections",
    "generate_injection",
]
