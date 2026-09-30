"""Migration diagnostics for retired processing environment switches."""

from dataclasses import fields
import logging
import os

from pycwb.config.processing import ExecutionProfile
from pycwb.constants.gpu_options import GPUOptions

logger = logging.getLogger(__name__)

_REPLACEMENTS = {
    (field.name.upper() if field.name.startswith("wdm_") else f"PYCWB_{field.name.upper()}"):
        f"execution_profile.{field.name}"
    for field in fields(ExecutionProfile)
}
_REPLACEMENTS.update({
    f"PYCWB_GPU_{field.name.upper()}": f"gpu.{field.name}"
    for field in fields(GPUOptions)
})
# Released switches whose replacements are top-level YAML keys. v1.1.0a3 read
# these and PYCWB_REGRESSION_ENGINE, PYCWB_REGRESSION_PERCENTILE_STRIDE and
# PYCWB_NUMBA_MAX_ENERGY_MODE; the remaining names were development-only.
_REPLACEMENTS.update({
    "PYCWB_MAX_ENERGY_BACKEND": "max_energy_backend",
    "PYCWB_COHERENCE_TIMING": "coherence_timing",
})


def warn_legacy_environment() -> None:
    """Log ignored switches and their YAML replacements without applying values.

    Called when a config is loaded, never in numerical loops. Only known
    retired settings are inspected; installation paths, library thread counts,
    and active test/documentation environment controls remain valid.
    """
    retired = [
        f"{name} (use YAML {replacement})"
        for name, replacement in sorted(_REPLACEMENTS.items())
        if name in os.environ
    ]
    if retired:
        logger.warning(
            "Ignoring retired execution environment variables: %s. "
            "Only explicit configuration controls processing; remove these variables.",
            "; ".join(retired),
        )
