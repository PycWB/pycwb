"""Pure semantic checks shared by offline validation and runtime restoration."""

from typing import Any

from pycwb.constants.execution import ExecutionSettings
from pycwb.constants.execution_profile import resolve_execution_profile
from pycwb.constants.gpu_options import gpu_options
from pycwb.utils.skymap_coord import validate_user_sky_config

RECONSTRUCTION_PRODUCT_FLAGS = (
    "save_waveform",
    "plot_waveform",
    "plot_trigger",
    "plot_sky_map",
)
OUTPUT_PRODUCT_FLAGS = RECONSTRUCTION_PRODUCT_FLAGS + ("save_cluster", "save_sky_map")


def validate_runtime_settings(config: Any) -> None:
    """Reject incompatible settings without loading catalogs, data or devices.

    Segment injections, CUDA availability, wavelet sizes and output paths require
    runtime checks as well. A YAML injection specification alone does not imply
    that every segment contains an injection.
    """
    ExecutionSettings.from_config(config)
    profile = resolve_execution_profile(getattr(config, "execution_profile", None))
    options = gpu_options(config)
    if options.dpf and not profile.scalar_dpf:
        raise ValueError("gpu.dpf requires execution_profile.scalar_dpf=true")
    if options.output_batch > 1 and getattr(config, "save_waveform", False):
        raise ValueError("GPU catalog batching requires background without saved waveforms")
    if options.q_reconstruction and any(
        getattr(config, flag, False) for flag in OUTPUT_PRODUCT_FLAGS
    ):
        raise ValueError(
            "Q-veto reconstruction requires background without saved waveforms or plots"
        )
    validate_user_sky_config(
        getattr(config, "sky_mask", None), context="sky_mask", default_coordsys="geo"
    )
    validate_user_sky_config(
        (getattr(config, "injection", None) or {}).get("sky_distribution"),
        context="injection.sky_distribution",
        default_coordsys="icrs",
    )
