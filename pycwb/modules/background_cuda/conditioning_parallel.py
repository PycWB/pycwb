"""Legacy config-driven adapter; new callers pass explicit scheduling options."""

from __future__ import annotations

from typing import TYPE_CHECKING
from collections.abc import Sequence

if TYPE_CHECKING:
    from pycwb.config import Config
    from pycwb.types.time_series import TimeSeries
    from pycwb.types.noise_rms import NoiseRMSMap

from pycwb.constants.gpu_options import gpu_options
from pycwb.modules.data_conditioning.parallel import condition_strains as _run


def condition_strains(
    config: Config, strains: list[TimeSeries]
) -> tuple[Sequence[TimeSeries], Sequence[NoiseRMSMap]]:
    """Adapt the historical GPU YAML settings to the CPU scheduling API."""
    options = gpu_options(config)
    return _run(
        config, strains, workers=options.condition_workers, validate=options.validate_conditioning
    )
