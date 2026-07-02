"""Max-energy projection and backend selection for native coherence."""

from __future__ import annotations

import os

from pycwb.config import Config
from pycwb.types.time_frequency_map import TimeFrequencyMap

from .time_delay_max_energy import time_delay_max_energy, time_delay_max_energy_numba


def _normalize_max_energy_backend(backend: str | None) -> str:
    """Validate and normalize a max-energy backend name."""
    backend = str(backend or "jax").strip().lower()
    aliases = {
        "auto": "auto",
        "hybrid": "auto",
        "xla": "jax",
        "jax": "jax",
        "numba": "numba",
        "nb": "numba",
    }
    if backend not in aliases:
        raise ValueError(
            "max_energy_backend must be one of {'jax', 'numba', 'auto'} "
            f"(got {backend!r})"
        )
    return aliases[backend]


def _auto_max_energy_backend_for_layers(layers: int) -> str:
    """Choose the fastest observed backend for a WDM layer count."""
    return "numba" if 32 <= int(layers) <= 256 else "jax"


def _max_energy_backend(config: Config | None = None, layers: int | None = None) -> str:
    """Resolve the configured max-energy backend."""
    backend = os.getenv("PYCWB_MAX_ENERGY_BACKEND")
    if backend is None and config is not None:
        backend = getattr(config, "max_energy_backend", None)
    backend = _normalize_max_energy_backend(backend)
    if backend == "auto" and layers is not None:
        return _auto_max_energy_backend_for_layers(layers)
    return backend


def _max_energy_backend_label(backend: str) -> str:
    """Human-readable backend label for timing logs."""
    if backend == "numba":
        mode = os.getenv("PYCWB_NUMBA_MAX_ENERGY_MODE")
        if mode:
            return f"{backend}:{mode}"
    return backend


def max_energy(
    tf_map: TimeFrequencyMap,
    max_delay: float,
    up_n: int,
    pattern: int,
    f_low: float | None = None,
    f_high: float | None = None,
    hist: list | None = None,
    backend: str = "jax",
) -> tuple[TimeFrequencyMap, float]:
    """
    Compute max-energy skymap projection for a detector TF map.

    Calls :func:`time_delay_max_energy` from the module-level pure-function
    implementation and returns a new TF map together with the Gamma-to-Gauss
    scaling parameter.

    Parameters
    ----------
    tf_map : TimeFrequencyMap
        Detector time-frequency map object.
    max_delay : float
        Maximum delay for the time series.
    up_n : int
        Upsample factor for decorrelation.
    pattern : int
        Wave packet pattern identifier.
    f_low : float | None, optional
        Low-frequency cutoff in Hz.
    f_high : float | None, optional
        High-frequency cutoff in Hz.
    hist : list | None, optional
        Optional histogram container for statistics.
    backend : {"jax", "numba", "auto"}, optional
        Backend for the time-delay max-energy kernel. ``"jax"`` preserves the
        historical behavior. ``"numba"`` uses the compiled numba pattern path
        and delegates pattern 0 to JAX. ``"auto"`` chooses from the WDM layer
        count.

    Returns
    -------
    tuple[TimeFrequencyMap, float]
        Updated TF map after max-energy projection and Gamma-to-Gauss scaling
        factor.
    """
    if hasattr(tf_map, "bandpass"):
        tf_map.bandpass(f_low=f_low, f_high=f_high)

    backend = _normalize_max_energy_backend(backend)
    if backend == "auto":
        backend = _auto_max_energy_backend_for_layers(int(tf_map.wavelet.M))
    if backend == "numba":
        new_tf_map, result = time_delay_max_energy_numba(
            tf_map,
            max_delay,
            downsample=up_n,
            pattern=pattern,
            hist=hist,
        )
    else:
        new_tf_map, result = time_delay_max_energy(
            tf_map,
            max_delay,
            downsample=up_n,
            pattern=pattern,
            hist=hist,
        )
    return new_tf_map, result
