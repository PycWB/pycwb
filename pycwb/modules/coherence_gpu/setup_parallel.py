"""Bounded CPU preparation for the GPU processor, retaining native arithmetic.

Runs in the job (parent) process once per trial. The per-resolution body is
the unchanged native ``_setup_coherence_single_res``; this module only changes
how the resolutions are scheduled and, for the exploratory max-energy
switches, which ``max_energy`` implementation that body sees through a
private :func:`~pycwb.utils.function_binding.specialize` clone.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import jax
import numpy as np

from pycwb.modules.coherence_native import setup as native
from pycwb.types.time_series import TimeSeries

from pycwb.constants.gpu_options import gpu_options

if TYPE_CHECKING:
    from pycwb.config import Config
    from pycwb.types.job import WaveSegment

logger = logging.getLogger(__name__)

MAX_SETUP_WORKERS = 8
"""Largest ``gpu.setup_workers``; one thread per resolution level of the reference configuration."""

MAX_PREFILTER_SETUP_WORKERS = 3
"""Largest worker count with ``gpu.wdm_prefilter=true``; each worker holds its own device packet buffers."""


def _validate_against_native(
    config: Config, strains: list[Any], setups: list[dict[str, Any]], **kwargs: Any
) -> None:
    """Require every map, threshold and cache array of ``setups`` to equal the native serial result bitwise.

    Explicitly a correctness run: it duplicates all preparation work and its
    runtime must not be quoted as the optimized setup benchmark.

    Raises
    ------
    AssertionError
        If any energy map, TF map, selection-cache key set, dtype, shape or
        byte differs from ``native.setup_coherence``.
    """
    expected = native.setup_coherence(config, strains, **kwargs)
    for a, b in zip(expected, setups, strict=True):
        np.testing.assert_array_equal(
            np.asarray(a["Eo"]).view("u8"), np.asarray(b["Eo"]).view("u8")
        )
        for left, right in zip(a["tf_maps"], b["tf_maps"], strict=True):
            np.testing.assert_array_equal(
                np.asarray(left.data).view("u8"), np.asarray(right.data).view("u8")
            )
        ac, bc = a["selection_cache"], b["selection_cache"]
        if ac.keys() != bc.keys():
            raise AssertionError(
                "Parallel setup selection cache keys differ from native serial"
            )
        for key in ac:
            if isinstance(ac[key], np.ndarray):
                if not (
                    ac[key].dtype == bc[key].dtype and ac[key].shape == bc[key].shape
                ):
                    raise AssertionError(
                        f"Parallel setup selection cache {key!r} dtype or shape differs from native"
                    )
                np.testing.assert_array_equal(
                    np.ascontiguousarray(ac[key]).view("u1"),
                    np.ascontiguousarray(bc[key]).view("u1"),
                )
    logger.info(
        "GPU job parallel setup validation: %d resolutions, all maps, thresholds and cache arrays bitwise exact",
        len(setups),
    )


def setup_coherence(
    config: Config,
    strains: list[Any],
    job_seg: WaveSegment | None = None,
    nRMS: list[Any] | None = None,
) -> list[dict[str, Any]]:
    """Prepare the per-resolution coherence setup with bounded CPU threads.

    Drop-in for the native ``setup_coherence``. Switches, read at call time:

    * ``gpu.setup_workers`` (default 1, at most ``MAX_SETUP_WORKERS``):
      threads over resolution levels. With 1 and no exploration switch the
      native function is called unchanged.
    * ``gpu.wdm_prefilter=true``: exact CUDA prefilter with a native CPU FFT;
      one projection per worker, at most three workers.
    * ``gpu.validate_setup=true``: compare against the native serial result
      bitwise after the parallel preparation.

    Parameters
    ----------
    config : Config
        Search configuration; ``nRES`` and ``rateANA`` are used here.
    strains : list
        Conditioned strains, one per detector, accepted by
        ``TimeSeries.from_input``.
    job_seg : WaveSegment or None, optional
        Forwarded to the native per-resolution body.
    nRMS : list or None, optional
        Whitening-noise maps, one per detector, stored on every resolution.

    Returns
    -------
    list of dict
        One native setup dictionary per resolution level.

    Raises
    ------
    ValueError
        If a switch value is out of range or the switch combination is not
        supported, or if ``nRMS`` does not match the detector count.
    AssertionError
        If ``gpu.validate_setup=true`` and the result differs from native.
    """
    workers = gpu_options(config).setup_workers
    use_prefilter = gpu_options(config).wdm_prefilter
    single = native._setup_coherence_single_res
    local = threading.local()
    if use_prefilter and workers > MAX_PREFILTER_SETUP_WORKERS:
        raise ValueError(
            "GPU prefilter exploration supports at most three setup workers"
        )
    if workers == 1 and not use_prefilter:
        return native.setup_coherence(config, strains, job_seg=job_seg, nRMS=nRMS)
    if nRMS is not None and len(nRMS) != len(strains):
        raise ValueError("One whitening-noise map is required per detector")
    normalized = [TimeSeries.from_input(strain) for strain in strains]
    up_n = max(1, int(config.rateANA / 1024))
    device = jax.devices("cpu")[0]

    def prepare(i: int) -> dict[str, Any]:
        # JAX default-device scopes are thread-local, so enter one in each worker.
        with jax.default_device(device):
            implementation = single
            if use_prefilter:
                # Packet accumulation and its device buffers belong to one
                # worker; never share mutable maxima across resolutions.
                if not hasattr(local, "single"):
                    from pycwb.utils.function_binding import specialize
                    from pycwb.modules.coherence_gpu.max_energy_hybrid import make_projection

                    local.single = specialize(
                        single, max_energy=make_projection(cuda_prefilter=True)
                    )
                implementation = local.single
            return implementation(i, config, normalized, up_n, job_seg=job_seg)

    if workers == 1:
        setups = [prepare(i) for i in range(config.nRES)]
    else:
        with ThreadPoolExecutor(max_workers=min(workers, config.nRES)) as pool:
            setups = list(pool.map(prepare, range(config.nRES)))
    if nRMS is not None:
        for result in setups:
            result["nRMS"] = nRMS
    if gpu_options(config).validate_setup:
        _validate_against_native(config, strains, setups, job_seg=job_seg, nRMS=nRMS)
    return setups
