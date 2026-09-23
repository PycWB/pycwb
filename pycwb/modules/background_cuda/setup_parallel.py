"""Bounded CPU preparation for the GPU processor, retaining native arithmetic.

Runs in the job (parent) process once per trial. The per-resolution body is
the unchanged native ``_setup_coherence_single_res``; this module only changes
how the resolutions are scheduled and, for the exploratory max-energy
switches, which ``max_energy`` implementation that body sees through a
private :func:`~.binding.specialize` clone.
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

from . import flags

if TYPE_CHECKING:
    from pycwb.config import Config
    from pycwb.types.job import WaveSegment

logger = logging.getLogger(__name__)

MAX_SETUP_WORKERS = 8
"""Largest ``PYCWB_GPU_SETUP_WORKERS``; one thread per resolution level of the reference configuration."""

MAX_PREFILTER_SETUP_WORKERS = 3
"""Largest worker count with ``PYCWB_GPU_WDM_PREFILTER=1``; each worker holds its own device packet buffers."""


def _validate_against_native(config: Config, strains: list[Any], setups: list[dict[str, Any]], **kwargs: Any) -> None:
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
        np.testing.assert_array_equal(np.asarray(a["Eo"]).view("u8"), np.asarray(b["Eo"]).view("u8"))
        for left, right in zip(a["tf_maps"], b["tf_maps"], strict=True):
            np.testing.assert_array_equal(np.asarray(left.data).view("u8"), np.asarray(right.data).view("u8"))
        ac, bc = a["selection_cache"], b["selection_cache"]
        if ac.keys() != bc.keys():
            raise AssertionError("Parallel setup selection cache keys differ from native serial")
        for key in ac:
            if isinstance(ac[key], np.ndarray):
                if not (ac[key].dtype == bc[key].dtype and ac[key].shape == bc[key].shape):
                    raise AssertionError(f"Parallel setup selection cache {key!r} dtype or shape differs from native")
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

    * ``PYCWB_GPU_SETUP_WORKERS`` (default 1, at most ``MAX_SETUP_WORKERS``):
      threads over resolution levels. With 1 and no exploration switch the
      native function is called unchanged.
    * ``PYCWB_GPU_MAX_ENERGY=1``: exploratory GPU FFT max-energy core; requires
      one worker, forbids ``PYCWB_GPU_VALIDATE_SETUP``.
    * ``PYCWB_GPU_WDM_PREFILTER=1``: exploratory exact CUDA prefilter core with
      one projection per worker thread; at most
      ``MAX_PREFILTER_SETUP_WORKERS`` workers. Mutually exclusive with
      ``PYCWB_GPU_MAX_ENERGY``.
    * ``PYCWB_GPU_VALIDATE_MAX_ENERGY_SELECTION=1`` (with the GPU core): keep
      a native CPU setup under ``"_cpu_reference_setup"`` in each resolution
      for the paired coherence comparison; requires
      ``PYCWB_GPU_VALIDATE_STAGES=1``.
    * ``PYCWB_GPU_VALIDATE_SETUP=1``: compare against the native serial result
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
        If ``PYCWB_GPU_VALIDATE_SETUP=1`` and the result differs from native.
    """
    workers = flags.worker_count("SETUP_WORKERS", maximum=MAX_SETUP_WORKERS)
    use_gpu_core = flags.enabled("MAX_ENERGY")
    use_prefilter = flags.enabled("WDM_PREFILTER")
    if use_gpu_core and use_prefilter:
        raise ValueError("Choose either GPU FFT exploration or exact prefilter exploration")
    if use_gpu_core:
        if flags.enabled("VALIDATE_SETUP"):
            raise ValueError("Strict serial-map parity is not established for the exploratory GPU FFT core")
        if workers != 1:
            raise ValueError("GPU max-energy exploration requires one setup worker")
        from .binding import specialize
        from .max_energy_jax import make_projection

        single = specialize(native._setup_coherence_single_res, max_energy=make_projection())
        setup = specialize(native.setup_coherence, _setup_coherence_single_res=single)
        result = setup(config, strains, job_seg=job_seg, nRMS=nRMS)
        if flags.enabled("VALIDATE_MAX_ENERGY_SELECTION"):
            if not flags.enabled("VALIDATE_STAGES"):
                raise ValueError("GPU max-energy selection validation requires paired stages")
            expected = native.setup_coherence(config, strains, job_seg=job_seg, nRMS=nRMS)
            for actual, cpu in zip(result, expected, strict=True):
                actual["_cpu_reference_setup"] = cpu
            logger.info("GPU max-energy validation: separate native CPU setup retained for coherence comparison")
        return result
    single = native._setup_coherence_single_res
    local = threading.local()
    if use_prefilter and workers > MAX_PREFILTER_SETUP_WORKERS:
        raise ValueError("GPU prefilter exploration supports at most three setup workers")
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
                    from .binding import specialize
                    from .max_energy_hybrid import make_projection

                    local.single = specialize(single, max_energy=make_projection(cuda_prefilter=True))
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
    if flags.enabled("VALIDATE_SETUP"):
        _validate_against_native(config, strains, setups, job_seg=job_seg, nRMS=nRMS)
    return setups
