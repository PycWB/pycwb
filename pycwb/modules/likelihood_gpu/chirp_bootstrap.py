"""Experimental CUDA scoring of native seeded micropixel chirp trials.

The native bootstrap samples :data:`chirp_bootstrap_plan.TRIALS` trials
sequentially from one uniform stream and scores each against every
micropixel. Sampling stays on the CPU (:func:`chirp_bootstrap_plan.prepare_bootstrap`);
the independent scoring runs in ``chirp_bootstrap.cu`` as a per-pixel
classification followed by a per-trial ordered reduction, so each score keeps
the native ordered sum across micropixels.
"""

from __future__ import annotations

import ctypes as ct
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from numba import cuda, njit

from pycwb.modules.likelihood_gpu.chirp_bootstrap_plan import TRIALS, finish_bootstrap, prepare_bootstrap
from pycwb.modules.gpu_utils.cuda_runtime import load_module

MASK_BUDGET_BYTES = 256 * 1024**2
"""Largest ``(n_micropixels, trials)`` uint8 classification buffer per launch."""

SCORE_COLUMNS = 3
"""Per-trial ``bootstrap_reduce`` outputs: ``score, selected_count, symmetry``."""


@njit(cache=True)
def frequency_powers(f: np.ndarray) -> np.ndarray:
    """Precompute ``(freq / 128) ** (-8/3)`` for the three frequency offsets used by the kernel.

    Parameters
    ----------
    f : numpy.ndarray
        Float64 micropixel frequencies.

    Returns
    -------
    numpy.ndarray
        Float64 ``(n, 3)`` array for offsets ``-4, +4, 0`` Hz in that column
        order; zero where the shifted frequency is not positive. Computed with
        CPU ``**`` so no CUDA ``pow`` result enters the comparison.
    """
    out = np.zeros((len(f), 3))
    for i in range(len(f)):
        for j, offset in enumerate((-4.0, 4.0, 0.0)):
            freq = f[i] + offset
            if freq > 0:
                out[i, j] = (freq / 128.0) ** (-8.0 / 3.0)
    return out


class ChirpBootstrap:
    """CUDA replacement for ``likelihoodWP.chirp_micropixel._bootstrap``.

    Passed as the explicit ``bootstrap`` callback to ``estimate_chirp`` by
    :func:`make_chirp_update`.

    Attributes
    ----------
    module : CUDAModule
        Process-wide compiled ``chirp_bootstrap.cu``.
    """

    def __init__(self) -> None:
        self.module = load_module(Path(__file__).with_suffix(".cu"))

    def __call__(
        self,
        x: np.ndarray,
        f: np.ndarray,
        energy: np.ndarray,
        mindt: float,
        uniforms: np.ndarray,
    ) -> tuple[np.ndarray, int]:
        """Run the seeded bootstrap and return the native ``(result, cursor)`` pair.

        Parameters
        ----------
        x, f, energy : array-like
            Micropixel times, frequencies and energies; converted to
            contiguous float64 vectors of equal length ``n >= 13``.
        mindt : float
            Minimum time resolution passed to :func:`finish_bootstrap`.
        uniforms : array-like
            Float64 uniform stream, consumed in native order.

        Returns
        -------
        tuple[numpy.ndarray, int]
            ``([mass, merger, ellipticity, energy_fraction, symmetry], cursor)``.
            Five NaN when the stream ran out before all trials were sampled;
            five zeros when the best trial selected fewer than four pixels.

        Raises
        ------
        ValueError
            If the inputs are not equal-length vectors with at least 13 pixels.
        MemoryError
            If one trial's classification masks alone exceed
            :data:`MASK_BUDGET_BYTES`.
        """
        x, f, energy, uniforms = (
            np.ascontiguousarray(a, dtype=np.float64) for a in (x, f, energy, uniforms)
        )
        if x.ndim != 1 or f.shape != x.shape or energy.shape != x.shape or len(x) < 13:
            raise ValueError("Bootstrap requires at least 13 matching micropixels")
        _, weights, slopes, mergers, valid, ellipticity, cursor, ready = (
            prepare_bootstrap(x, f, energy, mindt, uniforms)
        )
        if not ready:
            return np.full(5, np.nan), cursor
        (
            device_x,
            device_f,
            device_weights,
            device_slopes,
            device_mergers,
            device_valid,
            device_powers,
        ) = (
            cuda.to_device(a)
            for a in (x, f, weights, slopes, mergers, valid, frequency_powers(f))
        )
        out = cuda.device_array((TRIALS, SCORE_COLUMNS), np.float64)
        # Trials are independent. Batch trials, not pixels, so each score
        # retains exactly the native ordered sum across all micropixels.
        batch = min(TRIALS, MASK_BUDGET_BYTES // len(x))
        if batch < 1:
            raise MemoryError("One bootstrap trial exceeds 256 MiB budget")
        masks = cuda.device_array((len(x), batch), np.uint8)
        for start in range(0, TRIALS, batch):
            count = min(batch, TRIALS - start)
            dimensions = [ct.c_int(v) for v in (len(x), start, count)]
            # Order matches bootstrap_classify(x, f, weights, slopes, mergers,
            # valid, frequency_powers, n, trial_start, trial_count, out).
            classify_args = [
                device_x,
                device_f,
                device_weights,
                device_slopes,
                device_mergers,
                device_valid,
                device_powers,
                *dimensions,
                masks,
            ]
            self.module.launch("bootstrap_classify", len(x) * count, classify_args)
            # Order matches bootstrap_reduce(masks, weights, valid, n,
            # trial_start, trial_count, out).
            reduce_args = [masks, device_weights, device_valid, *dimensions, out]
            self.module.launch("bootstrap_reduce", count, reduce_args)
        scores = out.copy_to_host()
        best = 0.0
        selected = 0
        slope = merger = symmetry = 0.0
        # Native forward loop with ">=": the last trial reaching the maximum wins.
        for i in range(TRIALS):
            if valid[i] and scores[i, 0] >= best:
                best = scores[i, 0]
                slope, merger = slopes[i], mergers[i]
                selected = int(scores[i, 1])
                symmetry = scores[i, 2]
        return finish_bootstrap(
            x, f, weights, mindt, slope, merger, selected, ellipticity, symmetry, cursor
        )


def make_chirp_update() -> Callable[..., None]:
    """Configure native chirp selection while retaining native resets/fallbacks.

    Returns
    -------
    callable
        A replacement for ``likelihoodWP.likelihood._update_cluster_chirp_statistics``
        with the same keyword-only signature. It evaluates the native
        enabling conditions itself: when they hold it runs ``estimate_chirp``
        with :class:`ChirpBootstrap` passed as ``bootstrap`` and writes the
        chirp fields of ``cluster.cluster_meta``; otherwise it defers to the
        native function so resets and fallbacks stay native.
    """
    import importlib

    from functools import partial

    native = importlib.import_module("pycwb.modules.likelihoodWP.likelihood")
    chirp = importlib.import_module("pycwb.modules.likelihoodWP.chirp_micropixel")
    estimate = partial(chirp.estimate_chirp, bootstrap=ChirpBootstrap())

    def update(
        cluster: Any,
        config: Any,
        *,
        xgb_rho_mode: bool,
        chirp_seed: int,
        use_native_chirp: bool,
    ) -> None:
        active = (
            use_native_chirp
            and xgb_rho_mode
            and not getattr(config, "optim", False)
            and getattr(config, "cfg_search", "") in tuple("iecrpblsg")
            and getattr(config, "Search", "") in ("CBC", "BBH", "IMBHB")
        )
        if not active:
            return native._update_cluster_chirp_statistics(
                cluster,
                config,
                xgb_rho_mode=xgb_rho_mode,
                chirp_seed=chirp_seed,
                use_native_chirp=use_native_chirp,
            )
        result = estimate(cluster.pixel_arrays, config.rateANA, chirp_seed)
        meta = cluster.cluster_meta
        meta.mchirp = result.mass
        meta.mchirp_error = result.mass_error
        meta.chirp_merger_time = result.merger_time
        meta.chirp_merger_time_error = result.merger_time_error
        meta.chirp_ellipticity = result.ellipticity
        meta.chirp_energy_fraction = result.energy_fraction
        meta.chirp_symmetry = result.symmetry
        return None

    return update
