"""Experimental ordered full likelihood sky scan; numerical gate required.

One thread per selected sky direction evaluates the native ``sky_scratch``
arithmetic fused per pixel and ordered across pixels (``likelihood_scan.cu``).
The best-direction tie-breaking is done on the host with the same
"last maximum wins" rule as ``likelihoodWP.sky_scan.scan_sky_for_best_fit``.
"""

from __future__ import annotations

import ctypes as ct
from pathlib import Path
from typing import Any

import numpy as np

from . import flags
from .cuda_runtime import DeviceBuffers, load_module
from .geometry_cache import resident_geometry

MAP_COUNT = 11
"""Per-direction statistics returned as sky maps, in native tuple order."""

KERNEL_COLUMNS = MAP_COUNT + 1
"""Kernel output row width: the 11 maps plus the tie-breaking statistic."""


class LikelihoodScan:
    """CUDA replacement for the native likelihood sky scan.

    ``processor.py`` binds one instance to three native hooks of
    ``likelihoodWP.likelihood.likelihood`` when ``PYCWB_GPU_LIKELIHOOD=1``:
    ``_scan_sky_for_best_fit``, ``_scan_sky_grouped_delays`` and
    ``_scan_sky_scratch``. All three receive the same fourteen leading
    positional arguments; the grouped/scratch variants append
    ``group_order, group_offsets`` (see :meth:`__call__`).

    Attributes
    ----------
    module : CUDAModule
        Process-wide compiled ``likelihood_scan.cu``.
    geometry : dict
        Cache of device copies of ``FP``, ``FX`` (float32) and ``ml`` (int32)
        keyed by :func:`geometry_cache.geometry_key`; geometry is immutable for
        one trial and the entry retains the host arrays.
    workspace : Workspace or None
        Reusable device slots when ``PYCWB_GPU_REUSE_WORKSPACE=1`` was set at
        construction time.
    buffers : DeviceBuffers
        Upload/output/download helper wrapping ``workspace``.
    """

    def __init__(self) -> None:
        self.module = load_module(Path(__file__).with_suffix(".cu"))
        self.geometry: dict[tuple[int, ...], Any] = {}
        self.workspace: Any | None = None
        if flags.enabled("REUSE_WORKSPACE"):
            from .workspace import Workspace

            self.workspace = Workspace()
        self.buffers = DeviceBuffers(self.workspace)

    def __call__(
        self,
        n_ifo: int,
        n_pix: int,
        n_sky: int,
        FP: np.ndarray,
        FX: np.ndarray,
        rms: np.ndarray,
        td00: np.ndarray,
        td90: np.ndarray,
        ml: np.ndarray,
        REG: np.ndarray,
        netCC: float,
        delta_regulator: float,
        network_energy_threshold: float,
        sky_valid_indices: np.ndarray,
        *group_args: Any,
    ) -> tuple[Any, ...]:
        """Scan the selected sky directions and return the native statistics tuple.

        Parameters
        ----------
        n_ifo : int
            Detector count, 2 or 3.
        n_pix : int
            Cluster pixel count, at least 1.
        n_sky : int
            Sky-grid size (axis 0 of ``FP``/``FX``, axis 1 of ``ml``).
        FP, FX : numpy.ndarray
            Antenna patterns, shape ``(n_sky, n_ifo)``; resident as float32.
        rms : numpy.ndarray
            Pixel noise weights, shape ``(n_pix, n_ifo)``; uploaded as float32.
        td00, td90 : numpy.ndarray
            Time-delayed in-phase/quadrature amplitudes, shape
            ``(n_delay, n_ifo, n_pix)``; uploaded as float32 on every call.
        ml : numpy.ndarray
            Sky-delay indices, shape ``(n_ifo, n_sky)``, resident as int32.
            ``ml + n_delay // 2`` must index axis 0 of ``td00``.
        REG : numpy.ndarray
            Regularization parameters; the kernel reads ``REG[0]`` and
            ``REG[1]`` as float32.
        netCC : float
            Network correlation threshold; directions below it are rejected.
        delta_regulator : float
            Accepted for hook-signature compatibility only. The native scan
            uses it solely for a probability map that is not part of the
            returned tuple, so the kernel does not receive it.
        network_energy_threshold : float
            Pixel energy threshold, passed as float64.
        sky_valid_indices : array-like
            Sky directions to evaluate; converted to a contiguous int64 vector
            and must be non-empty.
        *group_args
            ``group_order, group_offsets`` appended by the
            ``_scan_sky_grouped_delays`` and ``_scan_sky_scratch`` hooks. They
            only describe a CPU delay-reuse schedule and are ignored here; the
            kernel gathers each direction's delays directly.

        Returns
        -------
        tuple
            ``(l_max, antenna_prior, alignment, likelihood, null_energy,
            coherent_energy, correlation, sky_stat, disbalance, network_index,
            ellipticity, polarisation, sky_stat_max)``: ``l_max`` is an ``int``,
            the eleven maps are float32 arrays of length ``n_sky`` (zero
            outside ``sky_valid_indices``) and ``sky_stat_max`` is a ``float``.
            Ties keep the last maximum in ``sky_valid_indices`` order; when no
            direction has a non-negative statistic the result is
            ``(int(sky_valid_indices[0]), ..., 0.0)``.

        Raises
        ------
        ValueError
            If ``n_ifo`` is not 2 or 3, ``n_pix`` is 0, the sky mask is empty
            or out of range, the array shapes disagree, or a delay index falls
            outside ``td00``.
        """
        if n_ifo not in (2, 3) or n_pix < 1:
            raise ValueError("Requires nonempty pixels and 2/3 detectors")
        skies = np.ascontiguousarray(sky_valid_indices, dtype=np.int64)
        if skies.ndim != 1 or len(skies) == 0 or np.any(skies < 0) or np.any(skies >= n_sky):
            raise ValueError("Invalid sky mask")
        if (
            FP.shape != (n_sky, n_ifo)
            or FX.shape != FP.shape
            or rms.shape != (n_pix, n_ifo)
            or td00.shape != td90.shape
            or td00.ndim != 3
            or td00.shape[1:] != (n_ifo, n_pix)
            or ml.shape != (n_ifo, n_sky)
        ):
            raise ValueError("Invalid scan shapes")
        offset = td00.shape[0] // 2
        if np.any(ml + offset < 0) or np.any(ml + offset >= td00.shape[0]):
            raise ValueError("Invalid delays")
        device_fp, device_fx, delays = resident_geometry(
            self.geometry, (FP, FX, ml), (np.float32, np.float32, np.int32)
        )
        weights = self.buffers.upload("rms", rms, np.float32)
        device_td00 = self.buffers.upload("td00", td00, np.float32)
        device_td90 = self.buffers.upload("td90", td90, np.float32)
        indices = self.buffers.upload("skies", skies, np.int64)
        reg = self.buffers.upload("reg", REG, np.float32)
        out = self.buffers.output("output", (len(skies), KERNEL_COLUMNS), np.float32)
        # Order matches likelihood_scan(FP, FX, rms, td0, td9, ml, skies, nvalid,
        # nsky, npix, nd, offset, reg, threshold, netCC, maps).
        kernel_args = [
            device_fp,
            device_fx,
            weights,
            device_td00,
            device_td90,
            delays,
            indices,
            ct.c_int(len(skies)),
            ct.c_int(n_sky),
            ct.c_int(n_pix),
            ct.c_int(n_ifo),
            ct.c_int(offset),
            reg,
            ct.c_double(network_energy_threshold),
            ct.c_double(netCC),
            out,
        ]
        self.module.launch("likelihood_scan", len(skies), kernel_args)
        values = self.buffers.download(out, (len(skies), KERNEL_COLUMNS))
        maps = np.zeros((MAP_COUNT, n_sky), np.float32)
        maps[:, skies] = values[:, :MAP_COUNT].T
        eligible = np.flatnonzero(values[:, MAP_COUNT] >= 0.0)
        best, stat = int(skies[0]), 0.0
        if len(eligible):
            # argmax returns the first maximum; reversing makes the last one win.
            reverse = eligible[::-1]
            winner = reverse[np.argmax(values[reverse, MAP_COUNT])]
            best, stat = int(skies[winner]), float(values[winner, MAP_COUNT])
        return (best, *maps, stat)
