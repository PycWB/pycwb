"""Ordered CUDA subnet scan, opt-in through ``gpu.subnet=true``.

``subnet_scan.cu`` evaluates the native
``super_cluster_native.sub_net_cut.optimze_sky_loc_from_td`` direction loop
with one thread per sky direction, keeping the CPU FP32 reduction order and
the FP64 accumulation of the BLAS ``sdot`` products. :meth:`SubnetScan.scan_many`
batches many clusters that share the same geometry.
"""

from __future__ import annotations

import ctypes as ct
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numba import cuda

from .cuda_runtime import load_module
from .geometry_cache import resident_geometry

SCORE_COLUMNS = 6
"""Per-direction kernel outputs: ``AA, Eo, subnet, m, suball, EE`` as float64."""

BEST_COLUMNS = 7
"""``subnet_best`` output row: winning direction index followed by its six scores."""

SCORE_BUDGET_BYTES = 256 * 1024**2
"""Largest per-batch ``(clusters, n_sky, 6)`` float64 score buffer in :meth:`SubnetScan.scan_many`."""

SUBNET_BEST_THREADS = 128
"""Block size of the ``subnet_best`` reduction kernel.

``subnet_scan.cu`` declares ``__shared__ double bests[128]`` and
``__shared__ int indices[128]`` and starts its tree reduction at ``step=64``,
so the kernel must be launched with exactly this many threads per block and
one block per cluster.
"""


class SubnetScan:
    """CUDA counterpart to ``optimze_sky_loc_from_td``.

    ``processor.py`` binds an instance as ``optimze_sky_loc_from_td`` inside a
    specialized ``_sub_net_cut_prepared_packets`` when ``gpu.subnet=true``;
    :class:`subnet_batch.BatchedSubnet` uses :meth:`scan_many`.

    Attributes
    ----------
    module : CUDAModule
        Process-wide compiled ``subnet_scan.cu``.
    geometry : dict
        Cache of device copies of ``FP``, ``FX`` (float32) and ``ml`` (int32)
        keyed by :func:`geometry_cache.geometry_key`. Geometry is immutable
        for one trial; the entry retains the host arrays.
    """

    def __init__(self) -> None:
        self.module = load_module(Path(__file__).with_suffix(".cu"))
        self.geometry: dict[tuple[int, ...], Any] = {}

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
        network_energy_threshold: float,
        e2or: float,
        subcut: float,
    ) -> tuple[Any, ...]:
        """Scan every sky direction for one cluster.

        Parameters
        ----------
        n_ifo : int
            Detector count, 2 or 3.
        n_pix : int
            Cluster pixel count, at least 1.
        n_sky : int
            Sky-grid size, at least 1.
        FP, FX : numpy.ndarray
            Antenna patterns, shape ``(n_sky, n_ifo)``; resident as float32.
        rms : numpy.ndarray
            Pixel noise weights, shape ``(n_pix, n_ifo)``; uploaded as float32.
        td00, td90 : numpy.ndarray
            Time-delayed amplitudes, shape ``(n_delay, n_ifo, n_pix)``; uploaded
            as float32 on every call.
        ml : numpy.ndarray
            Sky-delay indices, shape ``(n_ifo, n_sky)``; ``ml + n_delay // 2``
            must index axis 0 of ``td00``. Resident as int32.
        network_energy_threshold : float
            Pixel energy threshold, passed as float32.
        e2or : float
            Sub-network energy threshold; the kernel receives ``2 * e2or`` as
            float32, matching the native ``Es``.
        subcut : float
            Sub-network cut; negative disables it. Passed as float64.

        Returns
        -------
        tuple
            ``(l_max, stat, Em, Am, lm, Vm, suball, EE)`` as in the native
            function, with ``l_max == lm`` the winning direction index, ``Vm``
            an ``int`` and the rest ``numpy.float64``. The first direction with
            the strictly positive maximum ``stat`` wins; all-zero scores return
            ``(0, 0.0, 0.0, 0.0, 0, 0, 0.0, 0.0)``.

        Raises
        ------
        ValueError
            If the dimensions or array shapes are invalid or a delay index
            falls outside ``td00``.
        """
        if n_ifo not in (2, 3) or n_sky < 1 or n_pix < 1:
            raise ValueError("Expected positive sky/pixel counts and 2/3 detectors")
        shapes = [
            (n_sky, n_ifo),
            (n_sky, n_ifo),
            (n_pix, n_ifo),
            td00.shape,
            td00.shape,
            (n_ifo, n_sky),
        ]
        values = [FP, FX, rms, td00, td90, ml]
        if (
            td00.ndim != 3
            or td00.shape[1:] != (n_ifo, n_pix)
            or any(
                np.shape(x) != shape for x, shape in zip(values, shapes, strict=True)
            )
        ):
            raise ValueError("Invalid subnet array shapes")
        offset = td00.shape[0] // 2
        if np.any(ml + offset < 0) or np.any(ml + offset >= td00.shape[0]):
            raise ValueError("Delay out of bounds")
        device_fp, device_fx, delays = resident_geometry(
            self.geometry, (FP, FX, ml), (np.float32, np.float32, np.int32)
        )
        device_rms, device_td00, device_td90 = (
            cuda.to_device(np.ascontiguousarray(x, dtype=np.float32))
            for x in (rms, td00, td90)
        )
        out = cuda.device_array((n_sky, SCORE_COLUMNS), np.float64)
        # Order matches subnet_scan(FP, FX, rms, td0, td9, ml, nsky, npix, nd,
        # offset, threshold, Es, subcut, scores).
        kernel_args = [
            device_fp,
            device_fx,
            device_rms,
            device_td00,
            device_td90,
            delays,
            ct.c_int(n_sky),
            ct.c_int(n_pix),
            ct.c_int(n_ifo),
            ct.c_int(offset),
            ct.c_float(network_energy_threshold),
            ct.c_float(2 * e2or),
            ct.c_double(subcut),
            out,
        ]
        self.module.launch("subnet_scan", n_sky, kernel_args)
        scores = out.copy_to_host()
        eligible = np.flatnonzero(scores[:, 0] > 0.0)
        if not len(eligible):
            return (0, 0.0, 0.0, 0.0, 0, 0, 0.0, 0.0)
        best = int(eligible[np.argmax(scores[eligible, 0])])
        a = scores[best]
        return best, a[0], a[1], a[2], best, int(a[3]), a[4], a[5]

    def scan_many(
        self,
        inputs: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
        n_ifo: int,
        n_sky: int,
        FP: np.ndarray,
        FX: np.ndarray,
        ml: np.ndarray,
        threshold: float,
        e2or: float,
        subcut: float,
    ) -> list[tuple[Any, ...]]:
        """Batch nonempty ``(rms, td00, td90)`` packets sharing immutable geometry.

        Scores have a :data:`SCORE_BUDGET_BYTES` batch limit. The
        best-direction reduction stays on the device (``subnet_best``) and
        preserves the first strictly positive maximum.

        Parameters
        ----------
        inputs : sequence of tuple
            One ``(rms, td00, td90)`` packet per cluster with the shapes of
            :meth:`__call__`; every packet must have at least one pixel. Delay
            counts may differ between packets.
        n_ifo, n_sky : int
            Detector count (2 or 3) and sky-grid size.
        FP, FX, ml : numpy.ndarray
            Shared geometry as in :meth:`__call__`.
        threshold : float
            Pixel energy threshold, passed as float32.
        e2or, subcut : float
            As in :meth:`__call__`.

        Returns
        -------
        list of tuple
            One ``(l_max, stat, Em, Am, lm, Vm, suball, EE)`` tuple per packet,
            in input order, with ``l_max``, ``lm`` and ``Vm`` as ``int``.

        Raises
        ------
        ValueError
            If the geometry or a packet has invalid shapes or delays, the sky
            grid alone exceeds the score budget, or a batch needs more than
            32-bit flat indexing.
        """
        if n_ifo not in (2, 3) or n_sky < 1:
            raise ValueError("Invalid detector/sky dimensions")
        if (
            FP.shape != (n_sky, n_ifo)
            or FX.shape != FP.shape
            or ml.shape != (n_ifo, n_sky)
        ):
            raise ValueError("Invalid geometry shapes")
        max_batch = SCORE_BUDGET_BYTES // (n_sky * SCORE_COLUMNS * 8)
        if max_batch < 1:
            raise ValueError("Sky grid exceeds score workspace budget")
        for rms, td0, td9 in inputs:
            if (
                rms.ndim != 2
                or rms.shape[1] != n_ifo
                or len(rms) < 1
                or td0.ndim != 3
                or td0.shape != (td0.shape[0], n_ifo, len(rms))
                or td9.shape != td0.shape
            ):
                raise ValueError("Invalid packet shapes")
            offset = td0.shape[0] // 2
            if np.any(ml + offset < 0) or np.any(ml + offset >= td0.shape[0]):
                raise ValueError("Invalid packet delays")
        device_fp, device_fx, delays = resident_geometry(
            self.geometry, (FP, FX, ml), (np.float32, np.float32, np.int32)
        )
        results: list[tuple[Any, ...]] = []
        for begin in range(0, len(inputs), max_batch):
            batch = inputs[begin : begin + max_batch]
            nc = len(batch)
            pixels = np.array([len(x[0]) for x in batch], np.int32)
            pixel_offsets = np.concatenate(
                ([0], np.cumsum(pixels[:-1], dtype=np.int64))
            ).astype(np.int32)
            sizes = np.array([x[1].size for x in batch], np.int64)
            td_offsets = np.concatenate(([0], np.cumsum(sizes[:-1]))).astype(np.int32)
            if sizes.sum() > np.iinfo(np.int32).max:
                raise ValueError("Packet batch exceeds 32-bit indexing")
            # Flat float32 concatenations of every packet's rms, td00 and td90.
            device_rms, device_td00, device_td90 = (
                cuda.to_device(
                    np.ascontiguousarray(
                        np.concatenate([x[j].ravel() for x in batch]), dtype=np.float32
                    )
                )
                for j in range(3)
            )
            delay_counts = np.array([x[1].shape[0] for x in batch], np.int32)
            (
                device_pixels,
                device_pixel_offsets,
                device_td_offsets,
                device_delay_counts,
            ) = (
                cuda.to_device(x)
                for x in (pixels, pixel_offsets, td_offsets, delay_counts)
            )
            scores = cuda.device_array((nc, n_sky, SCORE_COLUMNS), np.float64)
            best = cuda.device_array((nc, BEST_COLUMNS), np.float64)
            # Order matches subnet_batch(FP, FX, rms, td0, td9, ml, pixels,
            # pixel_offsets, td_offsets, delays, nc, nsky, nd, threshold, Es,
            # subcut, scores).
            batch_args = [
                device_fp,
                device_fx,
                device_rms,
                device_td00,
                device_td90,
                delays,
                device_pixels,
                device_pixel_offsets,
                device_td_offsets,
                device_delay_counts,
                ct.c_int(nc),
                ct.c_int(n_sky),
                ct.c_int(n_ifo),
                ct.c_float(threshold),
                ct.c_float(2 * e2or),
                ct.c_double(subcut),
                scores,
            ]
            self.module.launch("subnet_batch", nc * n_sky, batch_args)
            # One block of SUBNET_BEST_THREADS per cluster; see the constant's docstring.
            best_args = [scores, ct.c_int(nc), ct.c_int(n_sky), best]
            self.module.launch(
                "subnet_best",
                nc * SUBNET_BEST_THREADS,
                best_args,
                threads=SUBNET_BEST_THREADS,
            )
            for a in best.copy_to_host():
                results.append(
                    (int(a[0]), a[1], a[2], a[3], int(a[0]), int(a[4]), a[5], a[6])
                )
        return results
