"""Ordered CUDA support and bounded atomic sparse compaction with host ordering.

``selection_cuda.cu`` aligns the detector energy maps for one lag
(``align_support``) and compacts the pixels that pass the native support cuts
into fixed-capacity buffers with an atomic counter (``select_sparse``). The
atomic order is arbitrary, so the host restores native ``(frequency, time)``
order with a stable sort before returning. Overflowing the capacity is an
error, never permission to drop pixels.
"""

from __future__ import annotations

import ctypes as ct
from pathlib import Path
from typing import Any

import numpy as np
from numba import cuda

from .cuda_runtime import CUDAModule, load_module

OUTPUT_BUDGET_BYTES = 256 * 1024**2
"""Largest sparse payload (``capacity * (24 + 16 * n_detectors)`` bytes) per session."""

DEFAULT_CAPACITY = 65536
"""Default number of sparse output rows reserved per :meth:`SelectionSession.select` call."""


class SelectionSession:
    """Device-resident detector maps and sparse-selection buffers for one job segment.

    Parameters
    ----------
    maps : numpy.ndarray
        Float64 detector energy maps, shape ``(n_detectors, n_frequency, n_time)``,
        all axes non-empty. Copied to the device once.
    valid_start : int
        First time bin of the valid (non-edge) interval.
    nn_valid : int
        Length of the valid interval; lag shifts wrap inside it.
    ib : int
        Lowest frequency layer considered, ``0 <= ib <= n_frequency``.
    veto : array-like, optional
        Int16 per-time-bin veto flags, shape ``(n_time,)``; all ones when omitted.
    module : CUDAModule, optional
        Compiled ``selection_cuda.cu``; loaded through
        :func:`cuda_runtime.load_module` when omitted.

    Attributes
    ----------
    maps, veto : device arrays
        Resident inputs.
    support : device array
        Float64 ``(n_frequency, n_time)`` combined support map, rewritten per call.
    live : device array
        Boolean ``(n_time,)`` mask of live time bins, rewritten per call.
    counter : device array
        Single uint32 atomic counter reset before each ``select_sparse`` launch.
    outputs : list or None
        Sparse buffers ``[frequency int64, time int64, total float64,
        energies float64 (capacity, n_detectors), detector indices int64
        (capacity, n_detectors)]``; reallocated when the capacity changes.

    Raises
    ------
    ValueError
        If ``maps`` is not a non-empty float64 3-D array, the bounds are
        inconsistent, the map exceeds 31-bit indexing or ``veto`` has the
        wrong shape.
    """

    def __init__(
        self,
        maps: np.ndarray,
        valid_start: int,
        nn_valid: int,
        ib: int,
        veto: np.ndarray | None = None,
        module: CUDAModule | None = None,
    ) -> None:
        maps = np.asarray(maps)
        if maps.dtype != np.float64 or maps.ndim != 3 or min(maps.shape) < 1:
            raise ValueError("Expected nonempty FP64 detector/frequency/time maps")
        _nd, nf, nt = maps.shape
        self.shape = maps.shape
        self.start, self.length, self.ib = int(valid_start), int(nn_valid), int(ib)
        if not 0 <= self.start <= self.start + self.length <= nt or not 0 <= self.ib <= nf:
            raise ValueError("Invalid support bounds")
        if nf * nt >= 2**31:
            raise ValueError("Map exceeds CUDA index range")
        veto = np.ones(nt, np.int16) if veto is None else np.asarray(veto, dtype=np.int16)
        if veto.shape != (nt,):
            raise ValueError("Veto must match the time axis")
        self.module = module or load_module(Path(__file__).with_suffix(".cu"))
        self.maps = cuda.to_device(np.ascontiguousarray(maps))
        self.veto = cuda.to_device(np.ascontiguousarray(veto))
        self.support = cuda.device_array((nf, nt), np.float64)
        self.live = cuda.device_array(nt, np.bool_)
        self.counter = cuda.device_array(1, np.uint32)
        self.zero = np.zeros(1, np.uint32)
        self.capacity = 0
        self.outputs: list[Any] | None = None

    def select(
        self,
        shifts: np.ndarray,
        energy_threshold: float,
        ie: int,
        edge: int,
        capacity: int = DEFAULT_CAPACITY,
    ) -> tuple[tuple[np.ndarray, ...], np.ndarray]:
        """Select the network pixels of one lag.

        Parameters
        ----------
        shifts : numpy.ndarray
            Integer per-detector time shifts, shape ``(n_detectors,)``; reduced
            modulo the valid length.
        energy_threshold : float
            Finite, non-negative pixel energy threshold ``Eo``.
        ie : int
            Highest frequency layer considered; clamped to ``[ib, n_frequency - 1)``.
        edge : int
            Time-edge margin in bins, at least 2.
        capacity : int, optional
            Sparse rows reserved; ``1 <= capacity < 2**31``.

        Returns
        -------
        tuple
            ``((frequency, time, total, energies, detector_indices), live)``:
            host copies of the first ``count`` rows of :attr:`outputs`, stably
            sorted by ``frequency * n_time + time``, and the ``(n_time,)``
            boolean live mask.

        Raises
        ------
        ValueError
            If ``shifts`` has the wrong shape or dtype, the threshold is not a
            finite non-negative number, or ``capacity`` is out of range.
        MemoryError
            If the sparse payload would exceed :data:`OUTPUT_BUDGET_BYTES`.
        OverflowError
            If more pixels pass than ``capacity``; nothing is returned.
        """
        nd, nf, nt = self.shape
        shifts = np.asarray(shifts)
        if shifts.shape != (nd,) or shifts.dtype.kind not in "iu":
            raise ValueError("Expected integer detector shifts")
        eo = float(energy_threshold)
        if not np.isfinite(eo) or eo < 0:
            raise ValueError("Threshold must be finite and nonnegative")
        if not isinstance(capacity, int) or not 1 <= capacity < 2**31:
            raise ValueError("Invalid sparse output capacity")
        if capacity * (24 + 16 * nd) > OUTPUT_BUDGET_BYTES:
            raise MemoryError("Sparse payload exceeds 256 MiB capacity budget")
        if capacity != self.capacity:
            self.outputs = [
                cuda.device_array(capacity, np.int64),
                cuda.device_array(capacity, np.int64),
                cuda.device_array(capacity, np.float64),
                cuda.device_array((capacity, nd), np.float64),
                cuda.device_array((capacity, nd), np.int64),
            ]
            self.capacity = capacity
        device_shifts = cuda.to_device(np.ascontiguousarray(shifts % max(1, self.length), dtype=np.int64))
        dims = [
            ct.c_int(nd),
            ct.c_int(nf),
            ct.c_int(nt),
            ct.c_int(self.start),
            ct.c_int(self.length),
        ]
        self.counter.copy_to_device(self.zero)
        # Order matches align_support(maps, shifts, veto, nd, nf, nt, start,
        # length, ib, eo, support, live).
        align_args = [
            self.maps,
            device_shifts,
            self.veto,
            *dims,
            ct.c_int(self.ib),
            ct.c_double(eo),
            self.support,
            self.live,
        ]
        self.module.launch("align_support", nf * nt, align_args)
        # Order matches select_sparse(support, maps, shifts, nd, nf, nt, start,
        # length, ib, ie, edge, eo, capacity, counter, frequency, time, total,
        # energies, detector_indices).
        select_args = [
            self.support,
            self.maps,
            device_shifts,
            *dims,
            ct.c_int(self.ib),
            ct.c_int(min(max(int(ie), self.ib), nf - 1)),
            ct.c_int(max(int(edge), 2)),
            ct.c_double(eo),
            ct.c_uint(capacity),
            self.counter,
            *self.outputs,
        ]
        self.module.launch("select_sparse", nf * nt, select_args)
        count = int(self.counter.copy_to_host()[0])
        if count > capacity:
            raise OverflowError(f"{count} selected pixels exceed capacity {capacity}")
        host = [v[:count].copy_to_host() for v in self.outputs]
        order = np.argsort(host[0] * nt + host[1], kind="stable")
        return tuple(v[order] for v in host), self.live.copy_to_host()
