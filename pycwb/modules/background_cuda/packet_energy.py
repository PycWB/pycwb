"""Experimental CUDA packet energy; native CPU FFT input, ordered FP64 math.

``packet_energy.cu`` evaluates the native packet-pattern energy of one WDM
coefficient map and keeps a running maximum over successive delay shifts on
the device, so the full coefficient map is uploaded once per shift and only the
final maximum map is downloaded.
"""

from __future__ import annotations

import ctypes as ct
from pathlib import Path
from typing import Any

import numpy as np
from numba import cuda

from .cuda_runtime import load_module

INPUT_BUDGET_BYTES = 512 * 1024**2
"""Largest complex128 coefficient map accepted per call."""


class PacketEnergy:
    """Device-resident packet-energy maximum for one coefficient-map shape.

    Attributes
    ----------
    module : CUDAModule
        Process-wide compiled ``packet_energy.cu``.
    shape : tuple[int, int] or None
        ``(n_frequency, n_time)`` of the maps currently resident; buffers are
        reallocated when a different shape arrives.
    source : device array or None
        Complex128 ``(n_time, n_frequency)`` time-major input buffer.
    output : device array or None
        Float64 ``(n_time, n_frequency)`` running maximum.
    """

    def __init__(self) -> None:
        self.module = load_module(Path(__file__).with_suffix(".cu"))
        self.shape: tuple[int, int] | None = None
        self.source: Any | None = None
        self.output: Any | None = None

    def __call__(
        self,
        data: np.ndarray,
        pattern: int,
        edge: float,
        wavelet_rate: float,
        f_low: float,
        f_high: float,
        df: float,
        accumulate: bool = False,
    ) -> None:
        """Evaluate one shift's packet energy into the resident maximum.

        Parameters
        ----------
        data : numpy.ndarray
            Complex128 WDM coefficients, shape ``(n_frequency, n_time)``;
            uploaded transposed to time-major order.
        pattern : int
            Native packet pattern, ``0..10``.
        edge : float
            Segment edge in seconds, excluded from the time range.
        wavelet_rate : float
            Time-bin rate used to convert ``edge`` to bins.
        f_low, f_high : float
            Frequency band in Hz, converted to layer indices with ``df``.
        df : float
            Frequency resolution of one layer in Hz.
        accumulate : bool, optional
            ``False`` overwrites the resident maximum, ``True`` takes the
            elementwise maximum with it.

        Raises
        ------
        ValueError
            If ``data`` is not a complex128 2-D array, ``pattern`` is out of
            range, the map exceeds 31-bit indexing or
            :data:`INPUT_BUDGET_BYTES`, or ``accumulate`` is requested across
            different map shapes.
        """
        data = np.asarray(data)
        if data.dtype != np.complex128 or data.ndim != 2 or not 0 <= pattern <= 10:
            raise ValueError("Packet energy requires complex128 frequency/time data and pattern 0..10")
        nf, nt = data.shape
        if nf * nt >= 2**31 or data.nbytes > INPUT_BUDGET_BYTES:
            raise ValueError("Packet workspace exceeds index or 512 MiB input budget")
        if self.shape != data.shape:
            if accumulate:
                raise ValueError("Cannot accumulate across packet shapes")
            self.source = cuda.device_array((nt, nf), np.complex128)
            self.output = cuda.device_array((nt, nf), np.float64)
            self.shape = data.shape
        self.source.copy_to_device(np.ascontiguousarray(data.T))
        jb = max(int(np.floor(edge * float(wavelet_rate) / 4.0)) * nf, 4 * nf)
        je = nf * nt - jb
        low = max(int(np.floor(f_low / df + 0.1)), 0)
        high = min(int(np.floor(f_high / df + 0.1)), nf - 1)
        if pattern in (1, 3, 4, 7, 8, 9):
            low += 1
            high -= 1
        if pattern in (5, 6):
            low += 2
            high -= 2
        # Order matches packet_energy(source, output, nf, nt, pattern, jb, je,
        # low, high, accumulate).
        dimensions = [ct.c_int(x) for x in (nf, nt, pattern, max(jb, 0), min(je, nf * nt), low, high, int(accumulate))]
        kernel_args = [self.source, self.output, *dimensions]
        self.module.launch("packet_energy", nf * nt, kernel_args)

    def result(self) -> np.ndarray:
        """Return the resident maximum as a float64 ``(n_frequency, n_time)`` host array.

        Returns
        -------
        numpy.ndarray
            A transposed view of a fresh host copy of :attr:`output`.
        """
        return self.output.copy_to_host().T
