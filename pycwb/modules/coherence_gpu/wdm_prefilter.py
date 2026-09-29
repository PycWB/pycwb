"""Isolated CUDA WDM prefilter prototype; not enabled in the job processor.

``wdm_prefilter.cu`` evaluates the block-contracted filter products that feed
the native WDM forward transform. Only ``mode=6`` (six filter blocks, the
native CPU YNN reduction order) has been validated.
"""

from __future__ import annotations

import ctypes as ct
from pathlib import Path

import numpy as np
from numba import cuda

from pycwb.utils.gpu.cuda_runtime import load_module

OUTPUT_BUDGET_BYTES = 256 * 1024**2
"""Largest float64 ``(n_time, 2 * m)`` prefilter output per call."""


class Prefilter:
    """CUDA WDM prefilter.

    Attributes
    ----------
    module : CUDAModule
        Process-wide compiled ``wdm_prefilter.cu``.
    """

    def __init__(self) -> None:
        self.module = load_module(Path(__file__).with_suffix(".cu"))

    def __call__(self, signal: np.ndarray, filt: np.ndarray, m: int, stride: int, nt: int, mode: int = 0) -> np.ndarray:
        """Contract the padded signal with the WDM filter blocks.

        Parameters
        ----------
        signal : array-like
            Padded float64 time series of length at least
            ``(nt - 1) * stride + 2 * len(filt)``.
        filt : array-like
            Float64 wavelet filter of ``1 + 2 * m * blocks`` taps.
        m : int
            Number of WDM layers; the output has ``2 * m`` columns.
        stride : int
            Sample step between output time bins.
        nt : int
            Number of output time bins.
        mode : int, optional
            Contraction/reduction mode ``0..6``; ``6`` is the validated
            six-block CPU YNN order.

        Returns
        -------
        numpy.ndarray
            Float64 array of shape ``(nt, 2 * m)``.

        Raises
        ------
        ValueError
            If the dimensions are invalid, the filter length is not a whole
            number of blocks, ``mode`` is unknown or does not match the block
            count, the signal padding is too short, or the output exceeds
            31-bit indexing or :data:`OUTPUT_BUDGET_BYTES`.
        """
        signal = np.ascontiguousarray(signal, dtype=np.float64)
        filt = np.ascontiguousarray(filt, dtype=np.float64)
        if signal.ndim != 1 or filt.ndim != 1 or m < 1 or stride < 1 or nt < 1:
            raise ValueError("Invalid WDM prefilter dimensions")
        if (len(filt) - 1) % (2 * m) or len(filt) < 2:
            raise ValueError("Filter must contain complete WDM blocks plus center")
        if mode not in range(7):
            raise ValueError("Unknown prefilter contraction/reduction mode")
        if mode == 6 and (len(filt) - 1) // (2 * m) != 6:
            raise ValueError("CPU YNN order is validated only for six filter blocks")
        if len(signal) < (nt - 1) * stride + 2 * len(filt):
            raise ValueError("Signal padding is too short")
        if 2 * m * nt >= 2**31 or 2 * m * nt * 8 > OUTPUT_BUDGET_BYTES:
            raise ValueError("Prefilter output exceeds index or 256 MiB workspace limit")
        source, taps = cuda.to_device(signal), cuda.to_device(filt)
        output = cuda.device_array((nt, 2 * m), np.float64)
        # Order matches wdm_prefilter(signal, filter, output, taps, columns,
        # stride, nt, mode).
        dimensions = [ct.c_int(x) for x in (len(filt), 2 * m, stride, nt, mode)]
        kernel_args = [source, taps, output, *dimensions]
        self.module.launch("wdm_prefilter", nt * 2 * m, kernel_args)
        return output.copy_to_host()
