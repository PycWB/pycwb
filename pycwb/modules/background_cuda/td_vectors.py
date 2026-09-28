"""Trial-resident CUDA TD filters; exact native ordered FP64 sums and CPU phases.

``td_vectors.cu`` reproduces ``pycwb.utils.td_vector_kernels.batch_get_td_vecs``
for one detector/layer cache whose padded quadrature planes and filter tables
stay resident on the device for the whole trial. Phase factors are computed by
the native CPU ``math`` functions (:func:`phase_table`) and uploaded, so no
CUDA libm result enters the sums.
"""

from __future__ import annotations

import ctypes as ct
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
from numba import cuda, njit

from pycwb.constants.gpu_options import gpu_options
from .cuda_runtime import CUDAModule, DeviceBuffers, load_module

logger = logging.getLogger(__name__)

OUTPUT_BUDGET_BYTES = 256 * 1024**2
"""Largest TD output batch per call: ``n_pix * (4K+2)`` float32 values, i.e. ``n_pix * (2K+1) * 8`` bytes."""

TD_WORKSPACE_BUDGET_BYTES = 384 * 1024**2
"""Resident budget of the shared TD workspace under ``gpu.reuse_td_workspace=true``."""

RESIDENT_INPUT_BUDGET_BYTES = 2 * 1024**3
"""Total bytes of detector/layer planes and filter tables kept on the device."""


@njit(cache=True)
def phase_table(M: int, J: int, radius: int) -> np.ndarray:
    """Tabulate the delay phase factors for every layer and sub-sample delay.

    Parameters
    ----------
    M : int
        Number of frequency layers (Nyquist index).
    J : int
        Maximum delay index of the filter bank.
    radius : int
        Largest absolute sub-sample delay used, ``<= J``.

    Returns
    -------
    numpy.ndarray
        Float64 array of shape ``(M + 1, 2 * radius + 1, 4)`` holding
        ``cos(phase), sin(phase), sin(low), sin(high)`` for layer ``m`` and
        delay ``d`` at ``[m, d + radius]``, computed with CPU ``math``.
    """
    result = np.empty((M + 1, 2 * radius + 1, 4), dtype=np.float64)
    for m in range(M + 1):
        for d in range(-radius, radius + 1):
            dt = float(d) * float(M) / float(J)
            phase = m * math.pi * dt / M
            low = (2.0 * m - 1.0) * dt * math.pi / (2.0 * M)
            high = (2.0 * m + 1.0) * dt * math.pi / (2.0 * M)
            result[m, d + radius, 0] = math.cos(phase)
            result[m, d + radius, 1] = math.sin(phase)
            result[m, d + radius, 2] = math.sin(low)
            result[m, d + radius, 3] = math.sin(high)
    return result


class TDSession:
    """Own one immutable detector/layer cache and its device buffers.

    Parameters
    ----------
    inputs : pycwb.types.td_batch_inputs.TDBatchInputs
        Prepared extraction inputs. ``padded00``/``padded90`` must be float32
        planes of shape ``(n_time + 2 * n_coeffs, n_cached_bands)``; ``T0``/``Tx``
        must be float64 tables of shape ``(2 * J + 1, 2 * n_coeffs + 1)``.
    module : CUDAModule, optional
        Compiled ``td_vectors.cu`` shared with other sessions; compiled through
        :func:`cuda_runtime.load_module` when omitted.
    workspace : Workspace, optional
        Shared reusable output/index slots. When omitted and
        ``gpu.reuse_td_workspace=true`` is set, a private workspace with
        :data:`TD_WORKSPACE_BUDGET_BYTES` is created.

    Attributes
    ----------
    device : list
        Device copies of ``padded00``, ``padded90``, ``T0`` and ``Tx`` in that
        order; they live as long as the session.
    phases : dict[int, device array]
        :func:`phase_table` uploads keyed by delay radius.
    buffers : DeviceBuffers
        Upload/output/download helper wrapping ``workspace``.

    Raises
    ------
    ValueError
        If the dimensions are non-positive or the planes/tables do not have
        the required dtypes and shapes.
    """

    def __init__(
        self,
        inputs: Any,
        module: CUDAModule | None = None,
        workspace: Any | None = None,
        *,
        options=None,
    ) -> None:
        self.options = gpu_options(options)
        if inputs.M <= 0 or inputs.J <= 0 or inputs.n_coeffs < 0:
            raise ValueError("Invalid TD dimensions")
        self.inputs = inputs
        self.module = module or load_module(Path(__file__).with_suffix(".cu"))
        p0, p9 = inputs.padded00, inputs.padded90
        shape = (2 * inputs.J + 1, 2 * inputs.n_coeffs + 1)
        if (
            p0.ndim != 2
            or p0.dtype != np.float32
            or p9.dtype != np.float32
            or p9.shape != p0.shape
        ):
            raise ValueError("CUDA TD requires matching compact FP32 planes")
        if (
            inputs.T0.shape != shape
            or inputs.Tx.shape != shape
            or inputs.T0.dtype != np.float64
            or inputs.Tx.dtype != np.float64
        ):
            raise ValueError("TD filter dimensions do not match inputs")
        self.device = [
            cuda.to_device(np.ascontiguousarray(a))
            for a in (
                p0,
                p9,
                np.asarray(inputs.T0, dtype=np.float64),
                np.asarray(inputs.Tx, dtype=np.float64),
            )
        ]
        self.phases: dict[int, Any] = {}
        self.workspace = workspace
        if workspace is None and self.options.reuse_td_workspace:
            from .workspace import Workspace

            self.workspace = Workspace(budget=TD_WORKSPACE_BUDGET_BYTES)
        self.buffers = DeviceBuffers(self.workspace)

    def extract_td_vecs(
        self, pixel_indices: np.ndarray, K: int, delay_stride: int = 1
    ) -> np.ndarray:
        """Extract time-delay vectors for ``pixel_indices``; mirrors ``TDBatchInputs.extract_td_vecs``.

        Parameters
        ----------
        pixel_indices : array-like
            Flat pixel indices into the TF plane with the global ``M + 1``
            frequency stride; converted to a contiguous int32 vector.
        K : int
            Delay half-range; each quadrature has ``2 * K + 1`` values.
        delay_stride : int, optional
            Spacing in fine filter-bank delay samples, at least 1.

        Returns
        -------
        numpy.ndarray
            Float32 array of shape ``(n_pix, 4 * K + 2)``: in-phase values in
            the first ``2 * K + 1`` columns, quadrature values in the rest. A
            new host array on every call (never a view of device storage).

        Raises
        ------
        ValueError
            If ``K`` or ``delay_stride`` are not valid integers, an index is
            negative, or a pixel's frequency/time support lies outside the
            cached planes.
        MemoryError
            If the output batch exceeds :data:`OUTPUT_BUDGET_BYTES`.
        """
        if (
            int(K) != K
            or K < 0
            or int(delay_stride) != delay_stride
            or delay_stride < 1
        ):
            raise ValueError("Invalid TD delay range or stride")
        K, stride = int(K), int(delay_stride)
        indices = np.ascontiguousarray(pixel_indices, dtype=np.int32)
        if indices.ndim != 1 or np.any(indices < 0):
            raise ValueError("TD indices must be a nonnegative vector")
        c = self.inputs
        bands = indices % (c.M + 1)
        if np.any(np.maximum(bands - 1, 0) < c.frequency_offset) or np.any(
            np.minimum(bands + 1, c.M) >= c.frequency_offset + c.padded00.shape[1]
        ):
            raise ValueError("Requested support outside cached frequency bands")
        shifts = 2 * ((K * stride + c.J) // (2 * c.J))
        times = indices // (c.M + 1)
        if np.any(times - shifts < 0) or np.any(
            times + shifts + 2 * c.n_coeffs >= len(c.padded00)
        ):
            raise ValueError("Requested support outside padded time range")
        count = len(indices) * (2 * K + 1)
        if count * 8 > OUTPUT_BUDGET_BYTES:
            raise MemoryError("TD output exceeds 256 MiB batch budget")
        if not len(indices):
            return np.empty((0, 4 * K + 2), np.float32)
        radius = min(K * stride, c.J)
        if radius not in self.phases:
            self.phases[radius] = cuda.to_device(phase_table(c.M, c.J, radius))
        output_shape = (len(indices), 4 * K + 2)
        out = self.buffers.output("output", output_shape, np.float32)
        device_indices = self.buffers.upload("indices", indices, np.int32)
        # Order matches td_vectors(indices, p0, p9, T0, Tx, phase, np, M, taps, K,
        # J, stride, bands, offset, radius, out).
        kernel_args = [
            device_indices,
            *self.device,
            self.phases[radius],
            ct.c_int(len(indices)),
            ct.c_int(c.M),
            ct.c_int(c.n_coeffs),
            ct.c_int(K),
            ct.c_int(c.J),
            ct.c_int(stride),
            ct.c_int(c.padded00.shape[1]),
            ct.c_int(c.frequency_offset),
            ct.c_int(radius),
            out,
        ]
        self.module.launch("td_vectors", count, kernel_args)
        result = self.buffers.download(out, output_shape)
        if self.options.validate_td:
            expected = c.extract_td_vecs(indices, K, delay_stride=stride)
            np.testing.assert_array_equal(expected.view("u4"), result.view("u4"))
            logger.info("GPU TD parity: values=%d exact=1", result.size)
        return result


class GPUTimeDelays:
    """Private replacement for native ``_populate_td_vectors``; keeps its ownership rules.

    Bound by ``processor.py`` into ``supercluster_single_lag`` when
    ``gpu.td=true``. Sessions are created lazily per distinct
    ``TDBatchInputs`` object (keyed by ``id``) and kept for the lifetime of
    this object, so the cache entries must stay alive as long as it does.

    Attributes
    ----------
    module : CUDAModule
        Compiled ``td_vectors.cu`` shared by every session.
    sessions : dict[int, TDSession]
        Resident sessions keyed by ``id(inputs)``.
    resident_bytes : int
        Host bytes of planes and tables mirrored on the device so far.
    workspace : Workspace or None
        Shared scratch under ``gpu.reuse_td_workspace=true``. Every
        extraction is synchronous, so detector/layer sessions can share it
        without retaining one peak allocation per layer.
    """

    def __init__(self, options=None) -> None:
        self.options = gpu_options(options)
        self.module = load_module(Path(__file__).with_suffix(".cu"))
        self.sessions: dict[int, TDSession] = {}
        self.resident_bytes = 0
        self.workspace: Any | None = None
        if self.options.reuse_td_workspace:
            from .workspace import Workspace

            self.workspace = Workspace(budget=TD_WORKSPACE_BUDGET_BYTES)

    def __call__(
        self,
        all_clusters: list[Any],
        n_ifo: int,
        K: int,
        td_inputs_cache: dict[int, Any],
        delay_stride: int = 1,
    ) -> Any:
        """Populate cluster TD vectors through resident device sessions.

        Parameters
        ----------
        all_clusters : list
            Clusters whose pixel-array TD amplitudes are replaced in place.
        n_ifo : int
            Detector count of each cache entry.
        K : int
            Delay half-range passed through to :meth:`TDSession.extract_td_vecs`.
        td_inputs_cache : dict
            Per-layer lists of ``TDBatchInputs`` (one per detector). Entries
            are wrapped in :class:`TDSession` on first use.
        delay_stride : int, optional
            Fine filter-bank delay spacing.

        Returns
        -------
        Any
            Whatever the native ``_populate_td_vectors`` returns.

        Raises
        ------
        MemoryError
            If mirroring another cache entry would exceed
            :data:`RESIDENT_INPUT_BUDGET_BYTES`.
        """
        import importlib

        native = importlib.import_module(
            "pycwb.modules.super_cluster_native.super_cluster"
        )
        owner = self

        class LazyInputs:
            """Dict-like view that hands out sessions in place of ``TDBatchInputs``."""

            def get(self, layer: int) -> list[TDSession] | None:
                values = td_inputs_cache.get(layer)
                if values is None:
                    return None
                result = []
                for value in values:
                    key = id(value)
                    if key not in owner.sessions:
                        size = sum(
                            a.nbytes
                            for a in (
                                value.padded00,
                                value.padded90,
                                value.T0,
                                value.Tx,
                            )
                        )
                        if owner.resident_bytes + size > RESIDENT_INPUT_BUDGET_BYTES:
                            raise MemoryError(
                                "TD resident input cache exceeds 2 GiB budget"
                            )
                        owner.sessions[key] = TDSession(
                            value, owner.module, owner.workspace, options=owner.options
                        )
                        owner.resident_bytes += size
                    result.append(owner.sessions[key])
                return result

        return native._populate_td_vectors(
            all_clusters, n_ifo, K, LazyInputs(), delay_stride=delay_stride
        )
