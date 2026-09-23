"""Ordered CUDA DPF-regulator pass with trial-scoped resident antenna patterns.

The kernel in ``dpf_regulator.cu`` evaluates one sky direction per thread with
the same ordered FP32 pixel/detector arithmetic as the native
``likelihoodWP.dpf_regulator.dpf_index_only`` loop. The final count reduction
stays on the host in float64, exactly as in ``calculate_dpf_scalar``.
"""

from __future__ import annotations

import ctypes as ct
from pathlib import Path
from typing import Any

import numpy as np

from . import flags
from .cuda_runtime import DeviceBuffers, load_module
from .geometry_cache import resident_geometry


class DPFRegulator:
    """Callable counterpart to ``likelihoodWP.dpf_regulator.calculate_dpf_scalar``.

    Bound by ``processor.py`` as ``_calculate_dpf_scalar`` inside a specialized
    clone of ``likelihoodWP.likelihood.likelihood`` when ``PYCWB_GPU_DPF=1``.

    Attributes
    ----------
    module : CUDAModule
        Process-wide compiled ``dpf_regulator.cu``.
    geometry : dict
        Cache of device copies of the antenna patterns keyed by
        :func:`geometry_cache.geometry_key`. ``FP``/``FX`` are immutable
        per-trial geometry; the entry retains the host arrays so their identity
        keys stay valid. Release this object at trial end so pointer reuse can
        never alias stale geometry.
    workspace : Workspace or None
        Reusable device slots when ``PYCWB_GPU_REUSE_WORKSPACE=1`` was set at
        construction time; otherwise every call allocates fresh device arrays.
    buffers : DeviceBuffers
        Upload/output/download helper wrapping ``workspace``.

    Notes
    -----
    Supports 2 or 3 detectors, the sizes for which the kernel keeps its
    fixed-length per-detector registers.
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
        FP: np.ndarray,
        FX: np.ndarray,
        rms: np.ndarray,
        n_sky: int,
        n_ifo: int,
        gamma_regulator: float,
        network_energy_threshold: float,
        sky_valid_indices: np.ndarray,
    ) -> float:
        """Return the DPF energy regulator for one cluster.

        Parameters
        ----------
        FP, FX : numpy.ndarray
            Plus/cross antenna patterns, shape ``(n_sky, n_ifo)``. Uploaded
            once per distinct geometry as float32 and kept resident.
        rms : numpy.ndarray
            Pixel noise weights, shape ``(n_pix, n_ifo)``; transferred as
            float32 on every call.
        n_sky, n_ifo : int
            Sky-grid size and detector count used to validate the inputs.
        gamma_regulator : float
            Threshold for counting directions by scalar DPF index.
        network_energy_threshold : float
            Multiplier applied to the final regulator.
        sky_valid_indices : array-like
            Sky directions to evaluate; converted to a contiguous int64 vector.

        Returns
        -------
        float
            ``(FF**2 / (ff**2 + 1e-9) - 1) * network_energy_threshold`` with the
            native float64 count reduction, or ``-network_energy_threshold``
            when no direction is selected (the kernel is not launched).

        Raises
        ------
        ValueError
            If ``n_ifo`` is not 2 or 3, the array shapes disagree, or a sky
            index lies outside ``[0, n_sky)``.
        """
        FP = np.asarray(FP)
        FX = np.asarray(FX)
        rms = np.asarray(rms)
        if (
            n_ifo not in (2, 3)
            or FP.shape != (n_sky, n_ifo)
            or FX.shape != FP.shape
            or rms.ndim != 2
            or rms.shape[1] != n_ifo
        ):
            raise ValueError("DPF CUDA requires 2/3 detectors and matching sky/pixel shapes")
        skies = np.ascontiguousarray(sky_valid_indices, dtype=np.int64)
        if skies.ndim != 1 or np.any(skies < 0) or np.any(skies >= n_sky):
            raise ValueError("Invalid sky indices")
        if not len(skies):
            return -float(network_energy_threshold)
        device_fp, device_fx = resident_geometry(self.geometry, (FP, FX), (np.float32, np.float32))
        weights = self.buffers.upload("weights", rms, np.float32)
        indices = self.buffers.upload("skies", skies, np.int64)
        out = self.buffers.output("output", (len(skies),), np.float64)
        # Order matches dpf_index(fp0, fx0, rms, skies, n_valid, n_pix, n_ifo, out).
        kernel_args = [
            device_fp,
            device_fx,
            weights,
            indices,
            ct.c_int(len(skies)),
            ct.c_int(len(rms)),
            ct.c_int(n_ifo),
            out,
        ]
        self.module.launch("dpf_index", len(skies), kernel_args)
        values = self.buffers.download(out, (len(skies),))
        FF = np.float64(len(skies))
        ff = np.float64(np.count_nonzero(values > gamma_regulator))
        return (FF**2 / (ff**2 + 1.0e-9) - 1) * network_energy_threshold
