"""Resident GPU pixel selection with the native coherence payload contract."""
from __future__ import annotations
from typing import Any
import jax
import numpy as np
from pycwb.constants.gpu_options import gpu_options
from pycwb.modules.coherence_native import selection
from .alignment_jax import AlignmentSession
from .selection_jax import select_pixels

INITIAL_SELECTION_CAPACITY = 65536
"""Initial sparse capacity; doubled on overflow, never truncated."""

class GPUSelector:
    """Resident-map pixel selection for one trial; a drop-in for the native selector.

    One instance owns the device copies of every resolution's energy maps and
    is called from exactly one thread. The JAX alignment/selection pair is the
    default; ``gpu.selection_cuda=true`` selects the direct CUDA backend.

    Attributes
    ----------
    sessions : dict
        ``id(selection_cache) -> (cache, session, veto)`` for resident maps.
    use_cuda : bool
        Whether the CUDA selection backend is enabled.
    cuda_module : CUDAModule or None
        Compiled selection kernels shared by every CUDA session.
    """

    def __init__(self, options=None) -> None:
        self.options = gpu_options(options)
        self.sessions: dict[int, tuple[dict[str, Any], Any, np.ndarray | None]] = {}
        self.use_cuda = self.options.selection_cuda
        self.cuda_module: Any | None = None

    def _session(self, cache: dict[str, Any], veto: np.ndarray | None) -> Any:
        """Return the resident session for ``cache``, rebuilding on a veto change."""
        key = id(cache)
        cached = self.sessions.get(key)
        same_veto = cached is not None and (
            (cached[2] is None and veto is None)
            or (
                cached[2] is not None
                and veto is not None
                and np.array_equal(cached[2], veto)
            )
        )
        if same_veto:
            return cached[1]
        if self.use_cuda:
            from pycwb.modules.coherence_gpu.selection_cuda import SelectionSession

            session = SelectionSession(
                cache["arrays_stack"],
                cache["valid_start"],
                cache["nn_valid"],
                cache["ib"],
                veto,
                self.cuda_module,
            )
            self.cuda_module = session.module
        else:
            session = AlignmentSession(
                cache["arrays_stack"],
                cache["valid_start"],
                cache["nn_valid"],
                cache["ib"],
                veto,
            )
        self.sessions[key] = (cache, session, None if veto is None else veto.copy())
        return session

    def __call__(
        self,
        tf_maps: Any,
        lag_index: int,
        energy_threshold: float,
        lag_shifts: Any = None,
        veto: np.ndarray | None = None,
        edge: float = 0.0,
        selection_cache: dict[str, Any] | None = None,
        *,
        preindex_shifts: bool = False,
    ) -> dict[str, Any]:
        """Select network pixels for one lag with the native payload layout.

        Parameters
        ----------
        tf_maps
            Ignored; the resident device maps come from ``selection_cache``.
        lag_index : int
            Zero-based lag index used to look up pre-indexed shifts.
        energy_threshold : float
            Network energy threshold ``Eo`` of this resolution.
        lag_shifts
            Per-detector time shifts in seconds, used when the cache has no
            pre-indexed shift table for ``lag_index``.
        veto : numpy.ndarray or None
            Per-time-bin keep mask. Like the native selector, a mask whose
            length differs from ``n_time`` is ignored.
        edge : float
            Segment edge in seconds; the cache already holds ``edge_bins``.
        selection_cache : dict
            Prepared native selection cache of this resolution. Required.

        Returns
        -------
        dict
            The native candidate payload: ``mask``, ``time``, ``frequency``,
            ``energy``, ``pix_det_energy``, ``pix_det_index``, ``rate``,
            ``layers``, ``start``, ``stop``, ``f_low``, ``f_high``,
            ``live_mask`` and ``live_samples``.

        Raises
        ------
        ValueError
            If no selection cache is supplied.
        OverflowError
            If the selected pixel count exceeds the whole map, which cannot
            happen for valid inputs; smaller overflows retry with more capacity.
        """
        cache = selection_cache
        if cache is None:
            raise ValueError("GPU selector requires a prepared selection cache")
        veto_array = (
            None
            if veto is None or len(veto) != cache["n_time"]
            else np.asarray(veto, dtype=np.int16)
        )
        session = self._session(cache, veto_array)
        table = cache.get("shift_bins_by_lag") if preindex_shifts else None
        if table is not None and 0 <= lag_index < len(table):
            shifts = np.asarray(table[lag_index], dtype=np.int64)
        else:
            shifts = selection._shift_bins_from_lag_shifts(
                lag_shifts, cache["n_ifo"], cache["rate"]
            )
        if not self.use_cuda:
            with jax.default_device(session.device):
                support, device_live = session.align(shifts[None, :], energy_threshold)
        capacity = INITIAL_SELECTION_CAPACITY
        full = cache["n_freq"] * cache["n_time"]
        while True:
            try:
                if self.use_cuda:
                    payload, live = session.select(
                        shifts,
                        energy_threshold,
                        cache["ie"],
                        cache["edge_bins"],
                        capacity,
                    )
                else:
                    with jax.default_device(session.device):
                        payload = select_pixels(
                            session,
                            support[0],
                            shifts,
                            energy_threshold,
                            cache["ie"],
                            cache["edge_bins"],
                            capacity,
                        )
                        live = np.asarray(device_live[0])
                break
            except OverflowError:
                if capacity >= full:
                    raise
                capacity = min(capacity * 2, full)
        frequency, time, energy, det_energy, det_index = payload
        mask = np.zeros((cache["n_freq"], cache["n_time"]), dtype=np.bool_)
        mask[frequency, time] = True
        return {
            "mask": mask,
            "time": time,
            "frequency": frequency,
            "energy": energy,
            "pix_det_energy": det_energy,
            "pix_det_index": det_index,
            "rate": float(cache["rate"]),
            "layers": int(cache["n_freq"]),
            "start": float(cache["start"]),
            "stop": float(cache["stop"]),
            "f_low": float(cache["f_low"]),
            "f_high": float(cache["f_high"]),
            "live_mask": live,
            "live_samples": int(np.sum(live)),
        }
