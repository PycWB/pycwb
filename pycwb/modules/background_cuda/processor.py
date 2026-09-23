"""Opt-in end-to-end background processor with GPU-accelerated lag stages.

The processor reuses the native preparation, scientific stages, output and
restart handling of :mod:`pycwb.workflow.subflow.process_job_segment_native`.
Accelerated stages are composed with :func:`~.binding.specialize`, which clones
the native caller with a private globals dictionary; no production module is
mutated. Stages without an enabled GPU switch execute the unchanged CPU code.

Select it in ``user_parameters.yaml`` with
``segment_processer: pycwb.modules.background_cuda.processor.process_job_segment``
and enable stages through the ``PYCWB_GPU_*`` switches documented in the
package README. All switches are read at call time through :mod:`.flags`.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable, Iterable
from typing import Any

import jax
import numpy as np

from pycwb.modules.coherence_native import pipeline, selection
from pycwb.workflow.subflow import process_job_segment_native as native

from . import flags
from .alignment_jax import AlignmentSession
from .binding import specialize
from .selection_jax import select_pixels

logger = logging.getLogger(__name__)

# Retained for modules written against the previous private name.
_specialize = specialize

INITIAL_SELECTION_CAPACITY = 65536
"""Sparse selection capacity tried first; doubled on overflow, never truncated."""

MAX_LAG_WORKERS = 6
"""Spawned GPU lag workers validated on the 12 GB reference device."""


class GPUSelector:
    """Resident-map pixel selection for one trial; a drop-in for the native selector.

    One instance owns the device copies of every resolution's energy maps and
    is called from exactly one thread. The JAX alignment/selection pair is the
    default; ``PYCWB_GPU_SELECTION_CUDA=1`` selects the direct CUDA backend.

    Attributes
    ----------
    sessions : dict
        ``id(selection_cache) -> (cache, session, veto)`` for resident maps.
    use_cuda : bool
        Whether the CUDA selection backend is enabled.
    cuda_module : CUDAModule or None
        Compiled selection kernels shared by every CUDA session.
    """

    def __init__(self) -> None:
        self.sessions: dict[int, tuple[dict[str, Any], Any, np.ndarray | None]] = {}
        self.use_cuda = flags.enabled("SELECTION_CUDA")
        self.cuda_module: Any | None = None

    def _session(self, cache: dict[str, Any], veto: np.ndarray | None) -> Any:
        """Return the resident session for ``cache``, rebuilding on a veto change."""
        key = id(cache)
        cached = self.sessions.get(key)
        same_veto = cached is not None and (
            (cached[2] is None and veto is None)
            or (cached[2] is not None and veto is not None and np.array_equal(cached[2], veto))
        )
        if same_veto:
            return cached[1]
        if self.use_cuda:
            from .selection_cuda import SelectionSession

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
                cache["arrays_stack"], cache["valid_start"], cache["nn_valid"], cache["ib"], veto
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
        veto_array = None if veto is None or len(veto) != cache["n_time"] else np.asarray(veto, dtype=np.int16)
        session = self._session(cache, veto_array)
        table = cache.get("shift_bins_by_lag")
        if table is not None and 0 <= lag_index < len(table):
            shifts = np.asarray(table[lag_index], dtype=np.int64)
        else:
            shifts = selection._shift_bins_from_lag_shifts(lag_shifts, cache["n_ifo"], cache["rate"])
        if not self.use_cuda:
            with jax.default_device(session.device):
                support, device_live = session.align(shifts[None, :], energy_threshold)
        capacity = INITIAL_SELECTION_CAPACITY
        full = cache["n_freq"] * cache["n_time"]
        while True:
            try:
                if self.use_cuda:
                    payload, live = session.select(shifts, energy_threshold, cache["ie"], cache["edge_bins"], capacity)
                else:
                    with jax.default_device(session.device):
                        payload = select_pixels(
                            session, support[0], shifts, energy_threshold, cache["ie"], cache["edge_bins"], capacity
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


def _build_analyzer() -> tuple[Callable[[Any, int], Any], GPUSelector]:
    """Build the per-lag analysis function with the enabled GPU stage bindings.

    Called once per serial caller or once per spawned worker, so every process
    owns its device sessions, geometry caches and workspaces.

    Returns
    -------
    tuple
        ``(analyze, selector)``: a clone of the native ``_run_lag_analysis``
        and the :class:`GPUSelector` whose sessions the caller releases.
    """
    selector = GPUSelector()
    select: Callable[..., Any] = selector
    if flags.enabled("VALIDATE_STAGES"):
        from .validation import paired

        select = paired(selector, selection.select_network_pixels, "selection")
    bindings: dict[str, Any] = {
        "coherence_single_lag": specialize(pipeline.coherence_single_lag, select_network_pixels=select)
    }
    if flags.enabled("EVENT_GEOMETRY_CACHE"):
        from .event_geometry import CachedGeometryEvent

        bindings["Event"] = CachedGeometryEvent
    if flags.enabled("DPF"):
        from .dpf_regulator import DPFRegulator

        likelihood_module = importlib.import_module("pycwb.modules.likelihoodWP.likelihood")
        bindings["likelihood"] = specialize(
            likelihood_module.likelihood, _calculate_dpf_scalar=DPFRegulator(), _SCALAR_DPF=True
        )
    if flags.enabled("LIKELIHOOD"):
        from .likelihood_scan import LikelihoodScan

        scan = LikelihoodScan()
        bindings["likelihood"] = specialize(
            bindings.get("likelihood", native.likelihood),
            _scan_sky_scratch=scan,
            _scan_sky_grouped_delays=scan,
            _scan_sky_for_best_fit=scan,
        )
    if flags.enabled("CHIRP"):
        from .chirp_bootstrap import make_chirp_update

        bindings["likelihood"] = specialize(
            bindings.get("likelihood", native.likelihood), _update_cluster_chirp_statistics=make_chirp_update()
        )
    if flags.enabled("SUBNET") or flags.enabled("SUBNET_BATCH"):
        from .subnet_scan import SubnetScan

        subnet = importlib.import_module("pycwb.modules.super_cluster_native.sub_net_cut")
        utils = importlib.import_module("pycwb.modules.super_cluster_native.utils")
        supercluster = importlib.import_module("pycwb.modules.super_cluster_native.super_cluster")
        packets = specialize(subnet._sub_net_cut_prepared_packets, optimze_sky_loc_from_td=SubnetScan())
        cut = specialize(subnet.sub_net_cut_from_pixel_arrays, _sub_net_cut_prepared_packets=packets)
        apply: Callable[..., Any] = specialize(utils.apply_subnet_cut, sub_net_cut_from_pixel_arrays=cut)
        if flags.enabled("SUBNET_BATCH"):
            from .subnet_batch import BatchedSubnet

            apply = BatchedSubnet()
        bindings["supercluster_single_lag"] = specialize(supercluster.supercluster_single_lag, apply_subnet_cut=apply)
    if flags.enabled("TD"):
        from .td_vectors import GPUTimeDelays

        bindings["supercluster_single_lag"] = specialize(
            bindings.get("supercluster_single_lag", native.supercluster_single_lag),
            _populate_td_vectors=GPUTimeDelays(),
        )
    if flags.enabled("VALIDATE_STAGES"):
        from .validation import paired

        for name, mutable_arg in (("coherence_single_lag", None), ("supercluster_single_lag", 2), ("likelihood", 1)):
            reference = getattr(native, name)
            bindings[name] = paired(bindings.get(name, reference), reference, name, mutable_arg)
    return specialize(native._run_lag_analysis, **bindings), selector


def _process_lags(context: Any, output_context: Any, skip_lags: dict[int, set[int]] | None) -> None:
    """Run every pending lag and write outputs through the parent-only writer.

    Parameters
    ----------
    context
        Native ``LagAnalysisContext`` with prepared inputs.
    output_context
        Native ``LagOutputContext``; only this process writes catalog files.
    skip_lags : dict or None
        Native resume record of already committed lags.
    """
    from .output_buffer import OutputWriter

    workers = flags.worker_count("LAG_WORKERS", maximum=MAX_LAG_WORKERS)
    writer = OutputWriter(output_context)
    if workers > 1 and not context.sub_job_seg.injections:
        from .process_parallel import process_lags

        process_lags(context, output_context, skip_lags, workers, writer.save)
    else:
        if workers > 1:
            logger.info("GPU lag workers run serially for injection jobs")
        analyze, selector = _build_analyzer()
        try:
            pending: Iterable[int] = native._iter_pending_lags(context, skip_lags)
            for lag in pending:
                writer.save(output_context, analyze(context, lag))
        finally:
            selector.sessions.clear()
    # On failure, leave pending progress uncommitted for native resume cleanup.
    writer.close()


def process_job_segment(*args: Any, **kwargs: Any) -> Any:
    """Configuration entry point; native CPU preparation with GPU lag stages.

    Accepts the native ``process_job_segment`` arguments. Preparation stages
    are replaced by their bounded parallel variants when the corresponding
    ``PYCWB_GPU_*_WORKERS`` switch is set, and lag processing always uses
    :func:`_process_lags`.

    Raises
    ------
    RuntimeError
        If JAX x64 is disabled or no GPU device is visible; both are checked
        before any preparation work.
    ValueError
        If a ``lag_processor`` is supplied; this entry point owns it.
    """
    if not jax.config.x64_enabled:
        raise RuntimeError("Set JAX_ENABLE_X64=1 for the experimental GPU processor")
    jax.devices("gpu")  # Fail before preparation if the GPU is unavailable.
    if "lag_processor" in kwargs:
        raise ValueError("This entry point owns its lag processor")
    # JAX CPU setup remains the numerical reference; the selector explicitly
    # enters a GPU device scope.
    process = native.process_job_segment
    # An input provider owns decode admission and returns job-local copies.
    # Keep its serial read path instead of spawning unaccounted nested readers.
    if flags.text("READ_WORKERS", "1") != "1" and kwargs.get("input_provider") is None:
        from .read_parallel import read_from_job_segment

        process = specialize(process, read_from_job_segment=read_from_job_segment)
    if flags.text("CONDITION_WORKERS", "1") != "1":
        from .conditioning_parallel import data_conditioning

        process = specialize(process, data_conditioning=data_conditioning)
    if flags.text("TD_SETUP_WORKERS", "1") != "1":
        from .td_setup_parallel import build_td_inputs_cache

        process = specialize(process, build_td_inputs_cache=build_td_inputs_cache)
    if flags.text("SETUP_WORKERS", "1") != "1" or flags.enabled("MAX_ENERGY") or flags.enabled("WDM_PREFILTER"):
        from .setup_parallel import setup_coherence

        process = specialize(process, setup_coherence=setup_coherence)
    if flags.enabled("OVERLAP_SETUP"):
        from .setup_overlap import OverlappedSetup

        preparation = OverlappedSetup(
            process.__globals__["setup_coherence"], process.__globals__["build_td_inputs_cache"]
        )
        process = specialize(
            process,
            setup_coherence=preparation.setup_coherence,
            build_td_inputs_cache=preparation.build_td_inputs_cache,
        )
    with jax.default_device(jax.devices("cpu")[0]):
        return process(*args, **kwargs, lag_processor=_process_lags)


# Native preparation accepts owned samples supplied by the allocation cache.
process_job_segment.supports_input_provider = True  # type: ignore[attr-defined]
