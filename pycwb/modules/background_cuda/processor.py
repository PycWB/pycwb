"""Opt-in end-to-end background processor with GPU-accelerated lag stages.

The processor reuses the native preparation, scientific stages, output and
restart handling of :mod:`pycwb.workflow.subflow.process_job_segment_native`.
Accelerated stages are composed with :func:`~pycwb.utils.function_binding.specialize`, which clones
the native caller with a private globals dictionary; no production module is
mutated. Stages without an enabled GPU switch execute the unchanged CPU code.

Select it in ``user_parameters.yaml`` with
``segment_processer: pycwb.modules.background_cuda.processor.process_job_segment``
and enable stages through the the ``gpu`` YAML mapping documented in the
package README. All switches are resolved from YAML from the job configuration.
"""

from __future__ import annotations

import importlib
from pycwb.constants.execution_profile import execution_profile
from pycwb.constants.gpu_options import gpu_options
import logging
from collections.abc import Callable, Iterable
from typing import Any

import jax
import numpy as np

from pycwb.modules.coherence_native import selection

coherence = importlib.import_module("pycwb.modules.coherence_native.coherence")
from pycwb.workflow.subflow import process_job_segment_native as native

from .alignment_jax import AlignmentSession
from pycwb.utils.function_binding import specialize
from .selection_jax import select_pixels

logger = logging.getLogger(__name__)


INITIAL_SELECTION_CAPACITY = 65536
"""Sparse selection capacity tried first; doubled on overflow, never truncated."""


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


def _build_analyzer(config=None) -> tuple[Callable[[Any, int], Any], GPUSelector]:
    """Build the per-lag analysis function with the enabled GPU stage bindings.

    Called once per serial caller or once per spawned worker, so every process
    owns its device sessions, geometry caches and workspaces.

    Returns
    -------
    tuple
        ``(analyze, selector)``: a clone of the native ``_run_lag_analysis``
        and the :class:`GPUSelector` whose sessions the caller releases.
    """
    options = gpu_options(config)
    selector = GPUSelector(options)
    select: Callable[..., Any] = selector
    if options.validate_stages:
        from .validation import paired

        select = paired(
            selector, selection.select_network_pixels, "selection", options=options
        )
    bindings: dict[str, Any] = {
        "coherence_single_lag": specialize(
            coherence.coherence_single_lag, select_network_pixels=select
        )
    }
    if options.dpf:
        if not execution_profile(config).scalar_dpf:
            raise ValueError("gpu.dpf requires execution_profile.scalar_dpf=true")
        from .dpf_regulator import DPFRegulator

        likelihood_module = importlib.import_module(
            "pycwb.modules.likelihoodWP.likelihood"
        )
        bindings["evaluate_cluster_likelihood"] = specialize(
            likelihood_module.evaluate_cluster_likelihood,
            _compute_dpf_regulator_scalar=DPFRegulator(options),
        )
    if options.likelihood:
        from .likelihood_scan import LikelihoodScan

        scan = LikelihoodScan(options)
        bindings["evaluate_cluster_likelihood"] = specialize(
            bindings.get(
                "evaluate_cluster_likelihood", native.evaluate_cluster_likelihood
            ),
            _scan_sky=scan.scan_sky,
        )
    if options.chirp:
        from .chirp_bootstrap import make_chirp_update

        bindings["evaluate_cluster_likelihood"] = specialize(
            bindings.get(
                "evaluate_cluster_likelihood", native.evaluate_cluster_likelihood
            ),
            _update_cluster_chirp_statistics=make_chirp_update(),
        )
    if options.subnet or options.subnet_batch:
        supercluster = importlib.import_module(
            "pycwb.modules.super_cluster_native.super_cluster"
        )
        if options.subnet_batch:
            from .subnet_batch import BatchedSubnet

            apply = BatchedSubnet(options)
        else:
            from .subnet_scan import SubnetScan

            subnet = importlib.import_module(
                "pycwb.modules.super_cluster_native.sub_net_cut"
            )
            utils = importlib.import_module("pycwb.modules.super_cluster_native.utils")
            packets = specialize(
                subnet._sub_net_cut_prepared_packets,
                optimze_sky_loc_from_td=SubnetScan(),
            )
            cut = specialize(
                subnet.sub_net_cut_from_pixel_arrays,
                _sub_net_cut_prepared_packets=packets,
            )
            apply = specialize(
                utils.apply_subnet_cut, sub_net_cut_from_pixel_arrays=cut
            )
        bindings["supercluster_single_lag"] = specialize(
            supercluster.supercluster_single_lag, apply_subnet_cut=apply
        )
    if options.td:
        from .td_vectors import GPUTimeDelays

        bindings["supercluster_single_lag"] = specialize(
            bindings.get("supercluster_single_lag", native.supercluster_single_lag),
            _populate_td_vectors=GPUTimeDelays(options),
        )
    if options.validate_stages:
        from .validation import paired

        for name, mutable_arg in (
            ("coherence_single_lag", None),
            ("supercluster_single_lag", 2),
            ("evaluate_cluster_likelihood", 1),
        ):
            reference = getattr(native, name)
            bindings[name] = paired(
                bindings.get(name, reference),
                reference,
                name,
                mutable_arg,
                options=options,
            )
    return specialize(native._run_lag_analysis, **bindings), selector


def _process_lags(
    context: Any, output_context: Any, skip_lags: dict[int, set[int]] | None
) -> None:
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

    workers = gpu_options(context.config).lag_workers
    writer = OutputWriter(output_context)
    if workers > 1 and not context.sub_job_seg.injections:
        from .process_parallel import process_lags

        process_lags(context, output_context, skip_lags, workers, writer.save)
    else:
        if workers > 1:
            logger.info("GPU lag workers run serially for injection jobs")
        analyze, selector = _build_analyzer(context.config)
        try:
            pending: Iterable[int] = native._iter_pending_lags(context, skip_lags)
            for lag in pending:
                writer.save(output_context, analyze(context, lag))
        finally:
            selector.sessions.clear()
    # On failure, leave pending progress uncommitted for native resume cleanup.
    writer.close()


def process_job_segment(working_dir, config, job_seg, *args: Any, **kwargs: Any) -> Any:
    """Configuration entry point; native CPU preparation with GPU lag stages.

    Accepts the native ``process_job_segment`` arguments. Preparation stages
    are replaced by their bounded parallel variants when the corresponding
    ``gpu`` worker count is set, and lag processing always uses
    :func:`_process_lags`.

    Raises
    ------
    RuntimeError
        If JAX x64 is disabled or no GPU device is visible; both are checked
        before any preparation work.
    ValueError
        If a ``lag_processor`` is supplied; this entry point owns it.
    """
    options = gpu_options(config)
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
    if options.read_workers > 1 and kwargs.get("input_provider") is None:
        from .read_parallel import read_from_job_segment

        process = specialize(process, read_from_job_segment=read_from_job_segment)
    if options.condition_workers > 1:
        from .conditioning_parallel import condition_strains

        process = specialize(process, condition_strains=condition_strains)
    if options.td_setup_workers > 1:
        from .td_setup_parallel import build_td_inputs_cache

        process = specialize(process, build_td_inputs_cache=build_td_inputs_cache)
    if options.setup_workers > 1 or options.wdm_prefilter:
        from .setup_parallel import setup_coherence

        process = specialize(process, setup_coherence=setup_coherence)
    if options.overlap_setup:
        from .setup_overlap import OverlappedSetup

        preparation = OverlappedSetup(
            process.__globals__["setup_coherence"],
            process.__globals__["build_td_inputs_cache"],
        )
        process = specialize(
            process,
            setup_coherence=preparation.setup_coherence,
            build_td_inputs_cache=preparation.build_td_inputs_cache,
        )
    with jax.default_device(jax.devices("cpu")[0]):
        return process(
            working_dir, config, job_seg, *args, **kwargs, lag_processor=_process_lags
        )


# Native preparation accepts owned samples supplied by the allocation cache.
process_job_segment.supports_input_provider = True  # type: ignore[attr-defined]


def requested_cores(config):
    """CPU slots for the busiest configured GPU preparation/analysis stage."""
    options = gpu_options(config)
    setup = (options.setup_workers + options.td_setup_workers if options.overlap_setup
             else max(options.setup_workers, options.td_setup_workers))
    return max(options.lag_workers, options.read_workers, options.condition_workers, setup)


process_job_segment.requested_cores = requested_cores
