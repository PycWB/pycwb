"""Pin every private production symbol that ``background_cuda`` binds by name.

The package composes GPU stages by replacing module globals of native callers
through :func:`~pycwb.modules.background_cuda.binding.specialize`. A rename of
any bound name would otherwise silently fall back to CPU execution, so each
expected attribute is asserted here.
"""

from __future__ import annotations

import importlib
import inspect

import pytest

BOUND_SYMBOLS: dict[str, tuple[str, ...]] = {
    "pycwb.workflow.subflow.process_job_segment_native": (
        "_run_lag_analysis",
        "_save_lag_outputs",
        "_iter_pending_lags",
        "_cleanup_lag_output_state",
        "likelihood",
        "supercluster_single_lag",
        "Event",
    ),
    "pycwb.workflow.subflow.process_job_segment_parallel": (
        "_worker_context",
        "_process_shared_inputs",
        "_consume_bounded",
        "_initialize_process",
    ),
    "pycwb.modules.likelihoodWP.likelihood": (
        "_calculate_dpf_scalar",
        "_SCALAR_DPF",
        "_scan_sky_scratch",
        "_scan_sky_grouped_delays",
        "_scan_sky_for_best_fit",
        "_update_cluster_chirp_statistics",
        "likelihood",
    ),
    "pycwb.modules.likelihoodWP.chirp_micropixel": ("_bootstrap", "estimate_chirp"),
    "pycwb.modules.super_cluster_native.sub_net_cut": (
        "_sub_net_cut_prepared_packets",
        "sub_net_cut_from_pixel_arrays",
        "_load_selected_pixel_arrays",
        "optimze_sky_loc_from_td",
    ),
    "pycwb.modules.super_cluster_native.utils": ("apply_subnet_cut", "_top_loudest_indices"),
    "pycwb.modules.super_cluster_native.super_cluster": ("supercluster_single_lag", "_populate_td_vectors"),
    "pycwb.modules.coherence_native.pipeline": ("coherence_single_lag",),
    "pycwb.modules.coherence_native.selection": ("select_network_pixels", "_shift_bins_from_lag_shifts"),
    "pycwb.modules.coherence_native.setup": ("setup_coherence", "_setup_coherence_single_res"),
    "pycwb.modules.coherence_native.time_delay_jax": (
        "_t2w_data_jax",
        "time_delay_max_energy",
        "_time_delay_max_energy_pattern_jit",
        "_time_delay_max_energy_complex_jit",
    ),
    "pycwb.modules.coherence_native.projection": ("max_energy",),
    "pycwb.utils.td_vector_batch": ("_build_td_inputs_single_level", "build_td_inputs_cache"),
    "pycwb.modules.catalog.catalog": ("_write_table_atomic", "PROGRESS_SCHEMA", "Catalog"),
    "pycwb.workflow.subflow.job_segment_progress": ("_catalog_path",),
    "pycwb.workflow.subflow.job_segment_output": ("_postprocess_saved_triggers",),
    "pycwb.workflow.subflow.postprocess_and_plots": ("reconstruct_waveforms_flow",),
    "pycwb.modules.reconstruction": ("get_network_MRA_wave",),
}

CASES = [(module, name) for module, names in BOUND_SYMBOLS.items() for name in names]


@pytest.mark.parametrize(("module_name", "symbol"), CASES, ids=[f"{m.rsplit('.', 1)[1]}.{n}" for m, n in CASES])
def test_bound_symbol_exists(module_name: str, symbol: str) -> None:
    module = importlib.import_module(module_name)
    assert hasattr(module, symbol), f"{module_name} no longer defines {symbol}"


def test_native_processor_reexports_callables_bound_by_the_processor() -> None:
    native = importlib.import_module("pycwb.workflow.subflow.process_job_segment_native")
    for name in ("_run_lag_analysis", "_save_lag_outputs", "_iter_pending_lags", "_cleanup_lag_output_state"):
        assert inspect.isfunction(getattr(native, name)), name
    assert callable(native.likelihood)
    assert callable(native.supercluster_single_lag)
    assert inspect.isclass(native.Event)


def test_specialize_accepts_every_processor_binding_target() -> None:
    """The names the processor rebinds must exist as globals of the rebound functions."""
    from pycwb.modules.background_cuda.binding import specialize

    likelihood = importlib.import_module("pycwb.modules.likelihoodWP.likelihood")
    subnet = importlib.import_module("pycwb.modules.super_cluster_native.sub_net_cut")
    utils = importlib.import_module("pycwb.modules.super_cluster_native.utils")
    supercluster = importlib.import_module("pycwb.modules.super_cluster_native.super_cluster")
    pipeline = importlib.import_module("pycwb.modules.coherence_native.pipeline")
    chirp = importlib.import_module("pycwb.modules.likelihoodWP.chirp_micropixel")
    native = importlib.import_module("pycwb.workflow.subflow.process_job_segment_native")
    shared = importlib.import_module("pycwb.workflow.subflow.process_job_segment_parallel")

    sentinel = object()
    specialize(pipeline.coherence_single_lag, select_network_pixels=sentinel)
    specialize(likelihood.likelihood, _calculate_dpf_scalar=sentinel, _SCALAR_DPF=True)
    specialize(
        likelihood.likelihood,
        _scan_sky_scratch=sentinel,
        _scan_sky_grouped_delays=sentinel,
        _scan_sky_for_best_fit=sentinel,
    )
    specialize(likelihood.likelihood, _update_cluster_chirp_statistics=sentinel)
    specialize(chirp.estimate_chirp, _bootstrap=sentinel)
    specialize(subnet._sub_net_cut_prepared_packets, optimze_sky_loc_from_td=sentinel)
    specialize(subnet.sub_net_cut_from_pixel_arrays, _sub_net_cut_prepared_packets=sentinel)
    specialize(utils.apply_subnet_cut, sub_net_cut_from_pixel_arrays=sentinel)
    specialize(supercluster.supercluster_single_lag, apply_subnet_cut=sentinel, _populate_td_vectors=sentinel)
    specialize(
        native._run_lag_analysis,
        coherence_single_lag=sentinel,
        supercluster_single_lag=sentinel,
        likelihood=sentinel,
        Event=sentinel,
    )
    specialize(native._save_lag_outputs, _postprocess_saved_triggers=sentinel)
    specialize(shared._consume_bounded, native=sentinel)
    specialize(
        shared._process_shared_inputs,
        _initialize_process=sentinel,
        _analyze_process=sentinel,
        _consume_bounded=sentinel,
    )
    specialize(
        native.process_job_segment,
        read_from_job_segment=sentinel,
        data_conditioning=sentinel,
        build_td_inputs_cache=sentinel,
        setup_coherence=sentinel,
    )


def test_event_output_py_has_exactly_two_local_detector_imports() -> None:
    """``event_geometry`` rewrites exactly two ``Detector`` imports in ``Event.output_py``."""
    from pycwb.types.network_event import Event

    source = inspect.getsource(Event.output_py)
    assert source.count("from pycwb.types.detector import Detector") == 2
