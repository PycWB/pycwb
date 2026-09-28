"""Pin remaining native adapters and compatibility imports during migration."""

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
        "evaluate_cluster_likelihood",
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
        "_compute_dpf_regulator_scalar",
        "_scan_sky",
        "_update_cluster_chirp_statistics",
        "evaluate_cluster_likelihood",
    ),
    "pycwb.modules.likelihoodWP.chirp_micropixel": ("_bootstrap", "estimate_chirp"),
    "pycwb.modules.super_cluster_native.sub_net_cut": (
        "_sub_net_cut_prepared_packets",
        "sub_net_cut_from_pixel_arrays",
        "_load_selected_pixel_arrays",
        "optimze_sky_loc_from_td",
    ),
    "pycwb.modules.super_cluster_native.utils": (
        "apply_subnet_cut",
        "_top_loudest_indices",
    ),
    "pycwb.modules.super_cluster_native.super_cluster": (
        "supercluster_single_lag",
        "_populate_td_vectors",
    ),
    "pycwb.modules.coherence_native.coherence": ("coherence_single_lag",),
    "pycwb.modules.coherence_native.selection": (
        "select_network_pixels",
        "_shift_bins_from_lag_shifts",
    ),
    "pycwb.modules.coherence_native.setup": (
        "setup_coherence",
        "_setup_coherence_single_res",
    ),
    "pycwb.modules.coherence_native.time_delay_jax": (
        "_t2w_data_jax",
        "time_delay_max_energy",
        "_time_delay_max_energy_pattern_jit",
        "_time_delay_max_energy_complex_jit",
    ),
    "pycwb.modules.coherence_native.projection": ("max_energy",),
    "pycwb.utils.td_vector_batch": (
        "_build_td_inputs_single_level",
        "build_td_inputs_cache",
    ),
    "pycwb.modules.catalog.catalog": (
        "_write_table_atomic",
        "PROGRESS_SCHEMA",
        "Catalog",
    ),
    "pycwb.workflow.subflow.job_segment_progress": ("_catalog_path",),
    "pycwb.workflow.subflow.job_segment_output": ("_postprocess_saved_triggers",),
    "pycwb.workflow.subflow.postprocess_and_plots": ("reconstruct_waveforms_flow",),
    "pycwb.modules.reconstruction": ("get_network_MRA_wave",),
}

CASES = [(module, name) for module, names in BOUND_SYMBOLS.items() for name in names]


@pytest.mark.parametrize(
    ("module_name", "symbol"),
    CASES,
    ids=[f"{m.rsplit('.', 1)[1]}.{n}" for m, n in CASES],
)
def test_bound_symbol_exists(module_name: str, symbol: str) -> None:
    module = importlib.import_module(module_name)
    assert hasattr(module, symbol), f"{module_name} no longer defines {symbol}"


def test_native_processor_reexports_callables_bound_by_the_processor() -> None:
    native = importlib.import_module(
        "pycwb.workflow.subflow.process_job_segment_native"
    )
    for name in (
        "_run_lag_analysis",
        "_save_lag_outputs",
        "_iter_pending_lags",
        "_cleanup_lag_output_state",
    ):
        assert inspect.isfunction(getattr(native, name)), name
    assert callable(native.evaluate_cluster_likelihood)
    assert callable(native.supercluster_single_lag)
    assert inspect.isclass(native.Event)
