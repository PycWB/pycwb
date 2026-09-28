"""Public import contracts after the readable-name migration.

Compatibility aliases are deliberately absent; migration instructions live in
README.md. Numerical behavior is covered by the packet/scan/regression suites.
"""

import importlib
import typing

import pytest

PREFIX = "pycwb.modules.likelihoodWP"
ENTRY_POINTS = [
    "prepare_likelihood_inputs",
    "evaluate_cluster_likelihood",
    "evaluate_fragment_clusters",
]


def test_package_and_module_entry_points_agree():
    package = importlib.import_module(PREFIX)
    orchestration = importlib.import_module(f"{PREFIX}.likelihood")
    assert package.__all__ == orchestration.__all__ == ENTRY_POINTS
    for name in ENTRY_POINTS:
        assert getattr(package, name) is getattr(orchestration, name)
        assert callable(getattr(package, name))
    assert not hasattr(orchestration, "setup_likelihood")
    assert not hasattr(orchestration, "likelihood_wrapper")
    assert not hasattr(orchestration, "likelihood")


@pytest.mark.parametrize("module,names,removed", [
    ("dpf", ["compute_dpf", "compute_dpf_into", "compute_dpf_regulator"],
     ["dpf_np_loops_vec", "dpf_np_loops_vec_into", "calculate_dpf"]),
    ("dpf_regulator", ["compute_dpf_index", "compute_dpf_regulator_scalar"],
     ["dpf_index_only", "calculate_dpf_scalar"]),
    ("sky_kernels", ["compute_pixel_energy_and_mask", "project_signal_packet",
                     "orthogonalize_quadratures", "compute_coherent_statistics"],
     ["load_data_from_td", "avx_GW_ps", "avx_ort_ps", "avx_stat_ps"]),
    ("packet_ops", ["build_wavelet_packet", "compute_gaussian_noise_correction",
                    "normalize_packet_amplitudes", "compute_null_packet",
                    "project_onto_network_plane", "compute_packet_norms",
                    "compute_signal_norms", "sum_xtalk_corrected_energy"],
     ["avx_packet_ps", "avx_noise_ps", "avx_setAMP_ps", "avx_loadNULL_ps",
      "avx_pol_ps", "packet_norm_numpy", "gw_norm_numpy", "xtalk_energy_sum_numpy"]),
    ("chirp_hough", ["update_chirp_mass_statistics"],
     ["get_chirp_mass", "_hough_count_overlaps_numba", "_fine_search_numba"]),
    ("detection_statistics", ["get_likelihood_rejection_reason",
                              "populate_detection_statistics", "populate_sky_localization"],
     ["threshold_cut", "fill_detection_statistic", "compute_sky_error_region",
      "get_error_region", "get_chirp_mass", "update_chirp_mass_statistics"]),
    ("pixel_data", ["extract_pixel_time_delay_data", "build_sky_delay_and_antenna_patterns"],
     ["load_data_from_pixels", "load_data_from_ifo", "load_data_from_pixels_vectorized"]),
])
def test_canonical_functions_replace_legacy_aliases(module, names, removed):
    implementation = importlib.import_module(f"{PREFIX}.{module}")
    for name in names:
        function = getattr(implementation, name)
        assert callable(function)
        assert function.__name__ == name
    for name in removed:
        assert not hasattr(implementation, name)


def test_split_modules_have_resolvable_annotations():
    from pycwb.modules.likelihoodWP import chirp_hough, detection_statistics, likelihood_setup, pixel_data

    for function in (
        pixel_data.extract_pixel_time_delay_data,
        pixel_data.build_sky_delay_and_antenna_patterns,
        likelihood_setup.prepare_likelihood_inputs,
        likelihood_setup.populate_pixel_noise_from_maps,
        detection_statistics.populate_detection_statistics,
        detection_statistics.get_likelihood_rejection_reason,
        chirp_hough.update_chirp_mass_statistics,
    ):
        assert typing.get_type_hints(function)
