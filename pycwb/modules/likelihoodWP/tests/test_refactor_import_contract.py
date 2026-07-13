"""Characterization tests for refactor import contract.

These tests lock down the public API and friendly aliases so that the split
(Phase A / Phase B) can be verified mechanically.  They must pass both before
and after the refactor.
"""

import pytest
import numpy as np
import importlib
import typing

# ---------------------------------------------------------------------------
# 0.1.1 — Public API entry-point imports
# ---------------------------------------------------------------------------

def test_package_level_imports():
    """Top-level package imports still work."""
    import pycwb.modules.likelihoodWP as likelihoodWP
    from pycwb.modules.likelihoodWP import (
        setup_likelihood, likelihood, likelihood_wrapper,
        prepare_likelihood_inputs, evaluate_cluster_likelihood,
        evaluate_fragment_clusters,
    )
    assert likelihoodWP.__all__ == [
        "setup_likelihood", "likelihood", "likelihood_wrapper",
        "prepare_likelihood_inputs",
        "evaluate_cluster_likelihood", "evaluate_fragment_clusters",
    ]
    assert setup_likelihood is not None
    assert likelihood is not None
    assert likelihood_wrapper is not None
    assert prepare_likelihood_inputs is setup_likelihood
    assert evaluate_cluster_likelihood is likelihood
    assert evaluate_fragment_clusters is likelihood_wrapper


def test_module_level_entry_imports():
    """Entry-point imports from likelihood.py work."""
    from pycwb.modules.likelihoodWP.likelihood import (
        setup_likelihood, likelihood, likelihood_wrapper,
    )
    assert setup_likelihood is not None
    assert likelihood is not None
    assert likelihood_wrapper is not None


def test_likelihood_facade_has_bounded_public_surface():
    """The entry-point facade does not keep accidental helper exports."""
    facade = importlib.import_module("pycwb.modules.likelihoodWP.likelihood")
    assert facade.__all__ == [
        "setup_likelihood", "likelihood", "likelihood_wrapper",
        "prepare_likelihood_inputs",
        "evaluate_cluster_likelihood", "evaluate_fragment_clusters",
    ]
    assert not hasattr(facade, "calculate_dpf")


def test_phase_module_star_exports_exclude_private_helpers():
    """Implementation helpers stay importable by explicit name but out of __all__."""
    from pycwb.modules.likelihoodWP import detection_statistics, likelihood_setup, pixel_data

    assert "_hough_count_overlaps_numba" not in detection_statistics.__all__
    assert "_fine_search_numba" not in detection_statistics.__all__
    assert "_populate_pixel_noise_rms" not in likelihood_setup.__all__
    assert "_load_data_from_pixel_arrays" not in pixel_data.__all__


def test_runtime_type_hints_resolve_for_public_functions():
    """Docs and IDE tooling can resolve annotations after the split."""
    from pycwb.modules.likelihoodWP import detection_statistics, likelihood_setup, pixel_data

    functions = [
        pixel_data.load_data_from_pixels,
        pixel_data.load_data_from_ifo,
        likelihood_setup.setup_likelihood,
        likelihood_setup._populate_pixel_noise_rms,
        detection_statistics.fill_detection_statistic,
        detection_statistics.threshold_cut,
    ]

    for fn in functions:
        assert typing.get_type_hints(fn)


def test_legacy_utils_shim_exports_packet_ops():
    """Old likelihoodWP.utils imports remain available during the shim period."""
    from pycwb.modules.likelihoodWP.packet_ops import avx_packet_ps as packet_avx_packet_ps
    from pycwb.modules.likelihoodWP.utils import avx_packet_ps, build_wavelet_packet

    assert avx_packet_ps is packet_avx_packet_ps
    assert build_wavelet_packet is avx_packet_ps


# ---------------------------------------------------------------------------
# 0.1.2 — Friendly alias identity (entry points)
# ---------------------------------------------------------------------------

def test_friendly_alias_identity():
    """Friendly aliases refer to the same function objects as legacy names."""
    from pycwb.modules.likelihoodWP.likelihood import (
        setup_likelihood, prepare_likelihood_inputs,
        likelihood, evaluate_cluster_likelihood,
        likelihood_wrapper, evaluate_fragment_clusters,
    )
    assert prepare_likelihood_inputs is setup_likelihood
    assert evaluate_cluster_likelihood is likelihood
    assert evaluate_fragment_clusters is likelihood_wrapper


# ---------------------------------------------------------------------------
# 0.1.3 — Friendly alias identity (phase functions)
# ---------------------------------------------------------------------------

def test_packet_alias_identity():
    """Packet kernel friendly aliases match legacy names."""
    from pycwb.modules.likelihoodWP.packet_ops import (
        avx_packet_ps, build_wavelet_packet,
        avx_noise_ps, compute_gaussian_noise_correction,
        avx_setAMP_ps, normalize_packet_amplitudes,
        avx_loadNULL_ps, compute_null_packet,
        avx_pol_ps, project_onto_network_plane,
        packet_norm_numpy, compute_packet_norms,
        gw_norm_numpy, compute_signal_norms,
        xtalk_energy_sum_numpy, sum_xtalk_corrected_energy,
        orthogonalize_and_rotate, orthogonalize_packet_basis,
    )
    assert build_wavelet_packet is avx_packet_ps
    assert compute_gaussian_noise_correction is avx_noise_ps
    assert normalize_packet_amplitudes is avx_setAMP_ps
    assert compute_null_packet is avx_loadNULL_ps
    assert project_onto_network_plane is avx_pol_ps
    assert compute_packet_norms is packet_norm_numpy
    assert compute_signal_norms is gw_norm_numpy
    assert sum_xtalk_corrected_energy is xtalk_energy_sum_numpy
    assert orthogonalize_packet_basis is orthogonalize_and_rotate


def test_likelihood_helper_alias_identity():
    """Phase helper aliases inside likelihood.py match legacy names."""
    from pycwb.modules.likelihoodWP.likelihood_setup import (
        _populate_pixel_noise_rms, _populate_pixel_noise_from_maps,
    )
    from pycwb.modules.likelihoodWP.pixel_data import (
        load_data_from_pixels, extract_pixel_time_delay_data,
        load_data_from_ifo, build_sky_delay_and_antenna_patterns,
        _load_data_from_pixel_arrays, _extract_pixel_array_time_delay_data,
    )
    from pycwb.modules.likelihoodWP.sky_scan import (
        find_optimal_sky_localization, scan_sky_for_best_fit,
    )
    from pycwb.modules.likelihoodWP.sky_statistics import (
        calculate_sky_statistics, compute_statistics_at_sky_position,
    )
    from pycwb.modules.likelihoodWP.detection_statistics import (
        threshold_cut, get_likelihood_rejection_reason,
        fill_detection_statistic, populate_detection_statistics,
        get_chirp_mass, update_chirp_mass_statistics,
        get_error_region, compute_sky_error_region,
        _hough_count_overlaps_numba, _count_chirp_track_overlaps_numba,
        _fine_search_numba, _fit_chirp_track_candidates_numba,
    )
    assert extract_pixel_time_delay_data is load_data_from_pixels
    assert build_sky_delay_and_antenna_patterns is load_data_from_ifo
    assert scan_sky_for_best_fit is find_optimal_sky_localization
    assert compute_statistics_at_sky_position is calculate_sky_statistics
    assert get_likelihood_rejection_reason is threshold_cut
    assert populate_detection_statistics is fill_detection_statistic
    assert update_chirp_mass_statistics is get_chirp_mass
    assert compute_sky_error_region is get_error_region
    assert _populate_pixel_noise_from_maps is _populate_pixel_noise_rms
    assert _extract_pixel_array_time_delay_data is _load_data_from_pixel_arrays
    assert _count_chirp_track_overlaps_numba is _hough_count_overlaps_numba
    assert _fit_chirp_track_candidates_numba is _fine_search_numba
