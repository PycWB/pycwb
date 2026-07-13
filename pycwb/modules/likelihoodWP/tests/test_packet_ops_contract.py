"""Characterization tests for packet-op contract and deterministic kernels.

Verifies alias identity and stable output for pure-NumPy packet helpers.
"""

import numpy as np
import pytest


class TestPacketOpAliases:
    """Verify alias identity for all packet-op kernels."""

    def test_all_aliases(self):
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


class TestComputeNullPacket:
    """compute_null_packet (avx_loadNULL_ps) deterministic test."""

    def test_simple_subtraction(self):
        from pycwb.modules.likelihoodWP.packet_ops import compute_null_packet

        d = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        D = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
        h = np.array([[0.5, 1.0], [1.5, 2.0]], dtype=np.float32)
        H = np.array([[2.5, 3.0], [3.5, 4.0]], dtype=np.float32)

        n, N = compute_null_packet(d, D, h, H)
        np.testing.assert_allclose(n, d - h, rtol=1e-6)
        np.testing.assert_allclose(N, D - H, rtol=1e-6)


class TestXtalkEnergySum:
    """xtalk_energy_sum_numpy behavior with trivial inputs."""

    def test_trivial(self):
        from pycwb.modules.likelihoodWP.packet_ops import xtalk_energy_sum_numpy

        p = np.array([[1.0, 0.5]], dtype=np.float64)   # (1, 2)
        q = np.array([[2.0, 1.0]], dtype=np.float64)
        mk = np.array([1.0, 0.0], dtype=np.float64)     # second pixel masked

        # Single pixel, zero xtalk — the inner loop sees no neighbours
        xtalks = np.zeros((0, 8), dtype=np.float64)
        xtalks_lookup = np.array([[0, 0], [0, 0]], dtype=np.int64)

        result = xtalk_energy_sum_numpy(p, q, xtalks, xtalks_lookup, mk)
        # Only pixel 0 is active but has zero xtalk neighbours → t=0 → g=0
        assert result == 0.0


class TestPacketNormNumpy:
    """packet_norm_numpy with trivial inputs."""

    def test_all_masked_returns_clamped(self):
        from pycwb.modules.likelihoodWP.packet_ops import compute_packet_norms

        p = np.array([[1.0, 2.0]], dtype=np.float32)
        q = np.array([[3.0, 4.0]], dtype=np.float32)
        mk = np.array([0.0, 0.0], dtype=np.float32)
        q_E = np.array([5.0], dtype=np.float32)
        xtalks = np.zeros((0, 8), dtype=np.float64)
        xtalks_lookup = np.array([[0, 0], [0, 0]], dtype=np.int64)

        det_snr, norm, rn, q_norm = compute_packet_norms(
            p, q, xtalks, xtalks_lookup, mk, q_E)
        # With all pixels masked, norm is clamped to 2.0 (C++ minimum),
        # so detector_snr = q_E * 2 / 2 = q_E.
        np.testing.assert_allclose(det_snr, [5.0])
        assert np.all(norm == 2.0)


class TestGWNormNumpy:
    """gw_norm_numpy with trivial inputs."""

    def test_basic(self):
        from pycwb.modules.likelihoodWP.packet_ops import compute_signal_norms

        q_norm = np.array([[0.5, 0.3]], dtype=np.float64)
        q_E = np.array([4.0])
        p_E = np.array([3.0])
        ec = np.array([0.1, 0.0], dtype=np.float64)

        total, norm, new_p_E, p_norm = compute_signal_norms(
            q_norm, q_E, p_E, ec)
        assert total > 0
        assert new_p_E[0] == 4.0  # q_E preserved


class TestOrthogonalizeAndRotate:
    """orthogonalize_and_rotate smoke test (unused but preserved)."""

    def test_import_and_call(self):
        from pycwb.modules.likelihoodWP.packet_ops import orthogonalize_and_rotate
        assert orthogonalize_and_rotate is not None
