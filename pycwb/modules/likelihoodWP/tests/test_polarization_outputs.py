"""Analytic network-plane rotation, mask, empty and permutation contracts."""

import numpy as np
import pytest
from pycwb.modules.likelihoodWP.packet_ops import avx_pol_ps


@pytest.mark.parametrize("n", [0, 1, 7, 16])
def test_complete_plane_rotation(n):
    # Orthonormal detector basis: projection is identity, phase rotates (3,4)
    # to (5,0). The cross component (2,-1) rotates to (0.4,-2.2).
    p = np.tile([[3.0], [2.0]], (1, n)).astype(np.float32)
    q = np.tile([[4.0], [-1.0]], (1, n)).astype(np.float32)
    f = np.tile([1.0, 0.0], (n, 1))
    F = np.tile([0.0, 1.0], (n, 1))
    original = p.copy(), q.copy()
    a, b, pol0, pol90 = avx_pol_ps(p, q, np.ones(n), np.ones(n), np.ones(n), f, F)
    np.testing.assert_allclose(a, np.tile([[5.0], [0.4]], (1, n)), rtol=2e-6, atol=1e-7)
    np.testing.assert_allclose(b, np.tile([[0.0], [-2.2]], (1, n)), rtol=2e-6, atol=1e-7)
    np.testing.assert_array_equal(p, original[0])
    np.testing.assert_array_equal(q, original[1])
    assert a.dtype == b.dtype == np.float32
    np.testing.assert_allclose(a * a + b * b, p * p + q * q, rtol=2e-6, atol=1e-7)


def test_mask_and_pixel_permutation():
    rng = np.random.default_rng(320)
    p = rng.normal(size=(3, 13))
    q = rng.normal(size=(3, 13))
    f = rng.normal(size=(13, 3))
    F = rng.normal(size=(13, 3))
    fp = np.sum(f * f, axis=1)
    fx = np.sum(F * F, axis=1)
    mask = np.array([1, 0, -1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1])
    perm = rng.permutation(13)
    a = avx_pol_ps(p, q, mask, fp, fx, f, F)
    b = avx_pol_ps(p[:, perm], q[:, perm], mask[perm], fp[perm], fx[perm], f[perm], F[perm])
    for x, y in zip(a[:2], b[:2]):
        assert np.isfinite(x).all()
        np.testing.assert_array_equal(x[:, perm], y)
        np.testing.assert_array_equal(x[:, mask <= 0], 0.0)
