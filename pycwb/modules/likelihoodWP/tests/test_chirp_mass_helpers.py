"""Characterization tests for chirp mass helper kernels.

Uses small synthetic arrays to verify _hough_count_overlaps_numba and
_fine_search_numba produce stable, finite results.
"""

import numpy as np
import pytest


class TestChirpMassHelpers:
    """Test chirp mass helper kernels."""

    def setup_method(self):
        """Build small synthetic data matching expected shapes."""
        rng = np.random.RandomState(123)
        self.n_pts = 20
        self.n_mass = 11  # -5 to 5 in steps of 1

        self.x = np.linspace(0.0, 10.0, self.n_pts, dtype=np.float64)
        self.y = np.linspace(0.5, 2.0, self.n_pts, dtype=np.float64)
        self.xerr = np.full(self.n_pts, 0.1, dtype=np.float64)
        self.yerr = np.full(self.n_pts, 0.05, dtype=np.float64)
        self.wgt = np.ones(self.n_pts, dtype=np.float64)

        # kk constant (same as C++: 256*pi/5 * (G*SM*pi/C^3)^(5/3) * sF^(8/3))
        self.kk = 1.0  # simplified for testing
        self.m_vals = np.linspace(-5.0, 5.0, self.n_mass, dtype=np.float64)

    # ------------------------------------------------------------------
    # _hough_count_overlaps_numba
    # ------------------------------------------------------------------
    def test_hough_returns_finite_integers(self):
        """Hough count returns non-negative integer array of correct shape."""
        from pycwb.modules.likelihoodWP.detection_statistics import _hough_count_overlaps_numba

        nsel_arr = _hough_count_overlaps_numba(
            self.x, self.y, self.xerr, self.yerr, self.kk, self.m_vals,
        )
        assert len(nsel_arr) == self.n_mass
        assert np.all(np.isfinite(nsel_arr))
        assert np.all(nsel_arr >= 0)
        assert nsel_arr.dtype == np.int64

    def test_hough_max_in_range(self):
        """Maximum count does not exceed number of points."""
        from pycwb.modules.likelihoodWP.detection_statistics import _hough_count_overlaps_numba

        nsel_arr = _hough_count_overlaps_numba(
            self.x, self.y, self.xerr, self.yerr, self.kk, self.m_vals,
        )
        assert int(np.max(nsel_arr)) <= self.n_pts

    # ------------------------------------------------------------------
    # _fine_search_numba
    # ------------------------------------------------------------------
    def test_fine_search_returns_finite(self):
        """Fine search returns finite (m0, b0)."""
        from pycwb.modules.likelihoodWP.detection_statistics import (
            _hough_count_overlaps_numba, _fine_search_numba,
        )

        nsel_arr = _hough_count_overlaps_numba(
            self.x, self.y, self.xerr, self.yerr, self.kk, self.m_vals,
        )
        nselmax = int(np.max(nsel_arr))
        cand_indices = np.where(nsel_arr == nselmax)[0].astype(np.int64)

        m0, b0 = _fine_search_numba(
            self.x, self.y, self.xerr, self.yerr, self.wgt,
            self.kk, self.m_vals, cand_indices, nselmax, 2.5,
        )
        assert np.isfinite(m0)
        assert np.isfinite(b0)

    # ------------------------------------------------------------------
    # Alias identity
    # ------------------------------------------------------------------
    def test_alias_identity(self):
        """Chirp helper aliases point to the same numba functions."""
        from pycwb.modules.likelihoodWP.detection_statistics import (
            _hough_count_overlaps_numba, _count_chirp_track_overlaps_numba,
            _fine_search_numba, _fit_chirp_track_candidates_numba,
        )
        assert _count_chirp_track_overlaps_numba is _hough_count_overlaps_numba
        assert _fit_chirp_track_candidates_numba is _fine_search_numba

    def test_alias_produces_same_output(self):
        """Both names produce identical output for the same input."""
        from pycwb.modules.likelihoodWP.detection_statistics import (
            _hough_count_overlaps_numba, _count_chirp_track_overlaps_numba,
        )

        a = _hough_count_overlaps_numba(
            self.x, self.y, self.xerr, self.yerr, self.kk, self.m_vals,
        )
        b = _count_chirp_track_overlaps_numba(
            self.x, self.y, self.xerr, self.yerr, self.kk, self.m_vals,
        )
        np.testing.assert_array_equal(a, b)
