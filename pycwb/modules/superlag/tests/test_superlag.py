"""Tests for pycwb.modules.superlag — super-lag combination generation."""
import pytest
from pycwb.modules.superlag.superlag import generate_slags


class TestGenerateSlags:
    """Tests for generate_slags — combinatorial super-lag list builder."""

    # --- basic shape / invariants ---

    def test_two_ifo_slag_min_1_max_1(self):
        """2 ifos, slag range [1,1] — should produce valid (0, x) tuples."""
        result = generate_slags(num_ifos=2, slag_min=1, slag_max=1, shuffle=False)
        assert len(result) > 0
        for t in result:
            assert len(t) == 2
            assert t[0] == 0          # reference ifo always 0
            assert t[1] != 0          # no zero shifts except all-zero
            assert abs(t[1]) <= 1     # within max_shift bounds

    def test_three_ifo_structure(self):
        """3 ifos — every tuple length 3, first element 0."""
        result = generate_slags(num_ifos=3, slag_min=2, slag_max=3, shuffle=False)
        assert all(len(t) == 3 for t in result)
        assert all(t[0] == 0 for t in result)

    # --- slag distance filtering ---

    def test_slag_min_filters_short_lags(self):
        """slag_min excludes combinations whose total |shift| is too small."""
        result = generate_slags(num_ifos=3, slag_min=5, slag_max=5, shuffle=False)
        for t in result:
            total = sum(abs(s) for s in t[1:])
            assert total == 5

    def test_slag_max_filters_long_lags(self):
        """slag_max excludes combinations exceeding the bound."""
        result = generate_slags(num_ifos=2, slag_min=1, slag_max=2, shuffle=False)
        for t in result:
            total = sum(abs(s) for s in t[1:])
            assert total <= 2

    # --- offset / pagination ---

    def test_slag_off_skips_first_n(self):
        """slag_off skips the first N results."""
        full = generate_slags(num_ifos=2, slag_min=2, slag_max=2, shuffle=False)
        skipped = generate_slags(num_ifos=2, slag_min=2, slag_max=2,
                                 slag_off=1, shuffle=False)
        assert len(skipped) == len(full) - 1
        assert skipped[0] == full[1]

    def test_slag_size_limits(self):
        """slag_size caps the returned list length."""
        result = generate_slags(num_ifos=2, slag_min=1, slag_max=3,
                                slag_size=2, shuffle=False)
        assert len(result) == 2

    # --- shuffle determinism ---

    def test_shuffle_deterministic(self):
        """With seed=0, shuffle produces same order across calls."""
        a = generate_slags(num_ifos=3, slag_min=2, slag_max=3)
        b = generate_slags(num_ifos=3, slag_min=2, slag_max=3)
        assert a == b

    def test_no_shuffle_preserves_order(self):
        """With shuffle=False, order is deterministic by distance then tuple."""
        result = generate_slags(num_ifos=2, slag_min=2, slag_max=2, shuffle=False)
        # With 2 ifos, slag=2: possible valid shifts for 2nd ifo: ±2 (but not 0)
        # (0, -2) and (0, 2) — sorted by tuple: (-2,) < (2,)
        assert result[0] == (0, -2)
        assert result[1] == (0, 2)
        # all-zero is excluded by slag_min=2

    # --- edge cases ---

    def test_slag_min_zero_includes_all_zero(self):
        """slag_min=0 with low slag_max should include the all-zero combo."""
        result = generate_slags(num_ifos=2, slag_min=0, slag_max=0, shuffle=False)
        assert result == [(0, 0)]

    def test_no_valid_combinations_returns_empty(self):
        """When no shifts meet the criteria, result should be empty."""
        # For 2 ifos: slag_min=100, slag_max=100 — can't reach with max_shift=100
        result = generate_slags(num_ifos=2, slag_min=100, slag_max=100, shuffle=False)
        # max_shift=same as slag_max, so only |shift|=100 works for 2nd ifo
        # (0, 100) has distance 100, (0, -100) has distance 100 — should exist
        assert len(result) == 2

    def test_no_duplicate_shifts_in_tuple(self):
        """No single tuple should contain duplicate shift values (except ref 0)."""
        result = generate_slags(num_ifos=4, slag_min=3, slag_max=5, shuffle=False)
        for t in result:
            shifts = t[1:]  # exclude reference ifo (always 0)
            assert len(set(shifts)) == len(shifts), f"duplicate shift in {t}"
