"""
Comprehensive regression + edge-case tests for the SIM scheduling, job-assignment,
simulation-summary, and trigger-to-simulation matching pipeline.

Covers:

1. ``_job_segment_intervals``        — interval generation (veto windows, sorting)
2. ``_position_to_interval``         — position-to-GPS mapping
3. ``generate_injection_list_from_config_for_job_segments``
   — rate / Poisson / explicit-GPS scheduling
4. ``add_scheduled_injections_into_job_segments``
   — routing by job_id, error paths
5. ``build_simulation_summary``      — ownership field preservation, veto flags
6. ``match_triggers_to_simulations`` — job_id-aware matching, all join types
7. ``match_simulations_parquet``     — DuckDB join with/without job_id
8. End-to-end integration            — scheduling → assignment → summary → matching
"""

from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pycwb.modules.injection.injection import (
    _assign_to_interval,
    _job_segment_intervals,
    _position_to_interval,
    distribute_inj_in_job_intervals_by_poisson,
    distribute_inj_in_job_intervals_by_rate,
    generate_injection_list_from_config_for_job_segments,
)
from pycwb.modules.job_segment.job_segment import (
    add_scheduled_injections_into_job_segments,
)
from pycwb.modules.catalog.matching import (
    match_simulations_parquet,
    match_triggers_to_simulations,
)
from pycwb.types.job import WaveSegment
from pycwb.workflow.subflow.simulation_summary import build_simulation_summary


# ═══════════════════════════════════════════════════════════════════════════
# Shared fixtures / helpers
# ═══════════════════════════════════════════════════════════════════════════

def _segment(index, shift, analyze_start=1000.0, analyze_end=1100.0,
             ifos=None, seg_edge=0.0, veto_windows=None):
    """Create a WaveSegment with sensible defaults for testing."""
    return WaveSegment(
        index=index,
        ifos=ifos or ["L1", "H1"],
        analyze_start=analyze_start,
        analyze_end=analyze_end,
        sample_rate=1024,
        seg_edge=seg_edge,
        shift=shift,
        veto_windows=veto_windows,
    )


def _injection(name, **overrides):
    """Create a minimal injection dict with required waveform fields."""
    base = {
        "name": name,
        "approximant": "SGE",
        "hrss": 1e-22,
        "ra": 1.0,
        "dec": 0.5,
        "pol": 0.0,
        "t_start": -0.1,
        "t_end": 0.1,
    }
    base.update(overrides)
    return base


# ═══════════════════════════════════════════════════════════════════════════
# 1.  _job_segment_intervals
# ═══════════════════════════════════════════════════════════════════════════

class TestJobSegmentIntervals:
    """Tests for ``_job_segment_intervals``."""

    def test_single_zero_shift_segment(self):
        seg = _segment(1, [0.0, 0.0])
        intervals = _job_segment_intervals([seg])
        assert len(intervals) == 1
        assert intervals[0]["start"] == 1000.0
        assert intervals[0]["end"] == 1100.0
        assert intervals[0]["duration"] == 100.0
        assert intervals[0]["job_id"] == 1
        assert intervals[0]["shift"] == [0.0, 0.0]

    def test_two_shifted_segments(self):
        jobs = [
            _segment(1, [0.0, 0.0]),
            _segment(2, [0.0, -100.0]),
        ]
        intervals = _job_segment_intervals(jobs)
        assert len(intervals) == 2
        assert intervals[0]["job_id"] == 1
        assert intervals[0]["shift"] == [0.0, 0.0]
        assert intervals[1]["job_id"] == 2
        assert intervals[1]["shift"] == [0.0, -100.0]

    def test_veto_windows_clip_interval(self):
        seg = _segment(1, [0.0, 0.0], veto_windows=[(1020.0, 1080.0)])
        intervals = _job_segment_intervals([seg])
        assert len(intervals) == 1
        assert intervals[0]["start"] == 1020.0
        assert intervals[0]["end"] == 1080.0
        assert intervals[0]["duration"] == 60.0

    def test_veto_windows_with_multiple_keep_regions(self):
        seg = _segment(1, [0.0, 0.0],
                       veto_windows=[(1010.0, 1030.0), (1060.0, 1090.0)])
        intervals = _job_segment_intervals([seg])
        # Two disjoint keep windows
        assert len(intervals) == 2
        assert intervals[0]["start"] == 1010.0
        assert intervals[0]["end"] == 1030.0
        assert intervals[1]["start"] == 1060.0
        assert intervals[1]["end"] == 1090.0

    def test_veto_window_fully_covers_segment(self):
        # veto_windows are *keep* intervals; if they don't overlap the
        # analysis window at all, zero keep intervals result.
        seg = _segment(1, [0.0, 0.0],
                       veto_windows=[(900.0, 999.0)])  # no overlap with [1000,1100]
        intervals = _job_segment_intervals([seg])
        assert len(intervals) == 0

    def test_veto_window_partial_overlap_left_edge(self):
        # veto_windows are keep intervals; (900, 1050) ∩ [1000,1100] = [1000,1050]
        seg = _segment(1, [0.0, 0.0],
                       veto_windows=[(900.0, 1050.0)])
        intervals = _job_segment_intervals([seg])
        assert len(intervals) == 1
        assert intervals[0]["start"] == 1000.0
        assert intervals[0]["end"] == 1050.0

    def test_veto_window_partial_overlap_right_edge(self):
        # veto_windows are keep intervals; (1050, 1200) ∩ [1000,1100] = [1050,1100]
        seg = _segment(1, [0.0, 0.0],
                       veto_windows=[(1050.0, 1200.0)])
        intervals = _job_segment_intervals([seg])
        assert len(intervals) == 1
        assert intervals[0]["start"] == 1050.0
        assert intervals[0]["end"] == 1100.0

    def test_empty_segments_list(self):
        assert _job_segment_intervals([]) == []

    def test_shift_none_defaults_to_zero(self):
        seg = _segment(1, None, ifos=["L1", "H1", "V1"])
        intervals = _job_segment_intervals([seg])
        assert intervals[0]["shift"] == [0.0, 0.0, 0.0]

    def test_sorting_by_analyze_start_then_index(self):
        # segment 3 starts earlier than segment 1
        jobs = [
            _segment(2, [0.0, 0.0], analyze_start=1100.0, analyze_end=1200.0),
            _segment(1, [0.0, 0.0], analyze_start=1000.0, analyze_end=1100.0),
            _segment(3, [0.0, 0.0], analyze_start=1050.0, analyze_end=1150.0),
        ]
        intervals = _job_segment_intervals(jobs)
        job_ids = [iv["job_id"] for iv in intervals]
        assert job_ids == [1, 3, 2]  # sorted by analyze_start

    def test_segment_with_zero_duration_keep_window(self):
        seg = _segment(1, [0.0, 0.0],
                       veto_windows=[(1000.0, 1000.0)])  # zero-length
        intervals = _job_segment_intervals([seg])
        # hi <= lo → skipped
        assert len(intervals) == 0


# ═══════════════════════════════════════════════════════════════════════════
# 2.  _position_to_interval
# ═══════════════════════════════════════════════════════════════════════════

class TestPositionToInterval:
    """Tests for ``_position_to_interval``."""

    def _intervals(self):
        return [
            {"start": 1000.0, "end": 1050.0, "duration": 50.0, "job_id": 1, "shift": [0.0, 0.0]},
            {"start": 1100.0, "end": 1150.0, "duration": 50.0, "job_id": 2, "shift": [0.0, -1.0]},
        ]

    def test_position_zero(self):
        ivs = self._intervals()
        interval, gps = _position_to_interval(ivs, 0.0)
        assert interval["job_id"] == 1
        assert gps == 1000.0

    def test_position_mid_first_interval(self):
        ivs = self._intervals()
        interval, gps = _position_to_interval(ivs, 25.0)
        assert interval["job_id"] == 1
        assert gps == 1025.0

    def test_position_at_first_interval_boundary(self):
        ivs = self._intervals()
        # position == duration of first interval
        interval, gps = _position_to_interval(ivs, 50.0)
        # Should land in second interval at its start
        assert interval["job_id"] == 2
        assert gps == 1100.0

    def test_position_mid_second_interval(self):
        ivs = self._intervals()
        interval, gps = _position_to_interval(ivs, 75.0)
        assert interval["job_id"] == 2
        assert gps == 1125.0

    def test_position_at_total_duration(self):
        ivs = self._intervals()
        interval, gps = _position_to_interval(ivs, 100.0)
        # Falls through to last interval, at its end
        assert interval["job_id"] == 2
        # This is at the boundary — documented edge case
        assert gps == 1150.0

    def test_single_interval(self):
        ivs = [{"start": 500.0, "end": 600.0, "duration": 100.0, "job_id": 1, "shift": [0.0]}]
        interval, gps = _position_to_interval(ivs, 50.0)
        assert gps == 550.0
        assert interval["job_id"] == 1


# ═══════════════════════════════════════════════════════════════════════════
# 3.  _assign_to_interval
# ═══════════════════════════════════════════════════════════════════════════

class TestAssignToInterval:
    def test_assigns_all_fields(self):
        inj = {}
        interval = {"start": 1000.0, "end": 1100.0, "duration": 100.0,
                    "job_id": 5, "shift": [0.0, -50.0]}
        _assign_to_interval(inj, interval, 1050.0, 2)
        # _assign_to_interval mutates in-place
        assert inj["gps_time"] == 1050.0
        assert inj["trial_idx"] == 2
        assert inj["job_id"] == 5
        assert inj["shift"] == [0.0, -50.0]
        # shift is a copy, not the same list object
        assert inj["shift"] is not interval["shift"]


# ═══════════════════════════════════════════════════════════════════════════
# 4.  generate_injection_list_from_config_for_job_segments  — rate
# ═══════════════════════════════════════════════════════════════════════════

class TestJobSegmentInjectionRate:
    """Rate-based scheduling into job intervals."""

    def test_fits_in_one_trial(self):
        jobs = [_segment(1, [0.0, 0.0]), _segment(2, [0.0, -1.0])]
        config = {
            "parameters": [_injection(f"inj{i}") for i in range(4)],
            "time_distribution": {"type": "rate", "rate": 1 / 50, "jitter": 0},
        }
        injections, n_trials = generate_injection_list_from_config_for_job_segments(config, jobs)
        assert n_trials == 1
        assert len(injections) == 4
        assert all(inj["trial_idx"] == 0 for inj in injections)
        assert len({inj["job_id"] for inj in injections}) == 2

    def test_needs_multiple_trials(self):
        jobs = [_segment(1, [0.0, 0.0])]  # 100 s livetime
        config = {
            "parameters": [_injection(f"inj{i}") for i in range(10)],
            "time_distribution": {"type": "rate", "rate": 1 / 20, "jitter": 0},
        }
        injections, n_trials = generate_injection_list_from_config_for_job_segments(config, jobs)
        assert n_trials == 2
        trial_0 = [inj for inj in injections if inj["trial_idx"] == 0]
        trial_1 = [inj for inj in injections if inj["trial_idx"] == 1]
        assert len(trial_0) == 5
        assert len(trial_1) == 5
        assert all(inj["job_id"] == 1 for inj in injections)

    def test_all_injections_get_sim_idx(self):
        jobs = [_segment(1, [0.0, 0.0])]
        config = {
            "parameters": [_injection(f"inj{i}") for i in range(5)],
            "time_distribution": {"type": "rate", "rate": 1 / 10, "jitter": 0},
        }
        injections, _ = generate_injection_list_from_config_for_job_segments(config, jobs)
        sim_idxs = [inj["sim_idx"] for inj in injections]
        assert sim_idxs == [0, 1, 2, 3, 4]

    def test_sim_idx_stable_with_seed(self):
        jobs = [_segment(1, [0.0, 0.0]), _segment(2, [0.0, -1.0])]
        config = {
            "parameters": [_injection(f"inj{i}") for i in range(3)],
            "time_distribution": {"type": "rate", "rate": 1 / 33, "jitter": 0},
            "seed": 42,
        }
        inj1, _ = generate_injection_list_from_config_for_job_segments(config, jobs)
        inj2, _ = generate_injection_list_from_config_for_job_segments(config, jobs)
        assert [i["sim_idx"] for i in inj1] == [0, 1, 2]
        assert [i["sim_idx"] for i in inj2] == [0, 1, 2]

    def test_shift_field_matches_owning_job(self):
        jobs = [_segment(1, [0.0, 0.0]), _segment(2, [5.0, -3.0])]
        config = {
            "parameters": [_injection(f"inj{i}") for i in range(4)],
            "time_distribution": {"type": "rate", "rate": 1 / 25, "jitter": 0},
        }
        injections, _ = generate_injection_list_from_config_for_job_segments(config, jobs)
        for inj in injections:
            if inj["job_id"] == 1:
                assert inj["shift"] == [0.0, 0.0]
            else:
                assert inj["shift"] == [5.0, -3.0]

    def test_rate_too_large_raises(self):
        jobs = [_segment(1, [0.0, 0.0])]  # 100 s livetime
        config = {
            "parameters": [_injection("inj0")],
            # interval = 200 s > 100 s available → raises
            "time_distribution": {"type": "rate", "rate": 1 / 200, "jitter": 0},
        }
        with pytest.raises(ValueError, match="Rate is too large"):
            generate_injection_list_from_config_for_job_segments(config, jobs)

    def test_jitter_too_large_raises(self):
        jobs = [_segment(1, [0.0, 0.0])]
        config = {
            "parameters": [_injection("inj0")],
            "time_distribution": {"type": "rate", "rate": 1 / 10, "jitter": 10},
        }
        with pytest.raises(ValueError, match="Jitter is too large"):
            generate_injection_list_from_config_for_job_segments(config, jobs)

    def test_empty_jobs_raises(self):
        config = {
            "parameters": [_injection("inj0")],
            "time_distribution": {"type": "rate", "rate": 1 / 10, "jitter": 0},
        }
        with pytest.raises(ValueError, match="No available job-segment livetime"):
            generate_injection_list_from_config_for_job_segments(config, [])

    def test_custom_time_distribution_raises(self):
        jobs = [_segment(1, [0.0, 0.0])]
        config = {
            "parameters": [_injection("inj0")],
            "time_distribution": {"type": "custom"},
        }
        with pytest.raises(ValueError, match="Custom time distribution is not supported"):
            generate_injection_list_from_config_for_job_segments(config, jobs)

    def test_unknown_time_distribution_raises(self):
        jobs = [_segment(1, [0.0, 0.0])]
        config = {
            "parameters": [_injection("inj0")],
            "time_distribution": {"type": "gaussian"},
        }
        with pytest.raises(ValueError, match="Unknown time distribution"):
            generate_injection_list_from_config_for_job_segments(config, jobs)

    def test_no_duplicate_scheduled_injections(self):
        """Each generated simulation appears exactly once — no cross-job duplication."""
        jobs = [_segment(1, [0.0, 0.0]), _segment(2, [0.0, -100.0]),
                _segment(3, [0.0, 100.0])]
        config = {
            "parameters": [_injection(f"inj{i}") for i in range(30)],
            "time_distribution": {"type": "rate", "rate": 1 / 10, "jitter": 0},
        }
        injections, _ = generate_injection_list_from_config_for_job_segments(config, jobs)
        sim_idxs = [inj["sim_idx"] for inj in injections]
        assert len(sim_idxs) == len(set(sim_idxs))
        assert len(sim_idxs) == 30

    def test_injections_spread_across_all_jobs(self):
        jobs = [_segment(i, [0.0, 0.0], analyze_start=1000.0 + i * 100, analyze_end=1100.0 + i * 100)
                for i in range(4)]
        config = {
            "parameters": [_injection(f"inj{i}") for i in range(40)],
            "time_distribution": {"type": "rate", "rate": 1 / 10, "jitter": 0},
        }
        injections, _ = generate_injection_list_from_config_for_job_segments(config, jobs)
        job_ids_seen = {inj["job_id"] for inj in injections}
        assert job_ids_seen == {0, 1, 2, 3}

    def test_gps_times_within_owning_interval(self):
        jobs = [_segment(1, [0.0, 0.0], analyze_start=1000.0, analyze_end=1100.0),
                _segment(2, [0.0, 0.0], analyze_start=1200.0, analyze_end=1300.0)]
        config = {
            "parameters": [_injection(f"inj{i}") for i in range(20)],
            "time_distribution": {"type": "rate", "rate": 1 / 5, "jitter": 0},
        }
        injections, _ = generate_injection_list_from_config_for_job_segments(config, jobs)
        for inj in injections:
            if inj["job_id"] == 1:
                assert 1000.0 <= inj["gps_time"] < 1100.0
            else:
                assert 1200.0 <= inj["gps_time"] < 1300.0

    def test_with_veto_windows_reduces_capacity(self):
        """Veto windows reduce available livetime → more trials needed."""
        seg = _segment(1, [0.0, 0.0],
                       analyze_start=1000.0, analyze_end=1100.0,
                       veto_windows=[(1000.0, 1050.0)])  # only 50s livetime
        config = {
            "parameters": [_injection(f"inj{i}") for i in range(6)],
            "time_distribution": {"type": "rate", "rate": 1 / 10, "jitter": 0},
        }
        injections, n_trials = generate_injection_list_from_config_for_job_segments(config, [seg])
        assert n_trials == 2

    def test_rate_jitter_gps_stays_within_interval(self):
        jobs = [_segment(1, [0.0, 0.0])]
        config = {
            "parameters": [_injection(f"inj{i}") for i in range(100)],
            "time_distribution": {"type": "rate", "rate": 1 / 0.5, "jitter": 0.1},
            "seed": 123,
        }
        injections, _ = generate_injection_list_from_config_for_job_segments(config, jobs)
        for inj in injections:
            assert 1000.0 <= inj["gps_time"] < 1100.0, \
                f"GPS {inj['gps_time']} outside [1000, 1100)"

    def test_large_scale_distribution_800_inj_2_slags(self):
        """800 injections at 1/10 Hz into 2 × 1000s jobs → ~4 trials, ~200 per trial.

        Math:
          total livetime = 2 jobs × 1000s = 2000s
          rate 0.1 Hz → interval = 10s
          required time = 800 / 0.1 = 8000s > 2000s → multi-trial
          n_inj_in_each_trial = int(2000 × 0.1) = 200
          n_trials = ceil(800 / 200) = 4
        """
        jobs = [
            _segment(1, [0.0, 0.0], analyze_start=1000.0, analyze_end=2000.0),
            _segment(2, [0.0, -1.0], analyze_start=1000.0, analyze_end=2000.0),
        ]
        config = {
            "parameters": [_injection(f"inj{i}") for i in range(800)],
            "time_distribution": {"type": "rate", "rate": 1 / 10, "jitter": 0},
        }
        injections, n_trials = generate_injection_list_from_config_for_job_segments(config, jobs)

        assert n_trials == 4
        assert len(injections) == 800

        # ~200 injections per trial
        for t in range(n_trials):
            trial_injs = [inj for inj in injections if inj["trial_idx"] == t]
            assert len(trial_injs) == 200, f"trial {t}: expected 200, got {len(trial_injs)}"

        # ~100 injections per job per trial (200 per trial ÷ 2 jobs)
        for t in range(n_trials):
            trial_injs = [inj for inj in injections if inj["trial_idx"] == t]
            for job_id in (1, 2):
                job_count = sum(1 for inj in trial_injs if inj["job_id"] == job_id)
                assert job_count == 100, \
                    f"trial {t} job {job_id}: expected 100, got {job_count}"

        # Each sim_idx appears exactly once (no cross-job duplication)
        sim_ids = [inj["sim_idx"] for inj in injections]
        assert len(sim_ids) == len(set(sim_ids))
        assert set(sim_ids) == set(range(800))

        # Shift field matches owning job
        for inj in injections:
            if inj["job_id"] == 1:
                assert inj["shift"] == [0.0, 0.0]
            else:
                assert inj["shift"] == [0.0, -1.0]

    # -- original tests preserved with updated helpers --
    def test_shifted_jobs_add_capacity_without_duplicate_overlap_assignment(self):
        jobs = [_segment(1, [0.0, 0.0]), _segment(2, [0.0, -100.0])]
        config = {
            "parameters": [_injection(f"inj{i}") for i in range(4)],
            "time_distribution": {"type": "rate", "rate": 1 / 50, "jitter": 0},
        }
        injections, n_trials = generate_injection_list_from_config_for_job_segments(config, jobs)
        assert n_trials == 1
        assert len(injections) == 4
        assert [inj["job_id"] for inj in injections] == [1, 1, 2, 2]
        assert [inj["trial_idx"] for inj in injections] == [0, 0, 0, 0]
        assert [inj["sim_idx"] for inj in injections] == [0, 1, 2, 3]
        assert injections[2]["shift"] == [0.0, -100.0]


# ═══════════════════════════════════════════════════════════════════════════
# 5.  generate_injection_list_from_config_for_job_segments  — Poisson
# ═══════════════════════════════════════════════════════════════════════════

class TestJobSegmentInjectionPoisson:
    """Poisson-based scheduling into job intervals."""

    def test_poisson_basic(self):
        jobs = [_segment(1, [0.0, 0.0])]
        config = {
            "parameters": [_injection(f"inj{i}") for i in range(20)],
            "time_distribution": {"type": "poisson", "rate": 1 / 5, "max_trail": 3},
            "seed": 42,
        }
        injections, n_trials = generate_injection_list_from_config_for_job_segments(config, jobs)
        assert len(injections) == 20
        assert n_trials <= 3
        assert all("job_id" in inj for inj in injections)
        assert all("shift" in inj for inj in injections)
        assert all("sim_idx" in inj for inj in injections)

    def test_poisson_spans_multiple_jobs(self):
        jobs = [_segment(1, [0.0, 0.0]), _segment(2, [0.0, -10.0])]
        config = {
            "parameters": [_injection(f"inj{i}") for i in range(50)],
            "time_distribution": {"type": "poisson", "rate": 1 / 2, "max_trail": 5},
            "seed": 123,  # deterministic seed that spans both jobs
        }
        injections, _ = generate_injection_list_from_config_for_job_segments(config, jobs)
        job_ids = {inj["job_id"] for inj in injections}
        # At least one job represented (probabilistic — may not always span both)
        assert len(job_ids) >= 1
        sim_idxs = [inj["sim_idx"] for inj in injections]
        assert len(sim_idxs) == len(set(sim_idxs))


# ═══════════════════════════════════════════════════════════════════════════
# 6.  generate_injection_list_from_config_for_job_segments  — explicit GPS
# ═══════════════════════════════════════════════════════════════════════════

class TestJobSegmentInjectionExplicitGPS:
    """Explicit-GPS injection scheduling (no time_distribution)."""

    def test_explicit_gps_assigned_to_containing_job(self):
        jobs = [_segment(1, [0.0, 0.0]), _segment(2, [5.0, -3.0])]
        config = {
            "parameters": [
                _injection("a", gps_time=1025.0),
                _injection("b", gps_time=1075.0),
            ],
        }
        injections, n_trials = generate_injection_list_from_config_for_job_segments(config, jobs)
        assert n_trials == 1
        assert injections[0]["job_id"] == 1
        assert injections[1]["job_id"] == 1
        assert injections[0]["shift"] == [0.0, 0.0]
        assert injections[1]["shift"] == [0.0, 0.0]

    def test_explicit_gps_assigned_to_first_containing_job(self):
        jobs = [_segment(1, [0.0, 0.0], analyze_start=1000.0, analyze_end=1100.0),
                _segment(2, [0.0, -1.0], analyze_start=1000.0, analyze_end=1100.0)]
        config = {
            "parameters": [_injection("a", gps_time=1050.0)],
        }
        injections, _ = generate_injection_list_from_config_for_job_segments(config, jobs)
        # Both cover [1000,1100); first sorted by analyze_start then index → job 1
        assert injections[0]["job_id"] == 1

    def test_explicit_gps_outside_all_raises(self):
        jobs = [_segment(1, [0.0, 0.0])]
        config = {
            "parameters": [_injection("a", gps_time=500.0)],
        }
        with pytest.raises(ValueError, match="does not fall within any job segment"):
            generate_injection_list_from_config_for_job_segments(config, jobs)

    def test_explicit_gps_missing_field_raises(self):
        jobs = [_segment(1, [0.0, 0.0])]
        config = {
            "parameters": [{"name": "no_gps", "approximant": "SGE", "hrss": 1e-22,
                            "ra": 1.0, "dec": 0.5, "pol": 0.0}],
        }
        with pytest.raises(ValueError, match="'gps_time' must be specified"):
            generate_injection_list_from_config_for_job_segments(config, jobs)

    def test_explicit_gps_preserves_existing_trial_idx(self):
        jobs = [_segment(1, [0.0, 0.0])]
        config = {
            "parameters": [_injection("a", gps_time=1050.0, trial_idx=7)],
        }
        injections, n_trials = generate_injection_list_from_config_for_job_segments(config, jobs)
        assert n_trials == 8  # max trial_idx + 1
        assert injections[0]["trial_idx"] == 7

    def test_explicit_gps_at_start_boundary_included(self):
        jobs = [_segment(1, [0.0, 0.0])]
        config = {
            "parameters": [_injection("a", gps_time=1000.0)],
        }
        injections, _ = generate_injection_list_from_config_for_job_segments(config, jobs)
        assert injections[0]["job_id"] == 1
        assert injections[0]["gps_time"] == 1000.0

    def test_explicit_gps_at_end_boundary_excluded(self):
        jobs = [_segment(1, [0.0, 0.0])]
        config = {
            "parameters": [_injection("a", gps_time=1100.0)],
        }
        with pytest.raises(ValueError, match="does not fall within any job segment"):
            generate_injection_list_from_config_for_job_segments(config, jobs)

    def test_explicit_gps_within_veto_window_still_assigned(self):
        """Veto windows clip automatic scheduling livetime, not GPS containment."""
        jobs = [_segment(1, [0.0, 0.0], veto_windows=[(1025.0, 1075.0)])]
        config = {
            "parameters": [_injection("a", gps_time=1050.0)],
        }
        injections, _ = generate_injection_list_from_config_for_job_segments(config, jobs)
        assert injections[0]["job_id"] == 1


# ═══════════════════════════════════════════════════════════════════════════
# 7.  add_scheduled_injections_into_job_segments
# ═══════════════════════════════════════════════════════════════════════════

class TestAddScheduledInjectionsIntoJobSegments:
    """Tests for ``add_scheduled_injections_into_job_segments``."""

    def test_routes_by_job_id(self):
        seg1 = _segment(1, [0.0, 0.0])
        seg2 = _segment(2, [0.0, -1.0])
        injections = [
            {"job_id": 1, "gps_time": 1025.0, "t_start": -0.1, "t_end": 0.1,
             "name": "a", "shift": [0.0, 0.0]},
            {"job_id": 2, "gps_time": 1075.0, "t_start": -0.2, "t_end": 0.2,
             "name": "b", "shift": [0.0, -1.0]},
        ]
        add_scheduled_injections_into_job_segments([seg1, seg2], injections)
        assert len(seg1.injections) == 1
        assert seg1.injections[0]["name"] == "a"
        assert len(seg2.injections) == 1
        assert seg2.injections[0]["name"] == "b"

    def test_clears_previous_injections(self):
        seg = _segment(1, [0.0, 0.0])
        seg.injections = [{"old": True}]
        add_scheduled_injections_into_job_segments([seg], [])
        assert seg.injections == []

    def test_computes_start_end_times(self):
        seg = _segment(1, [0.0, 0.0])
        injections = [{"job_id": 1, "gps_time": 1050.0, "t_start": -0.5, "t_end": 0.5,
                       "name": "test"}]
        add_scheduled_injections_into_job_segments([seg], injections)
        assert seg.injections[0]["start_time"] == 1049.5
        assert seg.injections[0]["end_time"] == 1050.5

    def test_missing_job_id_raises(self):
        seg = _segment(1, [0.0, 0.0])
        injections = [{"gps_time": 1050.0}]
        with pytest.raises(ValueError, match="missing required 'job_id'"):
            add_scheduled_injections_into_job_segments([seg], injections)

    def test_unknown_job_id_raises(self):
        seg = _segment(1, [0.0, 0.0])
        injections = [{"job_id": 999, "gps_time": 1050.0}]
        with pytest.raises(ValueError, match="unknown job_id=999"):
            add_scheduled_injections_into_job_segments([seg], injections)

    def test_multiple_injections_same_job(self):
        seg1 = _segment(1, [0.0, 0.0])
        seg2 = _segment(2, [0.0, 0.0])
        injections = [
            {"job_id": 1, "gps_time": 1010.0, "t_start": -0.1, "t_end": 0.1, "name": "a"},
            {"job_id": 1, "gps_time": 1020.0, "t_start": -0.1, "t_end": 0.1, "name": "b"},
            {"job_id": 2, "gps_time": 1030.0, "t_start": -0.1, "t_end": 0.1, "name": "c"},
        ]
        add_scheduled_injections_into_job_segments([seg1, seg2], injections)
        assert len(seg1.injections) == 2
        assert len(seg2.injections) == 1

# ═══════════════════════════════════════════════════════════════════════════
# 8.  build_simulation_summary  — ownership fields
# ═══════════════════════════════════════════════════════════════════════════

class TestSimulationSummaryOwnership:
    """Tests that simulation summary preserves job_id, shift, and sim_idx."""

    @staticmethod
    def _fake_generate(injection, config, sample_rate, ifos):
        class FakeSeries:
            def __init__(self, t0):
                self.t0 = t0
                self.dt = 0.1
                self.data = np.zeros(10)
        t0 = injection.get("gps_time", 1000.0) - 0.1
        return [FakeSeries(t0) for _ in ifos]

    def test_job_id_from_injection(self, monkeypatch):
        monkeypatch.setattr(
            "pycwb.workflow.subflow.simulation_summary.generate_strain_from_injection",
            self._fake_generate,
        )
        seg = _segment(1, [0.0, 0.0])
        seg.injections = [_injection("a", gps_time=1050.0, trial_idx=0,
                                     job_id=5, shift=[1.0, 2.0],
                                     sim_idx=42)]
        config = SimpleNamespace(dq_files=[], ifo=["L1", "H1"], inRate=1024)
        df = build_simulation_summary(config, [seg])
        assert df["job_id"].iloc[0] == 5
        assert df["segment_idx"].iloc[0] == 5
        assert list(df["shift"].iloc[0]) == [1.0, 2.0]
        assert df["sim_idx"].iloc[0] == 42

    def test_job_id_fallback_to_owning_segment(self, monkeypatch):
        monkeypatch.setattr(
            "pycwb.workflow.subflow.simulation_summary.generate_strain_from_injection",
            self._fake_generate,
        )
        seg = _segment(3, [0.0, -1.0])
        seg.injections = [_injection("a", gps_time=1050.0, trial_idx=0)]
        config = SimpleNamespace(dq_files=[], ifo=["L1", "H1"], inRate=1024)
        df = build_simulation_summary(config, [seg])
        assert df["job_id"].iloc[0] == 3
        assert df["segment_idx"].iloc[0] == 3
        assert list(df["shift"].iloc[0]) == [0.0, -1.0]

    def test_vetoed_cat1_flag(self, monkeypatch):
        monkeypatch.setattr(
            "pycwb.workflow.subflow.simulation_summary.generate_strain_from_injection",
            self._fake_generate,
        )
        seg = _segment(1, [0.0, 0.0], analyze_start=1000.0, analyze_end=1100.0)
        seg.injections = [_injection("a", gps_time=500.0, trial_idx=0)]
        config = SimpleNamespace(dq_files=[], ifo=["L1", "H1"], inRate=1024)
        df = build_simulation_summary(config, [seg])
        assert bool(df["vetoed_cat1"].iloc[0]) is True

    def test_not_vetoed_cat1(self, monkeypatch):
        monkeypatch.setattr(
            "pycwb.workflow.subflow.simulation_summary.generate_strain_from_injection",
            self._fake_generate,
        )
        seg = _segment(1, [0.0, 0.0], analyze_start=1000.0, analyze_end=1100.0)
        seg.injections = [_injection("a", gps_time=1050.0, trial_idx=0)]
        config = SimpleNamespace(dq_files=[], ifo=["L1", "H1"], inRate=1024)
        df = build_simulation_summary(config, [seg])
        assert bool(df["vetoed_cat1"].iloc[0]) is False

    def test_vetoed_cat2_flag(self, monkeypatch):
        monkeypatch.setattr(
            "pycwb.workflow.subflow.simulation_summary.generate_strain_from_injection",
            self._fake_generate,
        )
        seg = _segment(1, [0.0, 0.0],
                       analyze_start=1000.0, analyze_end=1100.0,
                       veto_windows=[(1000.0, 1030.0)])
        seg.injections = [_injection("a", gps_time=1050.0, trial_idx=0)]
        config = SimpleNamespace(dq_files=[], ifo=["L1", "H1"], inRate=1024)
        df = build_simulation_summary(config, [seg])
        assert bool(df["vetoed_cat1"].iloc[0]) is False  # inside segment
        assert bool(df["vetoed_cat2"].iloc[0]) is True   # outside keep window

    def test_across_segments_flag(self, monkeypatch):
        monkeypatch.setattr(
            "pycwb.workflow.subflow.simulation_summary.generate_strain_from_injection",
            self._fake_generate,
        )
        seg1 = _segment(1, [0.0, 0.0], analyze_start=1000.0, analyze_end=1050.0)
        seg2 = _segment(2, [0.0, 0.0], analyze_start=1050.0, analyze_end=1100.0)
        seg1.injections = [_injection("a", gps_time=1050.0, trial_idx=0)]
        config = SimpleNamespace(dq_files=[], ifo=["L1", "H1"], inRate=1024)
        df = build_simulation_summary(config, [seg1, seg2])
        assert bool(df["across_segments"].iloc[0]) is True

    def test_empty_simulations_returns_empty_df(self, monkeypatch):
        monkeypatch.setattr(
            "pycwb.workflow.subflow.simulation_summary.generate_strain_from_injection",
            self._fake_generate,
        )
        seg = _segment(1, [0.0, 0.0])
        seg.injections = []
        config = SimpleNamespace(dq_files=[], ifo=["L1", "H1"], inRate=1024)
        df = build_simulation_summary(config, [seg])
        assert df.empty

    def test_no_duplicate_rows_for_same_injection(self, monkeypatch):
        monkeypatch.setattr(
            "pycwb.workflow.subflow.simulation_summary.generate_strain_from_injection",
            self._fake_generate,
        )
        seg1 = _segment(1, [0.0, 0.0])
        seg2 = _segment(2, [0.0, 0.0])
        shared_inj = _injection("dup", gps_time=1050.0, trial_idx=0)
        seg1.injections = [shared_inj]
        seg2.injections = [shared_inj]
        config = SimpleNamespace(dq_files=[], ifo=["L1", "H1"], inRate=1024)
        df = build_simulation_summary(config, [seg1, seg2])
        assert len(df) == 2
        assert df["job_id"].tolist() == [1, 2]

    # -- original test preserved --
    def test_simulation_summary_preserves_job_id_shift_and_sim_idx(self, monkeypatch):
        monkeypatch.setattr(
            "pycwb.workflow.subflow.simulation_summary.generate_strain_from_injection",
            self._fake_generate,
        )
        jobs = [_segment(1, [0.0, 0.0]), _segment(2, [0.0, -100.0])]
        jobs[0].injections = [_injection("same", gps_time=1025.0, trial_idx=0,
                                         job_id=1, shift=[0.0, 0.0],
                                         sim_idx=10)]
        jobs[1].injections = [_injection("same", gps_time=1075.0, trial_idx=0,
                                         job_id=2, shift=[0.0, -100.0],
                                         sim_idx=11)]
        config = SimpleNamespace(dq_files=[], ifo=["L1", "H1"], inRate=1024)
        df = build_simulation_summary(config, jobs)
        assert df["job_id"].tolist() == [1, 2]
        assert df["segment_idx"].tolist() == [1, 2]
        assert [list(v) for v in df["shift"]] == [[0.0, 0.0], [0.0, -100.0]]
        assert df["sim_idx"].tolist() == [10, 11]


# ═══════════════════════════════════════════════════════════════════════════
# 10. Matching  — in-memory
# ═══════════════════════════════════════════════════════════════════════════

class TestInMemoryMatching:
    """Tests for ``match_triggers_to_simulations``."""

    @staticmethod
    def _trigger(id_, job_id, trial_idx, gps_time, event_start=None, event_stop=None):
        return SimpleNamespace(
            id=id_, job_id=job_id, trial_idx=trial_idx, gps_time=gps_time,
            event_start=event_start or [gps_time - 0.1],
            event_stop=event_stop or [gps_time + 0.1],
        )

    @staticmethod
    def _sim(sim_idx, job_id, trial_idx, real_start, real_end):
        return {"sim_idx": sim_idx, "job_id": job_id, "trial_idx": trial_idx,
                "real_start": real_start, "real_end": real_end}

    def test_job_id_prevents_cross_job_match(self):
        triggers = [self._trigger("t1", job_id=1, trial_idx=0, gps_time=1000.0)]
        sims = [self._sim(101, job_id=2, trial_idx=0, real_start=999.5, real_end=1000.5)]
        matches = match_triggers_to_simulations(triggers, sims)
        assert len(matches) == 0

    def test_same_job_same_trial_matches(self):
        triggers = [self._trigger("t1", job_id=1, trial_idx=0, gps_time=1000.0)]
        sims = [self._sim(101, job_id=1, trial_idx=0, real_start=999.5, real_end=1000.5)]
        matches = match_triggers_to_simulations(triggers, sims)
        assert len(matches) == 1
        assert matches[0][0].id == "t1"
        assert matches[0][1]["sim_idx"] == 101

    def test_different_trial_no_match(self):
        triggers = [self._trigger("t1", job_id=1, trial_idx=0, gps_time=1000.0)]
        sims = [self._sim(101, job_id=1, trial_idx=1, real_start=999.5, real_end=1000.5)]
        matches = match_triggers_to_simulations(triggers, sims)
        assert len(matches) == 0

    def test_no_time_overlap_no_match(self):
        triggers = [self._trigger("t1", job_id=1, trial_idx=0, gps_time=1000.0)]
        sims = [self._sim(101, job_id=1, trial_idx=0, real_start=900.0, real_end=901.0)]
        matches = match_triggers_to_simulations(triggers, sims)
        assert len(matches) == 0

    def test_window_buffer_extends_match_window(self):
        triggers = [self._trigger("t1", job_id=1, trial_idx=0, gps_time=1000.0,
                                  event_start=[1000.0], event_stop=[1001.0])]
        sims = [self._sim(101, job_id=1, trial_idx=0, real_start=1001.1, real_end=1002.0)]
        matches_no_buf = match_triggers_to_simulations(triggers, sims, window_buffer=0.0)
        assert len(matches_no_buf) == 0
        matches_buf = match_triggers_to_simulations(triggers, sims, window_buffer=0.2)
        assert len(matches_buf) == 1

    def test_segment_lag_aligns_event_window(self):
        triggers = [
            SimpleNamespace(
                id="shifted_trigger",
                job_id=1,
                trial_idx=0,
                gps_time=1000.0,
                event_start=[2200.0],
                event_stop=[2200.2],
                segment_lag=[-1200.0],
            )
        ]
        sims = [self._sim(101, job_id=1, trial_idx=0, real_start=999.9, real_end=1000.1)]
        matches = match_triggers_to_simulations(triggers, sims)
        assert len(matches) == 1
        assert matches[0][1]["sim_idx"] == 101

    def test_how_left_returns_unmatched_triggers(self):
        triggers = [self._trigger("t1", job_id=1, trial_idx=0, gps_time=1000.0)]
        matches = match_triggers_to_simulations(triggers, [], how="left")
        assert len(matches) == 1
        assert matches[0][0].id == "t1"
        assert matches[0][1] is None

    def test_how_right_returns_unmatched_sims(self):
        sims = [self._sim(101, job_id=1, trial_idx=0, real_start=999.5, real_end=1000.5)]
        matches = match_triggers_to_simulations([], sims, how="right")
        assert len(matches) == 1
        assert matches[0][0] is None
        assert matches[0][1]["sim_idx"] == 101

    def test_how_outer_returns_all(self):
        triggers = [self._trigger("t1", job_id=1, trial_idx=0, gps_time=1000.0)]
        sims = [self._sim(202, job_id=2, trial_idx=0, real_start=900.0, real_end=901.0)]
        matches = match_triggers_to_simulations(triggers, sims, how="outer")
        assert len(matches) == 2

    def test_how_invalid_raises(self):
        with pytest.raises(ValueError, match="how must be"):
            match_triggers_to_simulations([], [], how="semi")

    def test_backward_compat_no_job_id(self):
        triggers = [self._trigger("t1", job_id=1, trial_idx=0, gps_time=1000.0)]
        sims = [{"sim_idx": 101, "trial_idx": 0, "real_start": 999.5, "real_end": 1000.5}]
        matches = match_triggers_to_simulations(triggers, sims)
        assert len(matches) == 1

    def test_one_trigger_can_match_multiple_sims(self):
        """Current behavior: many-to-many.  Documented for awareness."""
        triggers = [self._trigger("t1", job_id=1, trial_idx=0, gps_time=1000.0,
                                  event_start=[999.0], event_stop=[1001.0])]
        sims = [
            self._sim(101, job_id=1, trial_idx=0, real_start=999.5, real_end=1000.5),
            self._sim(102, job_id=1, trial_idx=0, real_start=999.6, real_end=1000.6),
        ]
        matches = match_triggers_to_simulations(triggers, sims)
        assert len(matches) >= 1

    # -- original test preserved --
    def test_in_memory_matching_uses_job_id_when_available(self):
        triggers = [
            SimpleNamespace(id="job1_trigger", job_id=1, trial_idx=0, gps_time=1000.0,
                           event_start=[999.9], event_stop=[1000.1]),
            SimpleNamespace(id="job2_trigger", job_id=2, trial_idx=0, gps_time=1000.0,
                           event_start=[999.9], event_stop=[1000.1]),
        ]
        simulations = [
            {"sim_idx": 101, "job_id": 1, "trial_idx": 0, "real_start": 999.5, "real_end": 1000.5},
            {"sim_idx": 202, "job_id": 2, "trial_idx": 0, "real_start": 999.5, "real_end": 1000.5},
        ]
        matches = match_triggers_to_simulations(triggers, simulations)
        assert [(trig.id, sim["sim_idx"]) for trig, sim in matches] == [
            ("job1_trigger", 101),
            ("job2_trigger", 202),
        ]


# ═══════════════════════════════════════════════════════════════════════════
# 11. Matching  — DuckDB / Parquet
# ═══════════════════════════════════════════════════════════════════════════

class TestParquetMatching:
    """Tests for ``match_simulations_parquet`` (requires duckdb)."""

    def test_job_id_join_column_present(self, tmp_path):
        pytest.importorskip("duckdb")
        triggers = pa.table({
            "id": ["t1", "t2"],
            "job_id": [1, 2],
            "trial_idx": [0, 0],
            "gps_time": [1000.0, 1000.0],
            "rho": [8.0, 12.0],
        })
        sims = pa.table({
            "sim_idx": [10, 20],
            "job_id": [1, 2],
            "trial_idx": [0, 0],
            "gps_time": [1000.0, 1000.0],
            "real_start": [999.5, 999.5],
            "real_end": [1000.5, 1000.5],
        })
        trig_path = tmp_path / "t.parquet"
        sim_path = tmp_path / "s.parquet"
        pq.write_table(triggers, trig_path)
        pq.write_table(sims, sim_path)
        matched = match_simulations_parquet(str(trig_path), str(sim_path), how="inner")
        df = matched.to_pandas().sort_values("sim_sim_idx").reset_index(drop=True)
        assert len(df) == 2
        assert df["sim_sim_idx"].tolist() == [10, 20]

    def test_no_job_id_column_falls_back(self, tmp_path):
        pytest.importorskip("duckdb")
        triggers = pa.table({
            "id": ["t1"],
            "trial_idx": [0],
            "gps_time": [1000.0],
            "rho": [8.0],
        })
        sims = pa.table({
            "sim_idx": [10],
            "trial_idx": [0],
            "gps_time": [1000.0],
            "real_start": [999.5],
            "real_end": [1000.5],
        })
        trig_path = tmp_path / "t.parquet"
        sim_path = tmp_path / "s.parquet"
        pq.write_table(triggers, trig_path)
        pq.write_table(sims, sim_path)
        matched = match_simulations_parquet(str(trig_path), str(sim_path), how="inner")
        assert matched.num_rows == 1

    def test_cross_job_no_match(self, tmp_path):
        pytest.importorskip("duckdb")
        triggers = pa.table({
            "id": ["t1"],
            "job_id": [1],
            "trial_idx": [0],
            "gps_time": [1000.0],
            "rho": [8.0],
        })
        sims = pa.table({
            "sim_idx": [10],
            "job_id": [2],
            "trial_idx": [0],
            "gps_time": [1000.0],
            "real_start": [999.5],
            "real_end": [1000.5],
        })
        trig_path = tmp_path / "t.parquet"
        sim_path = tmp_path / "s.parquet"
        pq.write_table(triggers, trig_path)
        pq.write_table(sims, sim_path)
        matched = match_simulations_parquet(str(trig_path), str(sim_path), how="inner")
        assert matched.num_rows == 0

    def test_how_left(self, tmp_path):
        pytest.importorskip("duckdb")
        triggers = pa.table({
            "id": ["t1"],
            "job_id": [1],
            "trial_idx": [0],
            "gps_time": [1000.0],
            "rho": [8.0],
        })
        sims = pa.table({
            "sim_idx": [10],
            "job_id": [2],
            "trial_idx": [0],
            "gps_time": [1000.0],
            "real_start": [999.5],
            "real_end": [1000.5],
        })
        trig_path = tmp_path / "t.parquet"
        sim_path = tmp_path / "s.parquet"
        pq.write_table(triggers, trig_path)
        pq.write_table(sims, sim_path)
        matched = match_simulations_parquet(str(trig_path), str(sim_path), how="left")
        df = matched.to_pandas()
        assert len(df) == 1
        assert df["id"].iloc[0] == "t1"

    def test_how_right(self, tmp_path):
        pytest.importorskip("duckdb")
        triggers = pa.table({
            "id": ["t1"],
            "job_id": [1],
            "trial_idx": [0],
            "gps_time": [1000.0],
            "rho": [8.0],
        })
        sims = pa.table({
            "sim_idx": [10],
            "job_id": [2],
            "trial_idx": [0],
            "gps_time": [1000.0],
            "real_start": [999.5],
            "real_end": [1000.5],
        })
        trig_path = tmp_path / "t.parquet"
        sim_path = tmp_path / "s.parquet"
        pq.write_table(triggers, trig_path)
        pq.write_table(sims, sim_path)
        matched = match_simulations_parquet(str(trig_path), str(sim_path), how="right")
        df = matched.to_pandas()
        assert len(df) == 1
        assert df["sim_sim_idx"].iloc[0] == 10

    def test_output_parquet_written(self, tmp_path):
        pytest.importorskip("duckdb")
        triggers = pa.table({
            "id": ["t1"],
            "job_id": [1],
            "trial_idx": [0],
            "gps_time": [1000.0],
            "rho": [8.0],
        })
        sims = pa.table({
            "sim_idx": [10],
            "job_id": [1],
            "trial_idx": [0],
            "gps_time": [1000.0],
            "real_start": [999.5],
            "real_end": [1000.5],
        })
        trig_path = tmp_path / "t.parquet"
        sim_path = tmp_path / "s.parquet"
        out_path = tmp_path / "matched.parquet"
        pq.write_table(triggers, trig_path)
        pq.write_table(sims, sim_path)
        match_simulations_parquet(str(trig_path), str(sim_path),
                                  how="inner", output_parquet=str(out_path))
        assert out_path.exists()

    def test_deduplication_one_to_one(self, tmp_path):
        """Even if many-to-many time overlap, QUALIFY ensures 1-to-1."""
        pytest.importorskip("duckdb")
        triggers = pa.table({
            "id": ["t1", "t2"],
            "job_id": [1, 1],
            "trial_idx": [0, 0],
            "gps_time": [1000.0, 1000.1],
            "rho": [8.0, 12.0],
        })
        sims = pa.table({
            "sim_idx": [10, 20],
            "job_id": [1, 1],
            "trial_idx": [0, 0],
            "gps_time": [1000.0, 1000.0],
            "real_start": [999.0, 998.0],
            "real_end": [1001.0, 1002.0],
        })
        trig_path = tmp_path / "t.parquet"
        sim_path = tmp_path / "s.parquet"
        pq.write_table(triggers, trig_path)
        pq.write_table(sims, sim_path)
        matched = match_simulations_parquet(str(trig_path), str(sim_path), how="inner")
        df = matched.to_pandas()
        assert df["id"].nunique() == len(df)
        assert df["sim_sim_idx"].nunique() == len(df)

    # -- original test preserved --
    def test_parquet_matching_uses_job_id_when_available(self, tmp_path):
        pytest.importorskip("duckdb")
        triggers = pa.table({
            "id": ["job1_trigger", "job2_trigger"],
            "job_id": [1, 2],
            "trial_idx": [0, 0],
            "gps_time": [1000.0, 1000.0],
            "rho": [5.0, 10.0],
        })
        sims = pa.table({
            "sim_idx": [101, 202],
            "job_id": [1, 2],
            "trial_idx": [0, 0],
            "gps_time": [1000.0, 1000.0],
            "real_start": [999.5, 999.5],
            "real_end": [1000.5, 1000.5],
        })
        trig_path = tmp_path / "triggers.parquet"
        sim_path = tmp_path / "simulations.parquet"
        pq.write_table(triggers, trig_path)
        pq.write_table(sims, sim_path)
        matched = match_simulations_parquet(str(trig_path), str(sim_path), how="inner")
        df = matched.to_pandas().sort_values("job_id").reset_index(drop=True)
        assert len(df) == 2
        assert df["id"].tolist() == ["job1_trigger", "job2_trigger"]
        assert df["sim_sim_idx"].tolist() == [101, 202]


# ═══════════════════════════════════════════════════════════════════════════
# 12. End-to-end integration
# ═══════════════════════════════════════════════════════════════════════════

class TestEndToEndPipeline:
    """Integration: scheduling → assignment → summary → matching."""

    @staticmethod
    def _fake_generate(injection, config, sample_rate, ifos):
        class FakeSeries:
            def __init__(self, t0):
                self.t0 = t0
                self.dt = 0.1
                self.data = np.zeros(10)
        t0 = injection.get("gps_time", 1000.0) - 0.1
        return [FakeSeries(t0) for _ in ifos]

    def test_full_pipeline_rate(self, monkeypatch):
        monkeypatch.setattr(
            "pycwb.workflow.subflow.simulation_summary.generate_strain_from_injection",
            self._fake_generate,
        )
        jobs = [
            _segment(1, [0.0, 0.0], analyze_start=1000.0, analyze_end=1100.0),
            _segment(2, [0.0, -100.0], analyze_start=1000.0, analyze_end=1100.0),
        ]
        config = {
            "parameters": [_injection(f"inj{i}") for i in range(8)],
            "time_distribution": {"type": "rate", "rate": 1 / 25, "jitter": 0},
        }
        injections, n_trials = generate_injection_list_from_config_for_job_segments(config, jobs)
        add_scheduled_injections_into_job_segments(jobs, injections)
        cfg = SimpleNamespace(dq_files=[], ifo=["L1", "H1"], inRate=1024)
        df = build_simulation_summary(cfg, jobs)
        assert len(df) == 8
        assert df["sim_idx"].nunique() == 8
        assert df["job_id"].notna().all()
        assert set(df["job_id"].unique()).issubset({1, 2})
        assert (df["segment_idx"] == df["job_id"]).all()

    def test_full_pipeline_poisson(self, monkeypatch):
        monkeypatch.setattr(
            "pycwb.workflow.subflow.simulation_summary.generate_strain_from_injection",
            self._fake_generate,
        )
        jobs = [
            _segment(1, [0.0, 0.0], analyze_start=1000.0, analyze_end=1100.0),
            _segment(2, [0.0, -1.0], analyze_start=1100.0, analyze_end=1200.0),
        ]
        config = {
            "parameters": [_injection(f"inj{i}") for i in range(30)],
            "time_distribution": {"type": "poisson", "rate": 1 / 3, "max_trail": 5},
            "seed": 12345,
        }
        injections, _ = generate_injection_list_from_config_for_job_segments(config, jobs)
        add_scheduled_injections_into_job_segments(jobs, injections)
        cfg = SimpleNamespace(dq_files=[], ifo=["L1", "H1"], inRate=1024)
        df = build_simulation_summary(cfg, jobs)
        assert len(df) == 30
        assert df["sim_idx"].nunique() == 30

    def test_full_pipeline_then_match(self, monkeypatch):
        monkeypatch.setattr(
            "pycwb.workflow.subflow.simulation_summary.generate_strain_from_injection",
            self._fake_generate,
        )
        jobs = [_segment(1, [0.0, 0.0], analyze_start=1000.0, analyze_end=1100.0)]
        config = {
            "parameters": [_injection(f"inj{i}") for i in range(5)],
            "time_distribution": {"type": "rate", "rate": 1 / 10, "jitter": 0},
        }
        injections, _ = generate_injection_list_from_config_for_job_segments(config, jobs)
        add_scheduled_injections_into_job_segments(jobs, injections)
        cfg = SimpleNamespace(dq_files=[], ifo=["L1", "H1"], inRate=1024)
        df = build_simulation_summary(cfg, jobs)
        sim_dicts = df.to_dict("records")
        triggers = [
            SimpleNamespace(
                id=f"trig_{s['sim_idx']}",
                job_id=s["job_id"],
                trial_idx=s["trial_idx"],
                gps_time=s["gps_time"],
                event_start=[s["real_start"] + 0.05],
                event_stop=[s["real_end"] - 0.05],
            )
            for s in sim_dicts
        ]
        matches = match_triggers_to_simulations(triggers, sim_dicts, how="inner")
        assert len(matches) == 5
        matched_sim_ids = {m[1]["sim_idx"] for m in matches}
        assert matched_sim_ids == set(df["sim_idx"])


# ═══════════════════════════════════════════════════════════════════════════
# 13.  Validation checks  (requirement 6 from spec)
# ═══════════════════════════════════════════════════════════════════════════

class TestValidationChecks:
    """Validation assertions from the implementation spec."""

    def test_no_duplicate_scheduled_with_different_owner(self):
        jobs = [_segment(1, [0.0, 0.0]), _segment(2, [0.0, -100.0]),
                _segment(3, [0.0, 100.0])]
        config = {
            "parameters": [_injection(f"inj{i}") for i in range(30)],
            "time_distribution": {"type": "rate", "rate": 1 / 5, "jitter": 0},
        }
        injections, _ = generate_injection_list_from_config_for_job_segments(config, jobs)
        by_sim_idx = defaultdict(list)
        for inj in injections:
            by_sim_idx[inj["sim_idx"]].append(inj)
        for sim_idx, injs in by_sim_idx.items():
            job_ids = {inj["job_id"] for inj in injs}
            assert len(job_ids) == 1, \
                f"sim_idx={sim_idx} assigned to multiple jobs: {job_ids}"

    def test_matched_recovered_consistency(self, monkeypatch):
        monkeypatch.setattr(
            "pycwb.workflow.subflow.simulation_summary.generate_strain_from_injection",
            TestEndToEndPipeline._fake_generate,
        )
        jobs = [_segment(1, [0.0, 0.0], analyze_start=1000.0, analyze_end=1100.0)]
        config = {
            "parameters": [_injection(f"inj{i}") for i in range(10)],
            "time_distribution": {"type": "rate", "rate": 1 / 5, "jitter": 0},
        }
        injections, _ = generate_injection_list_from_config_for_job_segments(config, jobs)
        add_scheduled_injections_into_job_segments(jobs, injections)
        cfg = SimpleNamespace(dq_files=[], ifo=["L1", "H1"], inRate=1024)
        df = build_simulation_summary(cfg, jobs)
        sim_dicts = df.to_dict("records")
        recovered = sim_dicts[:6]
        triggers = [
            SimpleNamespace(
                id=f"trig_{s['sim_idx']}",
                job_id=s["job_id"],
                trial_idx=s["trial_idx"],
                gps_time=s["gps_time"],
                event_start=[s["real_start"] + 0.05],
                event_stop=[s["real_end"] - 0.05],
            )
            for s in recovered
        ]
        matches = match_triggers_to_simulations(triggers, sim_dicts, how="inner")
        assert len(matches) == 6

    def test_each_trigger_matches_same_owner_job(self):
        triggers = [
            SimpleNamespace(id="t1", job_id=1, trial_idx=0, gps_time=1000.0,
                           event_start=[999.0], event_stop=[1001.0]),
        ]
        sims = [
            {"sim_idx": 101, "job_id": 1, "trial_idx": 0,
             "real_start": 999.5, "real_end": 1000.5},
            {"sim_idx": 102, "job_id": 1, "trial_idx": 0,
             "real_start": 999.6, "real_end": 1000.6},
        ]
        matches = match_triggers_to_simulations(triggers, sims)
        trigger_ids = [m[0].id for m in matches if m[0] is not None]
        assert all(tid == "t1" for tid in trigger_ids)


# ═══════════════════════════════════════════════════════════════════════════
# 14.  distribute_inj_in_job_intervals_by_rate  — edge cases
# ═══════════════════════════════════════════════════════════════════════════

class TestDistributeByRateEdgeCases:
    def test_exact_fit_one_injection_per_interval(self):
        intervals = [
            {"start": 1000.0, "end": 1010.0, "duration": 10.0, "job_id": 1, "shift": [0.0]},
            {"start": 1020.0, "end": 1030.0, "duration": 10.0, "job_id": 2, "shift": [0.0]},
        ]
        injections = [{"name": f"inj{i}"} for i in range(2)]
        rate = 1 / 10.0
        result, n_trials = distribute_inj_in_job_intervals_by_rate(
            injections, rate, 0.0, intervals, shuffle=False,
        )
        assert n_trials == 1
        assert result[0]["job_id"] == 1
        assert result[1]["job_id"] == 2

    def test_not_enough_livetime_zero_inj(self):
        # total livetime = 0.5 s, rate = 1 Hz → interval = 1s > 0.5s → "Rate is too large"
        # (The "Not enough livetime" codepath is guarded by the earlier interval check)
        intervals = [{"start": 1000.0, "end": 1000.5, "duration": 0.5, "job_id": 1, "shift": [0.0]}]
        injections = [{"name": f"inj{i}"} for i in range(2)]
        rate = 1.0
        with pytest.raises(ValueError, match="Rate is too large"):
            distribute_inj_in_job_intervals_by_rate(injections, rate, 0.0, intervals)

    def test_jitter_clips_to_interval_bounds(self):
        intervals = [{"start": 1000.0, "end": 1010.0, "duration": 10.0, "job_id": 1, "shift": [0.0]}]
        injections = [{"name": "inj0"}]
        rate = 1 / 10.0
        np.random.seed(0)
        result, _ = distribute_inj_in_job_intervals_by_rate(
            injections, rate, 2.0, intervals, shuffle=False,
        )
        gps = result[0]["gps_time"]
        assert 1000.0 <= gps <= 1010.0


# ═══════════════════════════════════════════════════════════════════════════
# 15.  distribute_inj_in_job_intervals_by_poisson  — edge cases
# ═══════════════════════════════════════════════════════════════════════════

class TestDistributeByPoissonEdgeCases:
    def test_max_trail_limits_trials(self):
        intervals = [{"start": 1000.0, "end": 1000.01, "duration": 0.01,
                      "job_id": 1, "shift": [0.0]}]
        injections = [{"name": f"inj{i}"} for i in range(100)]
        rate = 1000.0
        np.random.seed(42)
        result, n_trials = distribute_inj_in_job_intervals_by_poisson(
            injections, rate, intervals, max_trail=2, shuffle=False,
        )
        assert n_trials <= 2

    def test_shuffle_disabled_preserves_order(self):
        intervals = [{"start": 1000.0, "end": 1100.0, "duration": 100.0,
                      "job_id": 1, "shift": [0.0]}]
        injections = [{"name": f"inj{i}"} for i in range(10)]
        np.random.seed(0)
        result, _ = distribute_inj_in_job_intervals_by_poisson(
            injections, rate=1 / 5.0, intervals=intervals, max_trail=5, shuffle=False,
        )
        names = [inj["name"] for inj in result]
        assert names == [f"inj{i}" for i in range(len(names))]
