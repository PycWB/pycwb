"""``profiling.span`` honours the ``PYCWB_GPU_PROFILE_LAGS`` range and validates it."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from pycwb.modules.background_cuda import profiling
from pycwb.modules.background_cuda.profiling import span


def test_span_is_a_no_op_when_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    options = {}
    with span(3, "analysis", tmp_path, options=options):
        touched = True
    assert touched
    assert not (tmp_path / "gpu_profiles").exists()


def test_span_is_a_no_op_for_empty_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    options = {}
    with span(3, "analysis", tmp_path, options=options):
        pass
    assert not (tmp_path / "gpu_profiles").exists()


@pytest.mark.parametrize("value", ["5", "1:2:3", "a:b", ":", "1:"])
def test_span_rejects_malformed_ranges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    options = {"profile_lags": value}
    with pytest.raises(ValueError), span(0, "analysis", tmp_path, options=options):
        pass


@pytest.mark.parametrize("value", ["3:3", "5:2", "-1:4"])
def test_span_rejects_non_increasing_or_negative_ranges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    options = {"profile_lags": value}
    with (
        pytest.raises(ValueError, match="0 <= start < stop"),
        span(0, "analysis", tmp_path, options=options),
    ):
        pass


def test_span_skips_lags_outside_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    options = {"profile_lags": "2:4"}
    for lag in (0, 1, 4, 5):
        with span(lag, "analysis", tmp_path, options=options):
            pass
    assert not (tmp_path / "gpu_profiles").exists()


def test_span_profiles_lags_inside_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    options = {"profile_lags": "2:4"}
    monkeypatch.setattr(profiling, "_profiles", {})
    for lag in (2, 3):
        with span(lag, "unit_label", tmp_path, options=options):
            sum(range(100))
    destination = tmp_path / "gpu_profiles" / f"unit_label_{os.getpid()}.pstats"
    assert destination.is_file()
    assert destination.stat().st_size > 0
    assert len(profiling._profiles) == 1


def test_span_disables_profile_when_body_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    options = {"profile_lags": "0:1"}
    monkeypatch.setattr(profiling, "_profiles", {})
    with (
        pytest.raises(RuntimeError, match="boom"),
        span(0, "failing", tmp_path, options=options),
    ):
        raise RuntimeError("boom")
    assert (tmp_path / "gpu_profiles" / f"failing_{os.getpid()}.pstats").is_file()
    # A second span on the same label must be able to re-enable the profiler.
    with span(0, "failing", tmp_path, options=options):
        pass
