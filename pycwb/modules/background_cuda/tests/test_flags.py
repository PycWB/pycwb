"""Call-time semantics of the ``PYCWB_GPU_*`` switch accessors."""

from __future__ import annotations

import pytest

from pycwb.modules.background_cuda import flags


def test_prefix_is_the_documented_namespace() -> None:
    assert flags.PREFIX == "PYCWB_GPU_"


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, False), ("1", True), ("0", False), ("true", False), ("", False), (" 1", False)],
)
def test_enabled_accepts_only_literal_one(monkeypatch: pytest.MonkeyPatch, value: str | None, expected: bool) -> None:
    if value is None:
        monkeypatch.delenv("PYCWB_GPU_DPF", raising=False)
    else:
        monkeypatch.setenv("PYCWB_GPU_DPF", value)
    assert flags.enabled("DPF") is expected


def test_enabled_is_read_at_call_time(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PYCWB_GPU_SUBNET", raising=False)
    assert not flags.enabled("SUBNET")
    monkeypatch.setenv("PYCWB_GPU_SUBNET", "1")
    assert flags.enabled("SUBNET")


def test_text_returns_raw_value_or_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PYCWB_GPU_READ_WORKERS", raising=False)
    assert flags.text("READ_WORKERS") is None
    assert flags.text("READ_WORKERS", "1") == "1"
    monkeypatch.setenv("PYCWB_GPU_READ_WORKERS", " 2 ")
    assert flags.text("READ_WORKERS", "1") == " 2 "


def test_integer_parses_and_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PYCWB_GPU_GC_INTERVAL", raising=False)
    assert flags.integer("GC_INTERVAL", 128) == 128
    monkeypatch.setenv("PYCWB_GPU_GC_INTERVAL", "64")
    assert flags.integer("GC_INTERVAL", 128) == 64
    monkeypatch.setenv("PYCWB_GPU_GC_INTERVAL", "-3")
    assert flags.integer("GC_INTERVAL", 128) == -3


@pytest.mark.parametrize("value", ["", "abc", "1.5", "0x10"])
def test_integer_rejects_non_integers(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("PYCWB_GPU_GC_INTERVAL", value)
    with pytest.raises(ValueError, match="PYCWB_GPU_GC_INTERVAL must be an integer"):
        flags.integer("GC_INTERVAL", 128)


def test_worker_count_default_and_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PYCWB_GPU_LAG_WORKERS", raising=False)
    assert flags.worker_count("LAG_WORKERS", maximum=6) == 1
    assert flags.worker_count("LAG_WORKERS", maximum=6, default=4) == 4
    for value in ("1", "3", "6"):
        monkeypatch.setenv("PYCWB_GPU_LAG_WORKERS", value)
        assert flags.worker_count("LAG_WORKERS", maximum=6) == int(value)


@pytest.mark.parametrize("value", ["0", "7", "-1", "100"])
def test_worker_count_raises_outside_bounds(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("PYCWB_GPU_LAG_WORKERS", value)
    with pytest.raises(ValueError, match="must be between 1 and 6"):
        flags.worker_count("LAG_WORKERS", maximum=6)


def test_worker_count_honours_custom_minimum(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYCWB_GPU_SETUP_WORKERS", "2")
    assert flags.worker_count("SETUP_WORKERS", maximum=3, minimum=2) == 2
    monkeypatch.setenv("PYCWB_GPU_SETUP_WORKERS", "1")
    with pytest.raises(ValueError, match="between 2 and 3"):
        flags.worker_count("SETUP_WORKERS", maximum=3, minimum=2)


def test_worker_count_propagates_parse_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYCWB_GPU_SETUP_WORKERS", "three")
    with pytest.raises(ValueError, match="must be an integer"):
        flags.worker_count("SETUP_WORKERS", maximum=3)
