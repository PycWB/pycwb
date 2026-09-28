"""Exact-leaf fingerprinting and paired stage checks of ``validation``."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from pycwb.utils.function_binding import specialize
from pycwb.modules.background_cuda.validation import leaves, paired
from pycwb.modules.likelihoodWP.results import SkyMapStatistics


@dataclass
class Packet:
    pixels: np.ndarray
    status: int = 0
    note: str | None = None


def _get_likelihood_rejection_reason() -> str:
    return "first_cut"


def _rejected_likelihood() -> tuple[None, None]:
    _get_likelihood_rejection_reason()
    return None, None


def test_leaves_of_scalars_and_none() -> None:
    assert leaves(None) == {"root": None}
    assert leaves(3) == {"root": 3}
    assert leaves(True) == {"root": True}
    assert leaves("x") == {"root": "x"}
    assert leaves(np.int32(4)) == {"root": np.int32(4)}
    assert leaves(np.bool_(False)) == {"root": np.bool_(False)}


def test_leaves_of_floats_are_bit_patterns() -> None:
    assert leaves(1.0) == {"root": "3ff0000000000000"}
    assert leaves(np.float32(1.0)) == leaves(1.0)
    assert leaves(0.0) != leaves(-0.0)
    assert leaves(float("nan")) == leaves(float("nan"))


def test_leaves_of_arrays_record_dtype_shape_and_bits() -> None:
    values = leaves(np.arange(4, dtype=np.float32).reshape(2, 2))
    dtype, shape, digest = values["root"]
    assert dtype == "<f4"
    assert shape == (2, 2)
    assert len(digest) == 64
    assert leaves(np.array([0.0])) != leaves(np.array([-0.0]))
    assert leaves(np.array([1, 2])) != leaves(np.array([2, 1]))
    assert leaves(np.array([1, 2], np.int32)) != leaves(np.array([1, 2], np.int64))
    assert leaves(np.zeros((2, 3))) != leaves(np.zeros((3, 2)))


def test_leaves_rejects_object_arrays() -> None:
    with pytest.raises(TypeError, match="Object arrays"):
        leaves(np.array([object()], dtype=object))


def test_leaves_rejects_uncovered_types() -> None:
    with pytest.raises(TypeError, match="Uncovered stage value"):
        leaves(object())
    with pytest.raises(TypeError, match=r"root\['inner'\]"):
        leaves({"inner": set()})


def test_leaves_of_lists_and_tuples_record_length_and_order() -> None:
    result = leaves([1, 2.0])
    assert result["root.length"] == 2
    assert result["root[0]"] == 1
    assert result["root[1]"] == leaves(2.0)["root"]
    assert leaves([1, 2]) != leaves([2, 1])
    assert leaves((1, 2)) == leaves([1, 2])
    assert leaves([]) == {"root.length": 0}


def test_leaves_of_dicts_record_sorted_keys() -> None:
    result = leaves({"b": None, "a": 1})
    assert result["root.keys"] == ("a", "b")
    assert result["root['a']"] == 1
    assert result["root['b']"] is None
    assert leaves({}) == {"root.keys": ()}
    assert leaves({"value": None}) != leaves({})


def test_leaves_of_dataclasses_cover_every_field() -> None:
    result = leaves(Packet(np.array([1, 2]), 1, "n"))
    assert set(result) == {"root.pixels", "root.status", "root.note"}
    assert result["root.status"] == 1
    assert result["root.note"] == "n"
    assert leaves(Packet(np.array([1, 2]), 0)) != leaves(Packet(np.array([1, 2]), 1))


def _sky_statistics(timing: float, l_max: int = 0) -> SkyMapStatistics:
    return SkyMapStatistics(
        l_max, *[np.ones(2) for _ in range(11)], stage_timings={"total": timing}
    )


def test_only_skymap_stage_timings_are_excluded() -> None:
    a, b = _sky_statistics(1.0), _sky_statistics(2.0)
    assert leaves(a) == leaves(b)
    assert not any(
        key.endswith("stage_timings") or "stage_timings" in key for key in leaves(a)
    )
    assert leaves(a) != leaves(_sky_statistics(1.0, l_max=1))
    # The exclusion is by type and field name, not by key name in general.
    assert leaves({"stage_timings": 1.0}) != leaves({"stage_timings": 2.0})


def test_paired_returns_candidate_result_when_equal() -> None:
    def stage(x: int) -> dict[str, int]:
        return {"value": x + 1}

    assert paired(stage, stage, "test")(1) == {"value": 2}


def test_paired_copies_mutable_argument_for_reference() -> None:
    def stage(packet: Packet) -> Packet:
        packet.pixels += 1
        return packet

    packet = Packet(np.array([1, 2]))
    result = paired(stage, stage, "test", 0)(packet)
    np.testing.assert_array_equal(result.pixels, [2, 3])
    assert result is packet


def test_paired_detects_membership_order_and_status_difference() -> None:
    def reference() -> Packet:
        return Packet(np.array([1, 2]), 0)

    for candidate in (
        lambda: Packet(np.array([2, 1]), 0),
        lambda: Packet(np.array([1, 2]), 1),
    ):
        with pytest.raises(AssertionError, match="native parity failed"):
            paired(candidate, reference, "test")()


def test_paired_detects_missing_none_field() -> None:
    with pytest.raises(AssertionError, match="native parity failed"):
        paired(dict, lambda: {"value": None}, "test")()


def test_paired_fingerprints_reference_before_candidate_runs() -> None:
    scratch = np.array([1.0])

    def candidate() -> np.ndarray:
        scratch[0] = 2.0
        return scratch

    with pytest.raises(AssertionError, match="native parity failed"):
        paired(candidate, lambda: scratch, "scratch")()


def test_paired_likelihood_detects_internal_hook_difference() -> None:
    candidate = specialize(
        _rejected_likelihood, _get_likelihood_rejection_reason=lambda: "different_cut"
    )
    with pytest.raises(AssertionError, match="internal parity failed"):
        paired(candidate, _rejected_likelihood, "likelihood")()


def test_paired_likelihood_passes_when_hooks_agree() -> None:
    candidate = specialize(
        _rejected_likelihood, _get_likelihood_rejection_reason=lambda: "first_cut"
    )
    assert paired(candidate, _rejected_likelihood, "likelihood")() == (None, None)


def test_paired_writes_failure_capture_when_requested(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(AssertionError):
        paired(
            lambda: {"v": 1},
            lambda: {"v": 2},
            "stage",
            options={"stage_failure_dir": str(tmp_path / "captures")},
        )()
    captured = list((tmp_path / "captures").glob("stage_*.pickle"))
    assert len(captured) == 1
