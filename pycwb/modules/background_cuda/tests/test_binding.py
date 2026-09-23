"""``specialize`` clones a function with replaced globals and never mutates its module."""

from __future__ import annotations

import pytest

from pycwb.modules.background_cuda import binding
from pycwb.modules.background_cuda.binding import specialize

HELPER_CONSTANT = 10


def _helper(x: int) -> int:
    return x + HELPER_CONSTANT


def _caller(x: int, scale: int = 2, *, offset: int = 1) -> int:
    """Doubles the helper result and adds an offset."""
    return _helper(x) * scale + offset


def test_specialize_rebinding_affects_clone_only() -> None:
    clone = specialize(_caller, _helper=lambda x: -x)
    assert clone(3) == -3 * 2 + 1
    assert _caller(3) == 13 * 2 + 1
    assert globals()["_helper"] is _helper


def test_specialize_returns_distinct_function_with_copied_globals() -> None:
    clone = specialize(_caller, HELPER_CONSTANT=0)
    assert clone is not _caller
    assert clone.__globals__ is not _caller.__globals__
    assert clone.__globals__["HELPER_CONSTANT"] == 0
    assert _caller.__globals__["HELPER_CONSTANT"] == 10
    # The helper still reads its own module globals, so the constant is untouched.
    assert clone(1) == _caller(1)


def test_specialize_preserves_metadata_and_defaults() -> None:
    clone = specialize(_caller, _helper=_helper)
    assert clone.__name__ == _caller.__name__
    assert clone.__qualname__ == _caller.__qualname__
    assert clone.__doc__ == _caller.__doc__
    assert clone.__module__ == _caller.__module__
    assert clone.__defaults__ == _caller.__defaults__ == (2,)
    assert clone.__kwdefaults__ == _caller.__kwdefaults__ == {"offset": 1}
    assert clone.__wrapped__ is _caller
    assert clone(5) == _caller(5)
    assert clone(5, 3, offset=4) == _caller(5, 3, offset=4)


def test_specialize_rejects_unknown_global() -> None:
    with pytest.raises(KeyError, match="_missing_hook"):
        specialize(_caller, _missing_hook=object())


def test_specialize_reports_every_missing_name_sorted() -> None:
    with pytest.raises(KeyError, match=r"\['_a_missing', '_b_missing'\]"):
        specialize(_caller, _b_missing=1, _a_missing=2)


@pytest.mark.parametrize("target", [len, "text", 3, type("K", (), {"__call__": lambda self: None})()])
def test_specialize_rejects_non_functions(target: object) -> None:
    with pytest.raises(TypeError, match="plain Python function"):
        specialize(target)  # type: ignore[arg-type]


def test_specialize_rejects_bound_methods() -> None:
    class Owner:
        def method(self) -> None:
            return None

    with pytest.raises(TypeError):
        specialize(Owner().method)


def test_specialize_supports_closures() -> None:
    factor = 7

    def closure(x: int) -> int:
        return _helper(x) * factor

    clone = specialize(closure, _helper=lambda x: x)
    assert clone(2) == 14
    assert closure(2) == 12 * 7


def test_specialize_can_chain() -> None:
    once = specialize(_caller, _helper=lambda x: 100)
    twice = specialize(once, HELPER_CONSTANT=0)
    assert twice(0) == 201
    assert once.__globals__["HELPER_CONSTANT"] == 10
