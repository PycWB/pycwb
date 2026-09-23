"""Private function binding used to compose accelerated stages without patching.

The native processors call their collaborators through module globals. To swap
one collaborator for a GPU implementation *for one caller only*, ``specialize``
clones the caller's function object with a copied globals dictionary in which
the named entries are replaced. The production module object is never mutated,
so other callers, tests and the serial CPU path keep the native binding.

Every binding names a private production symbol. The bound names are pinned by
``tests/test_bindings.py`` so that a rename in the native module fails loudly
instead of silently restoring CPU execution.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import update_wrapper
from types import FunctionType
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def specialize(function: F, **bindings: Any) -> F:
    """Return a clone of ``function`` whose globals contain ``bindings``.

    Parameters
    ----------
    function : callable
        A plain Python function. Builtins, bound methods and Numba dispatchers
        are rejected because they have no ``__code__``/``__globals__`` pair.
    **bindings
        Global names to replace in the clone. Each name must already exist in
        the function's module globals, which catches misspelled hook names.

    Returns
    -------
    callable
        A new function object with the same code, defaults, closure and
        metadata as ``function`` and the replaced globals.

    Raises
    ------
    TypeError
        If ``function`` is not a plain Python function.
    KeyError
        If a binding names a global the function's module does not define.
    """
    if not isinstance(function, FunctionType):
        raise TypeError(f"specialize requires a plain Python function, got {type(function).__name__}")
    missing = [name for name in bindings if name not in function.__globals__]
    if missing:
        raise KeyError(f"{function.__module__}.{function.__qualname__} has no global(s) {sorted(missing)}")
    clone = FunctionType(
        function.__code__,
        dict(function.__globals__, **bindings),
        function.__name__,
        function.__defaults__,
        function.__closure__,
    )
    clone.__kwdefaults__ = function.__kwdefaults__
    return update_wrapper(clone, function)  # type: ignore[return-value]
