"""Private event-method binding with bounded native detector geometry reuse.

``Event.output_py`` constructs a fresh ``Detector`` for every event in two
local ``from pycwb.types.detector import Detector`` statements. Building the
LAL geometry each time dominates the per-event output cost of a background
lag, so this module clones the method with those two imports removed and the
name ``Detector`` bound to :func:`cached_detector` in the clone's globals. All
native event arithmetic is the clone's own unchanged code; no production class
or module is patched.

Contract with ``Event.output_py``
---------------------------------
The AST rebinding in :func:`_bind_output` is deliberate and depends on exactly
these properties of ``pycwb.types.network_event.Event.output_py``:

1. Its source is retrievable with ``inspect.getsource`` (a plain ``def`` in a
   file, not generated or C-implemented).
2. It contains exactly two ``from pycwb.types.detector import Detector``
   statements, each importing the single name ``Detector`` without an alias.
3. Every use of ``Detector`` inside the method is a plain name lookup that
   accepts the ``Detector(name, geometry_model=...)`` call form.
4. Its module globals, defaults, closure, keyword defaults and annotations are
   sufficient to run the method body; the clone reuses them verbatim.

Failure on drift is loud and happens at import of this module: property 2
raises ``RuntimeError`` (wrong import shape or wrong import count) and
property 1 raises the ``inspect``/``ast`` errors themselves. Property 3 is
guarded at run time by ``PYCWB_GPU_VALIDATE_EVENT_GEOMETRY=1``, which runs the
native method on a deep copy and requires every event field to be bitwise
identical.
"""

from __future__ import annotations

import ast
import copy
import inspect
import logging
import textwrap
from collections.abc import Callable
from functools import lru_cache
from types import CodeType, FunctionType
from typing import Any

from pycwb.types.detector import Detector
from pycwb.types.network_event import Event

from . import flags

logger = logging.getLogger(__name__)

GEOMETRY_CACHE_SIZE = 8
"""Distinct ``(name, geometry_model)`` detectors kept; covers every detector of the supported networks."""

NATIVE_DETECTOR_IMPORTS = 2
"""Number of local ``Detector`` imports the native ``Event.output_py`` is required to contain."""


@lru_cache(maxsize=GEOMETRY_CACHE_SIZE)
def _geometry(name: str, geometry_model: str) -> Detector:
    """Construct and memoize the shared prototype for one detector geometry."""
    return Detector(name, geometry_model=geometry_model)


def cached_detector(name: str, *, geometry_model: str = "lal") -> Detector:
    """Return a private copy of a memoized ``Detector``.

    Parameters
    ----------
    name : str
        Detector name, for example ``"H1"``.
    geometry_model : str, optional
        Geometry model accepted by ``Detector``. Default is ``"lal"``.

    Returns
    -------
    Detector
        A deep copy, so callers keep the constructor's ownership semantics:
        independent arrays and objects that they may mutate freely.
    """
    # Independent arrays/objects preserve constructor ownership for callers.
    return copy.deepcopy(_geometry(name, geometry_model))


def _bind_output() -> Callable[..., Any]:
    """Clone ``Event.output_py`` with its local ``Detector`` imports bound to :func:`cached_detector`.

    Returns
    -------
    callable
        Unbound function with the native method's code (minus the two import
        statements), defaults, closure, keyword defaults and annotations, and
        a globals copy in which ``Detector`` is :func:`cached_detector`.

    Raises
    ------
    RuntimeError
        If the native method does not contain exactly
        ``NATIVE_DETECTOR_IMPORTS`` statements of the form
        ``from pycwb.types.detector import Detector``.
    """
    method = Event.output_py
    tree = ast.parse(textwrap.dedent(inspect.getsource(method)))

    class BindDetector(ast.NodeTransformer):
        count = 0

        def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.AST:
            if node.module == "pycwb.types.detector":
                if [(a.name, a.asname) for a in node.names] != [("Detector", None)]:
                    raise RuntimeError("Native detector import contract changed")
                self.count += 1
                return ast.copy_location(ast.Pass(), node)
            return node

    binding = BindDetector()
    tree = binding.visit(tree)
    if binding.count != NATIVE_DETECTOR_IMPORTS:
        raise RuntimeError("Expected exactly two native event detector imports")
    ast.fix_missing_locations(tree)
    # Keep native line numbers so tracebacks point at network_event.py.
    ast.increment_lineno(tree, method.__code__.co_firstlineno - 1)
    namespace = dict(method.__globals__, Detector=cached_detector)
    compiled = compile(tree, method.__code__.co_filename, "exec")
    code = next(item for item in compiled.co_consts if isinstance(item, CodeType) and item.co_name == method.__name__)
    bound = FunctionType(code, namespace, method.__name__, method.__defaults__, method.__closure__)
    bound.__kwdefaults__ = dict(method.__kwdefaults__ or {})
    bound.__annotations__ = dict(method.__annotations__)
    return bound


_output = _bind_output()


class CachedGeometryEvent(Event):
    """``Event`` whose ``output_py`` reuses memoized detector geometry.

    Instances are created by the GPU processor's private binding of the native
    ``Event`` name; every other attribute and method is inherited unchanged.
    """

    def output_py(
        self,
        job_segment: Any,
        cluster: Any,
        config: Any = None,
        *,
        lag_shifts: Any = None,
    ) -> Any:
        """Fill the event's output fields with cached detector geometry.

        Same signature and result as ``Event.output_py``. With
        ``PYCWB_GPU_VALIDATE_EVENT_GEOMETRY=1`` the native method also runs on
        deep copies of ``self`` and ``cluster`` and every field of the two
        events must match bitwise; this is the run-time guard for the AST
        binding contract described in the module docstring.

        Parameters
        ----------
        job_segment : WaveSegment
            Native job segment.
        cluster : Cluster
            Cluster of this event.
        config : Config or None, optional
            Native search configuration.
        lag_shifts : sequence of float or None, optional
            Per-detector lag shifts for this lag.

        Returns
        -------
        object
            Whatever the native ``Event.output_py`` returns.

        Raises
        ------
        AssertionError
            If validation is enabled and any event field differs from the
            native method's result.
        """
        expected = None
        if flags.enabled("VALIDATE_EVENT_GEOMETRY"):
            from .validation import leaves

            reference = copy.deepcopy(self)
            Event.output_py(reference, job_segment, copy.deepcopy(cluster), config, lag_shifts=lag_shifts)
            expected = leaves(reference)
        result = _output(self, job_segment, cluster, config, lag_shifts=lag_shifts)
        if expected is not None:
            if leaves(self) != expected:
                raise AssertionError("Cached geometry changed native event fields")
            logger.info("GPU event geometry parity: event=%s exact=1", self.hash_id)
        return result
