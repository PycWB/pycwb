"""Expensive, opt-in paired native stage checks; exclude from speed runs.

Every check here runs the native CPU implementation *and* the accelerated one
on the same inputs and compares complete fingerprints of the results, so a
validation run costs at least the sum of both. Its wall time must never be
quoted as a benchmark. The checks are enabled per stage by the
``gpu.validate_*`` switches read by the calling modules; this module
only reads ``gpu.stage_failure_dir`` (at call time, inside the paired
call) to decide whether to pickle a failing pair for inspection.
"""

from __future__ import annotations

from pycwb.utils.fingerprint import Fingerprint as Fingerprint
from pycwb.utils.fingerprint import leaves as _leaves

import copy
import logging
import os
import pickle
from collections.abc import Callable
from pathlib import Path
from typing import Any


from pycwb.modules.likelihoodWP.results import SkyMapStatistics

from pycwb.constants.gpu_options import gpu_options

logger = logging.getLogger(__name__)

LIKELIHOOD_HOOKS: dict[str, str] = {
    "_compute_dpf_regulator_scalar": "dpf",
    "_compute_dpf_regulator": "dpf",
    "_scan_sky": "sky_scan",
    "_compute_statistics_at_sky_position": "sky_statistics",
    "_get_likelihood_rejection_reason": "threshold_decision",
}
"""Native likelihood globals whose return values are traced during a paired ``"likelihood"`` check.

Keys are private function names looked up in the likelihood caller's globals;
only the names actually present are traced, so the CPU and GPU clones may bind
different subsets. Values are the stage labels reported in the trace and in
the parity log. The internal comparison is order-sensitive: the sequence of
``(label, fingerprint)`` records must be identical between the two runs.
"""

MAX_REPORTED_DIFFERENCES = 20
"""Differing leaf paths listed in a parity failure message; the full list is in the pickled failure."""



def leaves(value: Any, path: str = "root") -> Fingerprint:
    """Fingerprint scientific results, excluding measured sky-scan durations."""
    return _leaves(value, path, exclude=lambda obj, field: isinstance(obj, SkyMapStatistics) and field == "stage_timings")


def traced(
    function: Callable[..., Any],
    hooks: dict[str, str],
    records: list[tuple[str, Fingerprint]],
) -> Callable[..., Any]:
    """Return a clone of ``function`` that fingerprints the results of its hooked callees.

    Each hooked global is wrapped so its return value is fingerprinted with
    :func:`leaves` immediately, before the caller can reuse the scratch
    storage that native kernels hand back.

    Parameters
    ----------
    function : callable
        Plain Python function whose globals contain the hooked names.
    hooks : dict of str to str
        Global name to trace label, for example :data:`LIKELIHOOD_HOOKS`.
        Names absent from ``function.__globals__`` are skipped.
    records : list
        Receives ``(label, fingerprint)`` in call order; owned by the caller.

    Returns
    -------
    callable
        Private clone produced by :func:`~pycwb.utils.function_binding.specialize`.
    """
    from pycwb.utils.function_binding import specialize

    def record(callee: Callable[..., Any], label: str) -> Callable[..., Any]:
        def call(*args: Any, **kwargs: Any) -> Any:
            result = callee(*args, **kwargs)
            records.append((label, leaves(result)))
            return result

        return call

    from functools import partial

    keywords = {}
    args = ()
    if isinstance(function, partial):
        keywords = dict(function.keywords)
        args = function.args
        function = function.func
    callback_names = {
        "_compute_dpf_regulator_scalar": "scalar_regulator",
        "_scan_sky": "sky_scan",
    }
    for name, callback in callback_names.items():
        if name in hooks and callback in keywords:
            keywords[callback] = record(keywords[callback], hooks[name])
    clone = specialize(
        function,
        **{
            name: record(function.__globals__[name], label)
            for name, label in hooks.items()
            if name in function.__globals__
        },
    )
    return partial(clone, *args, **keywords) if args or keywords else clone


def paired(
    function: Callable[..., Any],
    reference: Callable[..., Any],
    stage: str,
    mutable_arg: int | None = None,
    *,
    options=None,
) -> Callable[..., Any]:
    """Wrap an accelerated stage so every call is checked against the native reference.

    The returned callable runs ``reference`` first on (copies of) the
    arguments, fingerprints the result before the accelerated call can reuse
    native scratch arrays, then runs ``function`` and compares. On success it
    returns the accelerated result so downstream stages consume exactly what a
    speed run would.

    Parameters
    ----------
    function : callable
        Accelerated implementation whose result is returned.
    reference : callable
        Native CPU implementation with the same signature.
    stage : str
        Stage label used in logs, error messages and failure file names.
        ``"coherence_single_lag"`` substitutes each setup's
        ``"_cpu_reference_setup"`` for the reference call;
        ``"likelihood"`` additionally traces :data:`LIKELIHOOD_HOOKS`.
    mutable_arg : int or None, optional
        Positional index of an argument both implementations mutate (for
        example the cluster list); the reference receives a deep copy so the
        two calls see independent objects.

    Returns
    -------
    callable
        ``check(*args, **kwargs)`` returning ``function``'s result.

    Notes
    -----
    Inside ``check``, ``gpu.stage_failure_dir`` is read at call time; if
    set, a failing pair is pickled to
    ``<dir>/<stage>_<pid>.pickle`` as ``(reference_snapshot, actual,
    differing_paths)`` before the ``AssertionError`` is raised. The snapshot is
    a deep copy taken before the accelerated call.
    """

    def check(*args: Any, **kwargs: Any) -> Any:
        cpu_args = list(args)
        if stage == "coherence_single_lag":
            cpu_args[0] = [s.get("_cpu_reference_setup", s) for s in args[0]]
        if mutable_arg is not None:
            cpu_args[mutable_arg] = copy.deepcopy(args[mutable_arg])
        # CPU and GPU see independent cluster objects and identical geometry.
        cpu_trace: list[tuple[str, Fingerprint]] = []
        gpu_trace: list[tuple[str, Fingerprint]] = []
        cpu, gpu = reference, function
        if stage in ("likelihood", "evaluate_cluster_likelihood"):
            cpu = traced(reference, LIKELIHOOD_HOOKS, cpu_trace)
            gpu = traced(function, LIKELIHOOD_HOOKS, gpu_trace)
        expected = cpu(*cpu_args, **kwargs)
        # Fingerprint before the second call, so reusable native scratch arrays
        # cannot overwrite the reference and hide a later difference.
        a = leaves(expected)
        capture = gpu_options(options).stage_failure_dir
        reference_snapshot = copy.deepcopy(expected) if capture else None
        actual = gpu(*args, **kwargs)
        b = leaves(actual)
        differing = [
            key
            for key in a.keys() | b.keys()
            if key not in a or key not in b or a[key] != b[key]
        ]
        if differing:
            if capture:
                directory = Path(capture)
                directory.mkdir(parents=True, exist_ok=True)
                with (directory / f"{stage}_{os.getpid()}.pickle").open("wb") as stream:
                    pickle.dump((reference_snapshot, actual, sorted(differing)), stream)
            raise AssertionError(
                f"{stage} native parity failed: {sorted(differing)[:MAX_REPORTED_DIFFERENCES]}"
            )
        if cpu_trace != gpu_trace:
            labels = [
                cpu_trace[i][0] if i < len(cpu_trace) else "extra_gpu_call"
                for i in range(max(len(cpu_trace), len(gpu_trace)))
                if i >= len(cpu_trace)
                or i >= len(gpu_trace)
                or cpu_trace[i] != gpu_trace[i]
            ]
            raise AssertionError(f"{stage} internal parity failed: {labels}")
        if stage in ("likelihood", "evaluate_cluster_likelihood"):
            logger.info(
                "GPU likelihood internal parity: calls=%s exact=1",
                ",".join(label for label, _ in cpu_trace),
            )
        logger.info("GPU stage parity: stage=%s leaves=%d exact=1", stage, len(a))
        return actual

    return check
