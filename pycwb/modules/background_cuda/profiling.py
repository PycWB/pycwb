"""Opt-in bounded diagnostic profiles; their walltime is not a speed benchmark.

``gpu.profile_lags=start:stop`` enables ``cProfile`` around the labelled
stages of every lag in ``[start, stop)``. Profiling overhead is significant, so
a profiled run must never be quoted as a timing result.

Each process (the parent for ``"output"``, each lag worker for
``"analysis"``) keeps one profile per label in the module-level ``_profiles``
and re-enables the same profile for every sampled lag, so the dumped
``.pstats`` file accumulates across the whole sampled lag range on purpose:
per-lag files would be too short to show where a stage spends its time. The
file is rewritten after every sampled lag, so an interrupted run still leaves
the statistics collected so far.
"""

from __future__ import annotations

import cProfile
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from pycwb.constants.gpu_options import gpu_options

_profiles: dict[tuple[str, str, str], cProfile.Profile] = {}
"""Per-process ``label -> Profile``; accumulates across the sampled lag range by design (see module docstring)."""

PROFILE_DIRECTORY_NAME = "gpu_profiles"
"""Subdirectory of the working directory receiving ``<label>_<pid>.pstats`` files."""


@contextmanager
def span(
    lag: int, label: str, directory: str | Path, *, options=None
) -> Iterator[None]:
    """Profile the enclosed block when ``lag`` falls in ``gpu.profile_lags``.

    The switch is read at every call, so the parent and its spawned workers
    agree on the sampled range without any shared state.

    Parameters
    ----------
    lag : int
        Lag index of the enclosed work.
    label : str
        Stage label, for example ``"analysis"`` or ``"output"``; one profile
        per label per process.
    directory : str or os.PathLike
        Working directory; profiles land in ``<directory>/gpu_profiles``.

    Yields
    ------
    None
        The block runs profiled when sampled, unprofiled otherwise.

    Raises
    ------
    ValueError
        If ``gpu.profile_lags`` is set but is not ``start:stop`` with
        integers satisfying ``0 <= start < stop``.
    """
    requested = gpu_options(options).profile_lags
    if not requested:
        yield
        return
    start, stop = map(int, requested.split(":"))
    if not start <= lag < stop:
        yield
        return
    destination = Path(directory) / PROFILE_DIRECTORY_NAME
    destination.mkdir(exist_ok=True)
    profile = _profiles.setdefault(
        (str(destination.resolve()), label, requested), cProfile.Profile()
    )
    profile.enable()
    try:
        yield
    finally:
        profile.disable()
        profile.dump_stats(str(destination / f"{label}_{os.getpid()}.pstats"))
