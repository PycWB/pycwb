"""Call-time accessors for the ``PYCWB_GPU_*`` environment switches.

Every switch of the experimental GPU processor is read here, at call time, so
that a spawned worker and its parent see the same environment and so that the
README table of switches has a single source of truth. Nothing in this module
is read at import time.
"""

from __future__ import annotations

import os

PREFIX = "PYCWB_GPU_"


def enabled(name: str) -> bool:
    """Return whether ``PYCWB_GPU_<name>`` is set to ``"1"``.

    Parameters
    ----------
    name : str
        Switch name without the ``PYCWB_GPU_`` prefix, for example ``"DPF"``.

    Returns
    -------
    bool
        ``True`` only for the literal value ``"1"``.
    """
    return os.environ.get(PREFIX + name) == "1"


def text(name: str, default: str | None = None) -> str | None:
    """Return the raw value of ``PYCWB_GPU_<name>`` or ``default`` when unset.

    Parameters
    ----------
    name : str
        Switch name without the ``PYCWB_GPU_`` prefix.
    default : str or None, optional
        Value returned when the variable is absent.

    Returns
    -------
    str or None
        The environment value, unmodified.
    """
    return os.environ.get(PREFIX + name, default)


def integer(name: str, default: int) -> int:
    """Return ``PYCWB_GPU_<name>`` parsed as an integer.

    Parameters
    ----------
    name : str
        Switch name without the ``PYCWB_GPU_`` prefix.
    default : int
        Value used when the variable is absent.

    Returns
    -------
    int
        Parsed value.

    Raises
    ------
    ValueError
        If the variable is present but is not a base-10 integer.
    """
    value = os.environ.get(PREFIX + name)
    if value is None:
        return int(default)
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"{PREFIX}{name} must be an integer, got {value!r}") from error


def worker_count(name: str, *, maximum: int, minimum: int = 1, default: int = 1) -> int:
    """Return a bounded worker count from ``PYCWB_GPU_<name>``.

    Parameters
    ----------
    name : str
        Switch name without the ``PYCWB_GPU_`` prefix, for example
        ``"SETUP_WORKERS"``.
    maximum : int
        Largest accepted count. Every bound in this module reflects a measured
        memory or concurrency limit; raise rather than clamp when exceeded.
    minimum : int, optional
        Smallest accepted count.
    default : int, optional
        Count used when the variable is absent.

    Returns
    -------
    int
        Validated worker count.

    Raises
    ------
    ValueError
        If the value is outside ``[minimum, maximum]`` or is not an integer.
    """
    value = integer(name, default)
    if not minimum <= value <= maximum:
        raise ValueError(f"{PREFIX}{name} must be between {minimum} and {maximum}, got {value}")
    return value
