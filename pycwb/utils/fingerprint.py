"""Exact structural fingerprints; scientific exclusions belong to callers."""

import dataclasses
import hashlib
import struct
from collections.abc import Callable
from typing import Any

import numpy as np

Fingerprint = dict[str, Any]
"""Flat ``path -> exact value`` mapping produced by :func:`leaves`."""


def leaves(
    value: Any, path: str = "root", *, exclude: Callable[[Any, str], bool] | None = None
) -> Fingerprint:
    """Flatten a stage result into exact, comparable leaves.

    Every dataclass field, list/tuple element and dict entry is visited. Arrays
    contribute dtype, shape and a SHA-256 of their contiguous bytes; floats
    contribute their IEEE-754 big-endian bit pattern, so ``-0.0`` and ``0.0``
    or two NaN payloads compare as different. The traversal is deterministic
    for a given structure, so two fingerprints are comparable with ``==``.

    Parameters
    ----------
    value
        Dataclass instance, ``np.ndarray``, list, tuple, dict, float, int,
        bool, str or ``None``, nested arbitrarily.
    path : str, optional
        Prefix for the leaf keys. Default is ``"root"``.

    Returns
    -------
    dict
        Leaf path to exact value. The optional ``exclude(instance, field_name)``
        callback omits explicitly identified diagnostic dataclass fields.

    Raises
    ------
    TypeError
        For object-dtype arrays (they need an explicit schema) and for any
        other type not listed above, so an unexpected type can never pass
        silently.
    """
    if dataclasses.is_dataclass(value):
        result: Fingerprint = {}
        for field in dataclasses.fields(value):
            # Native SkyMapStatistics records wall-clock durations here. This
            # explicit diagnostic exclusion does not cover scientific fields.
            if exclude is not None and exclude(value, field.name):
                continue
            result.update(
                leaves(getattr(value, field.name), path + "." + field.name, exclude=exclude)
            )
        return result
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("Object arrays need an explicit validation schema")
        return {
            path: (
                value.dtype.str,
                value.shape,
                hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
            )
        }
    if isinstance(value, (list, tuple)):
        result = {path + ".length": len(value)}
        for i, child in enumerate(value):
            result.update(leaves(child, f"{path}[{i}]", exclude=exclude))
        return result
    if isinstance(value, dict):
        result = {path + ".keys": tuple(sorted(value))}
        for key, child in value.items():
            result.update(leaves(child, f"{path}[{key!r}]", exclude=exclude))
        return result
    if isinstance(value, (float, np.floating)):
        return {path: struct.pack(">d", float(value)).hex()}
    if value is None or isinstance(value, (str, bool, int, np.integer, np.bool_)):
        return {path: value}
    raise TypeError(f"Uncovered stage value {path}: {type(value)}")
