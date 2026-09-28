"""Experimental device support cuts and stable bounded sparse compaction.

Requires the FP64 GPU :class:`alignment_jax.AlignmentSession`. No CPU pipeline
hooks. The explicit capacity is checked before returning any payload; overflow
is an error, never permission to discard selected pixels.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

DEFAULT_CAPACITY = 65536
"""Default number of sparse rows the fixed-shape selection emits."""


@partial(jax.jit, static_argnames=("start", "length", "ib", "ie", "edge", "capacity"))
def _select(
    combined: jax.Array,
    maps: jax.Array,
    shifts: jax.Array,
    eo: jax.Array,
    *,
    start: int,
    length: int,
    ib: int,
    ie: int,
    edge: int,
    capacity: int,
) -> tuple[jax.Array, ...]:
    """Apply the native support cuts to ``combined`` and compact ``capacity`` rows.

    Returns ``(count, frequency, time, total, energies, detector_indices)``
    with fixed shapes; rows beyond ``count`` are filler and must be dropped by
    the caller.
    """
    nf, nt = combined.shape
    f = jnp.arange(nf)[:, None]
    t = jnp.arange(nt)[None, :]

    # Rolls are safe because the native candidate domain excludes boundary bins.
    def at(df: int, dt: int) -> jax.Array:
        return jnp.roll(combined, (-df, -dt), axis=(0, 1))

    ct = at(1, 0) + at(0, 1) + at(1, 1)
    cb = at(-1, 0) + at(0, -1) + at(-1, -1)
    ht = at(1, 2) + jnp.where(f < nf - 2, at(2, 2) + at(2, 1), 0.0)
    hb = at(-1, -2) + jnp.where(f >= 2, at(-2, -2) + at(-2, -1), 0.0)
    em = 2 * eo
    eh = em * em
    reject = ((ct + cb) * combined < eh) & ((ct + ht) * combined < eh) & ((cb + hb) * combined < eh) & (combined < em)
    margin = max(edge, 2)
    mask = (f >= ib) & (f < min(max(ie, ib), nf - 1)) & (t >= margin) & (t < nt - margin) & ~(combined < eo) & ~reject
    count = jnp.sum(mask, dtype=jnp.int64)
    indices = jnp.nonzero(mask.ravel(), size=capacity, fill_value=0)[0]
    fi, ti = indices // nt, indices % nt
    energies = []
    det_indices = []
    total = jnp.zeros(capacity, dtype=jnp.float64)
    for d in range(maps.shape[0]):
        src = jnp.where(
            (ti >= start) & (ti < start + length) & (length > 0),
            start + (ti - start + shifts[d]) % max(1, length),
            ti,
        )
        raw = maps[d, fi, src]
        total = total + raw
        energies.append(jnp.where(raw > 0.0, raw, 0.0))
        det_indices.append(src * nf + fi)
    return (
        count,
        fi,
        ti,
        total,
        jnp.stack(energies, axis=1),
        jnp.stack(det_indices, axis=1),
    )


def select_pixels(
    session: Any,
    combined: jax.Array,
    shifts: np.ndarray,
    energy_threshold: float,
    ie: int,
    edge: int,
    capacity: int = DEFAULT_CAPACITY,
) -> tuple[np.ndarray, ...]:
    """Return the native-ordered sparse host payload for one lag (mask omitted).

    The caller may reconstruct the boolean mask from frequency/time. Live-mask
    handling stays with the alignment caller. This interface deliberately does
    not claim to be a drop-in ``select_network_pixels`` implementation.

    Parameters
    ----------
    session : AlignmentSession
        Provides the resident FP64 ``maps`` ``(n_detectors, n_frequency, n_time)``
        and the ``start``/``length``/``ib`` bounds.
    combined : jax.Array
        Device-resident float64 support map for the same lag and session,
        shape ``(n_frequency, n_time)``.
    shifts : numpy.ndarray
        Integer per-detector time shifts, shape ``(n_detectors,)``.
    energy_threshold : float
        Pixel energy threshold ``Eo``.
    ie : int
        Highest frequency layer considered.
    edge : int
        Time-edge margin in bins.
    capacity : int, optional
        Fixed number of rows the compiled selection emits; a distinct value
        triggers a recompile.

    Returns
    -------
    tuple of numpy.ndarray
        ``(frequency, time, total, energies, detector_indices)`` in native
        ``(frequency, time)`` order, each truncated to the selected count and
        copied so the payload does not retain capacity-sized buffers.

    Raises
    ------
    ValueError
        If ``capacity`` is not a positive integer, ``combined`` does not match
        the session maps, or ``shifts`` has the wrong shape or dtype.
    OverflowError
        If more pixels pass than ``capacity``; nothing is returned.
    """
    if not isinstance(capacity, int) or capacity < 1:
        raise ValueError("capacity must be a positive integer")
    if combined.shape != session.shape[1:] or combined.dtype != np.float64:
        raise ValueError("combined must match session FP64 map shape")
    shifts = np.asarray(shifts)
    if shifts.shape != (session.shape[0],) or shifts.dtype.kind not in "iu":
        raise ValueError("invalid detector shifts")
    result = _select(
        combined,
        session.maps,
        jax.device_put(shifts.astype(np.int64) % max(1, session.length), session.device),
        jax.device_put(np.float64(energy_threshold), session.device),
        start=session.start,
        length=session.length,
        ib=session.ib,
        ie=int(ie),
        edge=int(edge),
        capacity=capacity,
    )
    # Fixed output shapes avoid recompiling slices for every observed count.
    host = jax.device_get(result)
    count = int(host[0])
    if count > capacity:
        raise OverflowError(f"{count} selected pixels exceed capacity {capacity}; retry with a larger capacity")
    # Copy occupied rows so the payload does not retain capacity-sized buffers.
    return tuple(np.array(x[:count], copy=True) for x in host[1:])
