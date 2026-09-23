"""Fused GPU alignment with persistent FP64 maps; no global JAX configuration.

:class:`AlignmentSession` keeps the detector energy maps of one job segment on
the GPU and evaluates the native lag alignment (circular shift inside the
valid interval, veto, ``Eo``/``2 Eo`` clipping) for a batch of lags in one
jitted call.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

DEFAULT_SCRATCH_BYTES = 256 * 1024**2
"""Default bound on the aligned output plus live masks of one :meth:`AlignmentSession.align` batch."""


@partial(jax.jit, static_argnames=("start", "length", "ib"))
def _align(
    maps: jax.Array,
    shifts: jax.Array,
    veto: jax.Array,
    eo: jax.Array,
    *,
    start: int,
    length: int,
    ib: int,
) -> tuple[jax.Array, jax.Array]:
    """Return ``(value, live)`` for a ``(n_lag, n_detectors)`` batch of shifts.

    ``value`` is float64 ``(n_lag, n_frequency, n_time)``; ``live`` is boolean
    ``(n_lag, n_time)``.
    """
    t = jnp.arange(maps.shape[2], dtype=jnp.int64)
    live = jnp.broadcast_to((t >= start) & (t < start + length), (shifts.shape[0], len(t)))
    value = jnp.zeros((shifts.shape[0], maps.shape[1], maps.shape[2]), dtype=jnp.float64)
    for d in range(maps.shape[0]):
        src = start + (t[None, :] - start + shifts[:, d, None]) % max(1, length)
        # Empty intervals can start at n_time; keep inactive gathers in bounds.
        src = jnp.minimum(src, maps.shape[2] - 1)
        live = live & (veto[src] != 0)
        value = value + jnp.transpose(maps[d, :, src], (0, 2, 1))
    active = live[:, None, :] & (jnp.arange(maps.shape[1])[None, :, None] >= ib)
    value = jnp.where(value < eo, 0.0, jnp.where(value > 2 * eo, 2 * eo + 0.1, value))
    return jnp.where(active, value, 0.0), live


class AlignmentSession:
    """GPU-only lag alignment; the caller must enable JAX x64 before constructing a session.

    Parameters
    ----------
    maps : numpy.ndarray
        Float64 detector energy maps, shape ``(n_detectors, n_frequency, n_time)``,
        all axes non-empty. Copied to the first GPU once.
    valid_start : int
        First time bin of the valid interval.
    nn_valid : int
        Length of the valid interval; shifts wrap inside it.
    ib : int
        Lowest frequency layer kept, ``0 <= ib <= n_frequency``.
    veto : array-like, optional
        Per-time-bin veto flags, shape ``(n_time,)``, stored as int16; all
        ones when omitted.
    scratch_bytes : int, optional
        Bound on the output and live masks of one batch; see Notes.

    Attributes
    ----------
    device : jax.Device
        The GPU holding the resident arrays.
    shape : tuple[int, int, int]
        ``maps.shape``.
    bytes_per_lag : int
        Output bytes of one lag (float64 map plus one live byte per time bin).
    max_batch : int
        Largest number of lags :meth:`align` accepts.
    maps, veto : jax.Array
        Resident inputs.

    Raises
    ------
    RuntimeError
        If JAX x64 is not enabled.
    ValueError
        If ``maps`` is not a non-empty float64 3-D array, the interval or
        frequency bound is invalid, ``veto`` has the wrong shape, or the
        scratch budget cannot hold one lag.

    Notes
    -----
    ``scratch_bytes`` bounds output and live masks, not XLA intermediates,
    resident inputs or retained prior results. A batch may need additional
    device memory.
    """

    def __init__(
        self,
        maps: np.ndarray,
        valid_start: int,
        nn_valid: int,
        ib: int,
        veto: np.ndarray | None = None,
        scratch_bytes: int = DEFAULT_SCRATCH_BYTES,
    ) -> None:
        if not jax.config.x64_enabled:
            raise RuntimeError("Enable JAX_ENABLE_X64=1 to preserve native FP64 arithmetic")
        devices = jax.devices("gpu")
        self.device = devices[0]
        maps = np.asarray(maps)
        if maps.dtype != np.float64 or maps.ndim != 3 or min(maps.shape) < 1:
            raise ValueError("maps must be nonempty float64[detector,frequency,time]")
        self.shape = maps.shape
        self.start, self.length, self.ib = int(valid_start), int(nn_valid), int(ib)
        if self.start < 0 or self.length < 0 or self.start + self.length > maps.shape[2]:
            raise ValueError("invalid time interval")
        if not 0 <= self.ib <= maps.shape[1]:
            raise ValueError("invalid lower frequency bound")
        veto = np.ones(maps.shape[2], np.int16) if veto is None else np.asarray(veto)
        if veto.shape != (maps.shape[2],):
            raise ValueError("invalid veto shape")
        self.bytes_per_lag = maps.shape[1] * maps.shape[2] * 8 + maps.shape[2]
        self.max_batch = int(scratch_bytes) // self.bytes_per_lag
        if self.max_batch < 1:
            raise ValueError("scratch budget cannot hold one lag")
        self.maps = jax.device_put(np.ascontiguousarray(maps), self.device)
        self.veto = jax.device_put(veto.astype(np.int16), self.device)

    def align(self, shifts: np.ndarray, energy_threshold: float) -> tuple[jax.Array, jax.Array]:
        """Align a batch of lags.

        Parameters
        ----------
        shifts : numpy.ndarray
            Integer shifts, shape ``(n_lag, n_detectors)``, ``1 <= n_lag <= max_batch``;
            reduced modulo the valid length before upload.
        energy_threshold : float
            Finite, non-negative ``Eo``.

        Returns
        -------
        tuple[jax.Array, jax.Array]
            ``(value, live)`` on :attr:`device`: float64 ``(n_lag, n_frequency, n_time)``
            aligned support maps and boolean ``(n_lag, n_time)`` live masks.

        Raises
        ------
        ValueError
            If ``shifts`` is not an integer ``(n_lag, n_detectors)`` array, the
            batch is empty or exceeds ``max_batch``, or the threshold is not a
            finite non-negative number.
        """
        shifts = np.asarray(shifts)
        if shifts.ndim != 2 or shifts.shape[1] != self.shape[0] or shifts.dtype.kind not in "iu":
            raise ValueError("shifts must be integer[lag,detector]")
        if not 1 <= len(shifts) <= self.max_batch:
            raise ValueError("batch exceeds scratch budget or is empty")
        eo = float(energy_threshold)
        if not np.isfinite(eo) or eo < 0:
            raise ValueError("energy threshold must be finite and nonnegative")
        shifts = np.ascontiguousarray(shifts % max(1, self.length), dtype=np.int64)
        return _align(
            self.maps,
            jax.device_put(shifts, self.device),
            self.veto,
            jax.device_put(np.float64(eo), self.device),
            start=self.start,
            length=self.length,
            ib=self.ib,
        )
