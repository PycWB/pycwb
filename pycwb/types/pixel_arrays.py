"""Struct-of-Arrays (SoA) storage for all pixels in a cluster.

Motivation
----------
The original ``Pixel`` / ``Cluster`` design stores every field on individual
Python objects (array-of-structs, AoS).  This causes a round-trip overhead in
the hot path:

1. Supercluster computes a dense ``(n_pix, n_ifo, tsize)`` td-amp matrix.
2. It slices it into per-pixel ``pix.td_amp = [arr_ifo0, ...]`` Python lists.
3. Likelihood reassembles the matrix from those lists to call Numba / JAX.

``PixelArrays`` eliminates steps 2-3 by keeping pixel data in compact NumPy
arrays at the cluster level.

TD-amp storage
--------------
``td_amp`` is stored as a **flat 1-D array** with CSR-style offset pointers::

    for pixel i, ifo j:
        row  = i * n_ifo + j
        data = td_amp_flat[td_amp_offsets[row] : td_amp_offsets[row + 1]]

Even though ``tsize`` is globally fixed (``4 * config.TDSize + 2``) today, the
CSR layout generalises to variable-length vectors across resolutions without
any padding, and is the layout requested by the user.  When all rows have the
same length, ``.td_amp_dense()`` returns a zero-copy reshape as
``(n_pix, n_ifo, tsize)``.

JAX pytree
----------
``PixelArrays`` is registered as a JAX pytree node so that ``jax.jit``,
``jax.vmap``, and ``jax.grad`` can traverse and transform it as a tree of
device arrays.  Static metadata (``_n_ifo``) lives in the auxiliary slot and
is therefore not traced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np


@dataclass
class PixelArrays:
    """Column-major (SoA) storage for all pixels in a cluster.

    Scalar fields
    ~~~~~~~~~~~~~
    * ``time``, ``frequency``, ``layers``, ``rate``, ``core``,
      ``likelihood``, ``null`` — shape ``(n_pix,)``

    Per-IFO scalar fields
    ~~~~~~~~~~~~~~~~~~~~~
    * ``noise_rms``, ``wave``, ``w_90``, ``asnr``, ``a_90``,
      ``pixel_index`` — shape ``(n_ifo, n_pix)``

    TD-amplitude (CSR)
    ~~~~~~~~~~~~~~~~~~
    * ``td_amp_flat``    — ``(total_elements,)`` float32
    * ``td_amp_offsets`` — ``(n_pix * n_ifo + 1,)`` int32;
      ``td_amp_flat[offsets[row]:offsets[row+1]]`` gives the vector for
      ``row = pixel_idx * n_ifo + ifo_idx``.

    Static
    ~~~~~~
    * ``_n_ifo`` — integer, not a JAX leaf.
    """

    # ------------------------------------------------------------------ #
    # per-pixel scalars  (n_pix,)
    # ------------------------------------------------------------------ #
    time:       np.ndarray  # int32
    frequency:  np.ndarray  # int32
    layers:     np.ndarray  # int32
    rate:       np.ndarray  # float32
    core:       np.ndarray  # bool
    likelihood: np.ndarray  # float32
    null:       np.ndarray  # float32

    # ------------------------------------------------------------------ #
    # per-IFO scalars  (n_ifo, n_pix)
    # ------------------------------------------------------------------ #
    noise_rms:   np.ndarray  # float32
    wave:        np.ndarray  # float32
    w_90:        np.ndarray  # float32
    asnr:        np.ndarray  # float32
    a_90:        np.ndarray  # float32
    pixel_index: np.ndarray  # int32

    # ------------------------------------------------------------------ #
    # TD-amplitude: flat CSR layout
    # ------------------------------------------------------------------ #
    td_amp_flat:    np.ndarray  # (total_elements,) float32
    td_amp_offsets: np.ndarray  # (n_pix * n_ifo + 1,) int32

    # ------------------------------------------------------------------ #
    # static metadata (not a JAX leaf)
    # ------------------------------------------------------------------ #
    _n_ifo: int = field(default=0, repr=False)

    # ================================================================== #
    # Construction helpers
    # ================================================================== #

    @classmethod
    def from_pixels(cls, pixels: list, n_ifo: int) -> "PixelArrays":
        """Build from a list of ``Pixel`` objects (backward-compat entry).

        For the fast path (when a dense td-amp array is already available)
        prefer :meth:`from_arrays`.
        """
        n_pix = len(pixels)

        time       = np.array([p.time       for p in pixels], dtype=np.int32)
        frequency  = np.array([p.frequency  for p in pixels], dtype=np.int32)
        layers     = np.array([p.layers     for p in pixels], dtype=np.int32)
        rate       = np.array([p.rate       for p in pixels], dtype=np.float32)
        core       = np.array([p.core       for p in pixels], dtype=bool)
        likelihood = np.array([p.likelihood for p in pixels], dtype=np.float32)
        null       = np.array([p.null       for p in pixels], dtype=np.float32)

        noise_rms   = np.array([[p.data[j].noise_rms for p in pixels] for j in range(n_ifo)], dtype=np.float32)
        wave        = np.array([[p.data[j].wave       for p in pixels] for j in range(n_ifo)], dtype=np.float32)
        w_90        = np.array([[p.data[j].w_90       for p in pixels] for j in range(n_ifo)], dtype=np.float32)
        asnr        = np.array([[p.data[j].asnr       for p in pixels] for j in range(n_ifo)], dtype=np.float32)
        a_90        = np.array([[p.data[j].a_90       for p in pixels] for j in range(n_ifo)], dtype=np.float32)
        pixel_index = np.array([[p.data[j].index      for p in pixels] for j in range(n_ifo)], dtype=np.int32)

        # Build CSR td_amp from per-pixel lists
        segments: list[np.ndarray] = []
        for pix in pixels:
            td = getattr(pix, "td_amp", None)
            for j in range(n_ifo):
                if td is not None and j < len(td) and td[j] is not None:
                    segments.append(np.asarray(td[j], dtype=np.float32).ravel())
                else:
                    segments.append(np.zeros(0, dtype=np.float32))

        offsets = np.zeros(n_pix * n_ifo + 1, dtype=np.int32)
        for k, seg in enumerate(segments):
            offsets[k + 1] = offsets[k] + len(seg)
        td_amp_flat = np.concatenate(segments) if segments else np.zeros(0, dtype=np.float32)

        return cls(
            time=time, frequency=frequency, layers=layers, rate=rate,
            core=core, likelihood=likelihood, null=null,
            noise_rms=noise_rms, wave=wave, w_90=w_90, asnr=asnr, a_90=a_90,
            pixel_index=pixel_index,
            td_amp_flat=td_amp_flat, td_amp_offsets=offsets,
            _n_ifo=n_ifo,
        )

    @classmethod
    def from_arrays(
        cls,
        *,
        time: np.ndarray,
        frequency: np.ndarray,
        layers: np.ndarray,
        rate: np.ndarray,
        noise_rms: np.ndarray,
        pixel_index: np.ndarray,
        n_ifo: int,
        core: np.ndarray | None = None,
        likelihood: np.ndarray | None = None,
        null: np.ndarray | None = None,
        wave: np.ndarray | None = None,
        w_90: np.ndarray | None = None,
        asnr: np.ndarray | None = None,
        a_90: np.ndarray | None = None,
        td_amp_dense: np.ndarray | None = None,
    ) -> "PixelArrays":
        """Build directly from pre-computed NumPy arrays (preferred fast path).

        Parameters
        ----------
        time, frequency, layers, rate : (n_pix,) arrays
        noise_rms, pixel_index        : (n_ifo, n_pix) arrays
        n_ifo                         : int
        td_amp_dense                  : (n_pix, n_ifo, tsize) float32 | None
            When provided the data are packed into the CSR layout with a
            uniform stride (zero-copy reshape).
        """
        n_pix = len(time)
        _zeros_pix = lambda: np.zeros(n_pix, dtype=np.float32)
        _zeros_ifo = lambda: np.zeros((n_ifo, n_pix), dtype=np.float32)

        if td_amp_dense is not None:
            flat, offsets = _dense_to_csr(td_amp_dense)
        else:
            flat    = np.zeros(0, dtype=np.float32)
            offsets = np.zeros(n_pix * n_ifo + 1, dtype=np.int32)

        return cls(
            time        = np.asarray(time,        dtype=np.int32),
            frequency   = np.asarray(frequency,   dtype=np.int32),
            layers      = np.asarray(layers,       dtype=np.int32),
            rate        = np.asarray(rate,         dtype=np.float32),
            core        = np.asarray(core,         dtype=bool) if core is not None else np.zeros(n_pix, dtype=bool),
            likelihood  = np.asarray(likelihood,   dtype=np.float32) if likelihood is not None else _zeros_pix(),
            null        = np.asarray(null,         dtype=np.float32) if null        is not None else _zeros_pix(),
            noise_rms   = np.asarray(noise_rms,    dtype=np.float32),
            wave        = np.asarray(wave,         dtype=np.float32) if wave  is not None else _zeros_ifo(),
            w_90        = np.asarray(w_90,         dtype=np.float32) if w_90  is not None else _zeros_ifo(),
            asnr        = np.asarray(asnr,         dtype=np.float32) if asnr  is not None else _zeros_ifo(),
            a_90        = np.asarray(a_90,         dtype=np.float32) if a_90  is not None else _zeros_ifo(),
            pixel_index = np.asarray(pixel_index,  dtype=np.int32),
            td_amp_flat    = flat,
            td_amp_offsets = offsets,
            _n_ifo = n_ifo,
        )

    # ================================================================== #
    # TD-amplitude access
    # ================================================================== #

    def get_td_amp(self, pix_idx: int, ifo_idx: int) -> np.ndarray:
        """Return the td-amp vector for pixel ``pix_idx``, IFO ``ifo_idx``."""
        row = pix_idx * self._n_ifo + ifo_idx
        return self.td_amp_flat[self.td_amp_offsets[row]: self.td_amp_offsets[row + 1]]

    def td_amp_dense(self) -> np.ndarray:
        """Return td-amp as a dense ``(n_pix, n_ifo, tsize)`` float32 array.

        Requires uniform row length (same ``tsize`` for every pixel/IFO).
        Raises ``ValueError`` if sizes differ.
        """
        n_pix = len(self.time)
        n_ifo = self._n_ifo
        n_rows = n_pix * n_ifo
        if n_rows == 0 or len(self.td_amp_flat) == 0:
            return np.zeros((n_pix, n_ifo, 0), dtype=np.float32)
        sizes = self.td_amp_offsets[1: n_rows + 1] - self.td_amp_offsets[:n_rows]
        tsize = int(sizes[0])
        if not np.all(sizes == tsize):
            raise ValueError(
                "td_amp vectors have non-uniform lengths; cannot form a dense array. "
                "Use get_td_amp(i, j) for per-pixel access instead."
            )
        return self.td_amp_flat[: n_rows * tsize].reshape(n_pix, n_ifo, tsize)

    def has_td_amp(self) -> bool:
        """Return ``True`` if td-amp data is populated."""
        return len(self.td_amp_flat) > 0

    def set_td_amp_from_dense(self, dense: np.ndarray) -> None:
        """Replace the CSR td-amp with data from a dense array (in-place).

        Parameters
        ----------
        dense : (n_pix, n_ifo, tsize) float32
        """
        self.td_amp_flat, self.td_amp_offsets = _dense_to_csr(dense)

    # ================================================================== #
    # Write-back helpers (vectorised, no per-pixel loop)
    # ================================================================== #

    def set_waveform_data(
        self,
        wave: np.ndarray,   # (n_ifo, n_pix)
        w_90: np.ndarray,   # (n_ifo, n_pix)
        asnr: np.ndarray,   # (n_ifo, n_pix)
        a_90: np.ndarray,   # (n_ifo, n_pix)
        core_mask: np.ndarray,              # (n_pix,) bool/int
        energy_plus: np.ndarray,            # (n_pix,) float32
        energy_cross: np.ndarray,           # (n_pix,) float32
    ) -> None:
        """Vectorised replacement for the per-pixel write-back loop in
        ``_set_pixel_waveform_data`` (likelihood.py lines 977-990).

        Updates all arrays in-place without iterating over individual pixels.
        """
        self.wave[:] = np.asarray(wave, dtype=np.float32)
        self.w_90[:] = np.asarray(w_90, dtype=np.float32)
        self.asnr[:] = np.asarray(asnr, dtype=np.float32)
        self.a_90[:] = np.asarray(a_90, dtype=np.float32)
        self.core[:] = (core_mask > 0)
        self.likelihood[:] = np.where(
            core_mask > 0,
            -(np.asarray(energy_plus, dtype=np.float32) + np.asarray(energy_cross, dtype=np.float32)) / 2.0,
            0.0,
        )
        self.null[:] = 0.0

    # ================================================================== #
    # Subscript / length
    # ================================================================== #

    def __len__(self) -> int:
        return len(self.time)

    def __getitem__(self, idx):
        """User-friendly per-pixel or slice access.

        * **Integer index** → ``dict`` with all fields for that pixel.
        * **Slice / boolean mask / int array** → new ``PixelArrays``
          containing only the selected rows.
        """
        if isinstance(idx, (int, np.integer)):
            n_ifo = self._n_ifo
            return {
                "time":        int(self.time[idx]),
                "frequency":   int(self.frequency[idx]),
                "layers":      int(self.layers[idx]),
                "rate":        float(self.rate[idx]),
                "core":        bool(self.core[idx]),
                "likelihood":  float(self.likelihood[idx]),
                "null":        float(self.null[idx]),
                "noise_rms":   self.noise_rms[:, idx].copy(),
                "wave":        self.wave[:, idx].copy(),
                "w_90":        self.w_90[:, idx].copy(),
                "asnr":        self.asnr[:, idx].copy(),
                "a_90":        self.a_90[:, idx].copy(),
                "pixel_index": self.pixel_index[:, idx].copy(),
                "td_amp":      [self.get_td_amp(int(idx), j) for j in range(n_ifo)],
            }
        return self._subset(idx)

    def _subset(self, idx) -> "PixelArrays":
        """Return a ``PixelArrays`` containing only the selected pixels."""
        n_ifo = self._n_ifo
        # Normalise idx to a sorted integer array
        rows = np.arange(len(self.time), dtype=np.int64)[idx]
        if rows.ndim == 0:
            rows = rows.reshape(1)

        parts: list[np.ndarray] = []
        new_offsets = [np.int32(0)]
        for i in rows:
            for j in range(n_ifo):
                chunk = self.get_td_amp(int(i), j)
                parts.append(chunk)
                new_offsets.append(np.int32(new_offsets[-1] + len(chunk)))

        new_flat    = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
        new_offsets_arr = np.array(new_offsets, dtype=np.int32)

        return PixelArrays(
            time        = self.time[idx],
            frequency   = self.frequency[idx],
            layers      = self.layers[idx],
            rate        = self.rate[idx],
            core        = self.core[idx],
            likelihood  = self.likelihood[idx],
            null        = self.null[idx],
            noise_rms   = self.noise_rms[:, idx],
            wave        = self.wave[:, idx],
            w_90        = self.w_90[:, idx],
            asnr        = self.asnr[:, idx],
            a_90        = self.a_90[:, idx],
            pixel_index = self.pixel_index[:, idx],
            td_amp_flat    = new_flat,
            td_amp_offsets = new_offsets_arr,
            _n_ifo = n_ifo,
        )

    # ================================================================== #
    # Noise-RMS update (vectorised)
    # ================================================================== #

    def populate_noise_rms(self, nRMS: list) -> None:
        """Vectorised replacement for ``_populate_pixel_noise_rms``.

        Updates ``self.noise_rms`` in-place from TF noise maps produced by
        the whitening step.  No per-pixel Python loop.

        Parameters
        ----------
        nRMS : list[TimeFrequencyMap]
            One noise map per IFO.  ``data`` shape is ``(n_freq, n_time)``.
        """
        n_ifo = self._n_ifo
        freq_bins  = self.frequency.astype(np.int64)    # (n_pix,)
        layers_arr = self.layers.astype(np.int64)       # (n_pix,)
        time_bins  = np.where(
            layers_arr > 0,
            self.time.astype(np.int64) // layers_arr,
            0,
        )

        for i in range(n_ifo):
            arr = np.asarray(nRMS[i].data, dtype=np.float64)
            nf, nt = arr.shape
            fb = np.where(
                layers_arr > 0,
                np.round(freq_bins * nf / layers_arr).astype(np.int64),
                0,
            )
            fb = np.clip(fb, 0, nf - 1)
            tb = np.clip(time_bins, 0, nt - 1)
            vals = np.abs(arr[fb, tb]).astype(np.float32)
            # Only overwrite where the map has positive values
            self.noise_rms[i] = np.where(vals > 0.0, vals, self.noise_rms[i])

    # ================================================================== #
    # Concatenation
    # ================================================================== #

    @classmethod
    def concat(cls, arrays: list["PixelArrays"]) -> "PixelArrays":
        """Concatenate a list of ``PixelArrays`` along the pixel axis.

        Handles empty td-amp (most common case during the supercluster stage,
        before td vectors are computed) and uniform-stride td-amp efficiently.
        """
        if not arrays:
            raise ValueError("Cannot concatenate an empty list")
        n_ifo = arrays[0]._n_ifo

        time        = np.concatenate([a.time       for a in arrays])
        frequency   = np.concatenate([a.frequency  for a in arrays])
        layers      = np.concatenate([a.layers     for a in arrays])
        rate        = np.concatenate([a.rate       for a in arrays])
        core        = np.concatenate([a.core       for a in arrays])
        likelihood  = np.concatenate([a.likelihood for a in arrays])
        null        = np.concatenate([a.null       for a in arrays])
        noise_rms   = np.concatenate([a.noise_rms   for a in arrays], axis=1)
        wave        = np.concatenate([a.wave        for a in arrays], axis=1)
        w_90        = np.concatenate([a.w_90        for a in arrays], axis=1)
        asnr        = np.concatenate([a.asnr        for a in arrays], axis=1)
        a_90        = np.concatenate([a.a_90        for a in arrays], axis=1)
        pixel_index = np.concatenate([a.pixel_index for a in arrays], axis=1)

        all_empty = not any(a.has_td_amp() for a in arrays)
        if all_empty:
            n_rows = len(time) * n_ifo
            td_flat    = np.zeros(0, dtype=np.float32)
            td_offsets = np.zeros(n_rows + 1, dtype=np.int32)
        else:
            # All filled with uniform stride: concatenate dense, then pack CSR
            all_dense = [a.td_amp_dense() for a in arrays]  # each (n_pix_i, n_ifo, tsize)
            combined  = np.concatenate(all_dense, axis=0)
            td_flat, td_offsets = _dense_to_csr(combined)

        return cls(
            time=time, frequency=frequency, layers=layers, rate=rate,
            core=core, likelihood=likelihood, null=null,
            noise_rms=noise_rms, wave=wave, w_90=w_90, asnr=asnr, a_90=a_90,
            pixel_index=pixel_index,
            td_amp_flat=td_flat, td_amp_offsets=td_offsets,
            _n_ifo=n_ifo,
        )

    # ================================================================== #
    # Backward-compat: reconstruct list[Pixel]
    # ================================================================== #

    def to_pixel_list(self) -> list:
        """Reconstruct a ``list[Pixel]`` for code that still iterates over
        individual ``Pixel`` objects.  This is an escape hatch — avoid in
        hot paths.
        """
        from pycwb.types.network_pixel import Pixel, PixelData

        n_pix = len(self.time)
        n_ifo = self._n_ifo
        pixels = []
        for i in range(n_pix):
            data = [
                PixelData(
                    noise_rms = float(self.noise_rms[j, i]),
                    wave      = float(self.wave[j, i]),
                    w_90      = float(self.w_90[j, i]),
                    asnr      = float(self.asnr[j, i]),
                    a_90      = float(self.a_90[j, i]),
                    rank      = 0.0,
                    index     = int(self.pixel_index[j, i]),
                )
                for j in range(n_ifo)
            ]
            pix = Pixel(
                time         = int(self.time[i]),
                frequency    = int(self.frequency[i]),
                layers       = int(self.layers[i]),
                rate         = float(self.rate[i]),
                likelihood   = float(self.likelihood[i]),
                null         = float(self.null[i]),
                theta        = 0.0,
                phi          = 0.0,
                ellipticity  = 0.0,
                polarisation = 0.0,
                core         = bool(self.core[i]),
                data         = data,
                td_amp       = [self.get_td_amp(i, j) for j in range(n_ifo)],
                neighbors    = [],
            )
            pixels.append(pix)
        return pixels


# ====================================================================== #
# Internal helper
# ====================================================================== #

def _dense_to_csr(dense: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pack a dense ``(n_pix, n_ifo, tsize)`` array into CSR layout.

    Since ``tsize`` is uniform, the offset array is a simple arithmetic
    progression: ``offsets[row] = row * tsize``.  This means the flat array
    is just a C-contiguous reshape, with no data copy when ``dense`` is
    already float32 and C-contiguous.

    Returns
    -------
    flat    : (n_pix * n_ifo * tsize,) float32
    offsets : (n_pix * n_ifo + 1,) int32
    """
    n_pix, n_ifo, tsize = dense.shape
    flat = np.ascontiguousarray(dense, dtype=np.float32).ravel()
    n_rows = n_pix * n_ifo
    offsets = np.arange(0, n_rows * tsize + tsize, tsize, dtype=np.int32)
    return flat, offsets


# ====================================================================== #
# Empty helper
# ====================================================================== #

def empty_pixel_arrays(n_ifo: int = 0) -> PixelArrays:
    """Return an empty ``PixelArrays`` with zero pixels."""
    return PixelArrays.from_arrays(
        time        = np.zeros(0, dtype=np.int32),
        frequency   = np.zeros(0, dtype=np.int32),
        layers      = np.zeros(0, dtype=np.int32),
        rate        = np.zeros(0, dtype=np.float32),
        noise_rms   = np.zeros((n_ifo, 0), dtype=np.float32),
        pixel_index = np.zeros((n_ifo, 0), dtype=np.int32),
        n_ifo       = n_ifo,
    )


# ====================================================================== #
# JAX pytree registration
# ====================================================================== #

def _pa_flatten(pa: PixelArrays):
    leaves = [
        pa.time, pa.frequency, pa.layers, pa.rate, pa.core,
        pa.likelihood, pa.null,
        pa.noise_rms, pa.wave, pa.w_90, pa.asnr, pa.a_90, pa.pixel_index,
        pa.td_amp_flat, pa.td_amp_offsets,
    ]
    aux = pa._n_ifo  # static — not traced by JAX
    return leaves, aux

def _pa_unflatten(aux: int, leaves: list) -> PixelArrays:
    (time, frequency, layers, rate, core,
        likelihood, null,
        noise_rms, wave, w_90, asnr, a_90, pixel_index,
        td_amp_flat, td_amp_offsets) = leaves
    pa = object.__new__(PixelArrays)
    pa.time           = time
    pa.frequency      = frequency
    pa.layers         = layers
    pa.rate           = rate
    pa.core           = core
    pa.likelihood     = likelihood
    pa.null           = null
    pa.noise_rms      = noise_rms
    pa.wave           = wave
    pa.w_90           = w_90
    pa.asnr           = asnr
    pa.a_90           = a_90
    pa.pixel_index    = pixel_index
    pa.td_amp_flat    = td_amp_flat
    pa.td_amp_offsets = td_amp_offsets
    pa._n_ifo         = aux
    return pa

jax.tree_util.register_pytree_node(PixelArrays, _pa_flatten, _pa_unflatten)
