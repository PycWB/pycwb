from dataclasses import dataclass

import numpy as np


@dataclass
class TDBatchInputs:
    """
    Pre-computed padded quadrature planes and filter tables for batch
    time-delay vector extraction.

    Created by :meth:`pycwb.types.time_frequency_map.TimeFrequencyMap.prepare_td_inputs`.

    Attributes
    ----------
    padded00 : np.ndarray
        Zero-padded 00-phase plane, float32, shape ``(n_time + 2*n_coeffs, n_cached_bands)``.
    padded90 : np.ndarray
        Zero-padded 90-phase plane, float32, shape ``(n_time + 2*n_coeffs, n_cached_bands)``.
    T0 : np.ndarray
        Same-band TD filter table, float64, shape ``(2*J+1, 2*n_coeffs+1)``.
    Tx : np.ndarray
        Cross-band TD filter table, float64, shape ``(2*J+1, 2*n_coeffs+1)``.
    M : int
        Number of frequency layers (Nyquist index).
    n_coeffs : int
        Half-width of symmetric TD filters.
    J : int
        Maximum delay index (``M * L``).
    frequency_offset : int
        Global band index of the first cached column; pixel indices and phase
        parity always use the original ``M+1`` frequency stride.
    """

    padded00: np.ndarray
    padded90: np.ndarray
    T0: np.ndarray
    Tx: np.ndarray
    M: int
    n_coeffs: int
    J: int
    frequency_offset: int = 0

    def extract_td_vecs(self, pixel_indices, K, delay_stride=1):
        """
        Batch TD vector extraction for the given pixel indices.

        Parameters
        ----------
        pixel_indices : array-like, shape (n_pixels,)
            Flat pixel indices into the TF plane (int32).
        K : int
            TD filter half-range (corresponds to config.TDSize).

        delay_stride : int
            Spacing in filter-bank delay samples (default 1).

        Returns
        -------
        np.ndarray, shape (n_pixels, 4*K+2), dtype float32
        """
        from pycwb.utils.td_vector_kernels import batch_get_td_vecs

        if delay_stride < 1 or int(delay_stride) != delay_stride:
            raise ValueError("delay_stride must be a positive integer")
        indices = np.asarray(pixel_indices, dtype=np.int32)
        if self.frequency_offset or self.padded00.shape[1] != self.M + 1:
            bands = indices % (self.M + 1)
            if (np.any(np.maximum(bands - 1, 0) < self.frequency_offset)
                    or np.any(np.minimum(bands + 1, self.M) >= self.frequency_offset + self.padded00.shape[1])):
                raise ValueError("Requested pixel support lies outside the cached frequency band")
        return batch_get_td_vecs(
            indices,
            self.padded00,
            self.padded90,
            self.T0,
            self.Tx,
            self.M,
            self.n_coeffs,
            K,
            self.J,
            int(delay_stride),
            self.frequency_offset,
        )
