"""
Thread-safe per-IFO ring buffer for online data acquisition.

Stores a contiguous window of data in memory.  The data acquisition
thread calls :meth:`append` to add new data; the orchestrator calls
:meth:`snapshot` to extract analysis-ready windows.
"""

import logging
import threading

import numpy as np

logger = logging.getLogger(__name__)


class RingBuffer:
    """Fixed-capacity ring buffer for a single IFO channel.

    Parameters
    ----------
    capacity_seconds : float
        Maximum duration (seconds) of data retained.
    sample_rate : float
        Sample rate of incoming data.
    """

    def __init__(self, capacity_seconds: float, sample_rate: float):
        self.capacity_seconds = capacity_seconds
        self.sample_rate = sample_rate
        self._capacity_samples = int(capacity_seconds * sample_rate)
        self._buf = np.zeros(self._capacity_samples, dtype=np.float64)
        self._write_pos = 0          # total samples written (monotonic)
        self._start_gps = None       # GPS of the oldest sample currently held
        self._last_gps = None        # GPS of the most recently appended sample
        self._last_snapshot_gps = None
        self._gap_detected = False   # set True when a significant gap is seen
        self._pre_gap_gps = None     # last_gps before the gap (for partial emit)
        self._lock = threading.Lock()

    @property
    def last_gps(self):
        """GPS time up to which data is available (exclusive boundary)."""
        with self._lock:
            return self._last_gps

    def available_new_duration(self) -> float:
        """Seconds of data appended since the last snapshot trigger."""
        with self._lock:
            if self._last_gps is None or self._last_snapshot_gps is None:
                return 0.0
            return max(0.0, self._last_gps - self._last_snapshot_gps)

    def check_and_clear_gap(self):
        """Check if a gap was detected and clear the flag.

        Returns
        -------
        tuple (bool, float or None)
            ``(gap_detected, pre_gap_gps)`` — the GPS time just before the
            gap, or *None* if no gap.
        """
        with self._lock:
            if self._gap_detected:
                self._gap_detected = False
                pre_gap = self._pre_gap_gps
                self._pre_gap_gps = None
                return True, pre_gap
            return False, None

    def append(self, chunk_data: np.ndarray, chunk_start_gps: float) -> None:
        """Append a contiguous chunk of data.

        Parameters
        ----------
        chunk_data : np.ndarray
            1-D array of samples to append.
        chunk_start_gps : float
            GPS time of the first sample in *chunk_data*.
        """
        n = len(chunk_data)
        if n == 0:
            return

        with self._lock:
            chunk_end_gps = chunk_start_gps + n / self.sample_rate

            # Gap detection
            if self._last_gps is not None:
                expected_gps = self._last_gps + 1.0 / self.sample_rate
                gap = chunk_start_gps - expected_gps
                if abs(gap) > 1.0 / self.sample_rate:
                    if abs(gap) > 1.0:
                        logger.warning(
                            "RingBuffer: GPS gap of %.3f s detected "
                            "(expected %.6f, got %.6f). Resetting buffer.",
                            gap, expected_gps, chunk_start_gps,
                        )
                        self._pre_gap_gps = self._last_gps
                        self._gap_detected = True
                        self._reset_unlocked()

            # Write into circular buffer
            start_idx = self._write_pos % self._capacity_samples
            if start_idx + n <= self._capacity_samples:
                self._buf[start_idx:start_idx + n] = chunk_data
            else:
                first = self._capacity_samples - start_idx
                self._buf[start_idx:] = chunk_data[:first]
                self._buf[:n - first] = chunk_data[first:]

            self._write_pos += n
            self._last_gps = chunk_end_gps

            if self._start_gps is None:
                self._start_gps = chunk_start_gps
            else:
                # Evict oldest data beyond capacity
                total_held = min(self._write_pos, self._capacity_samples)
                self._start_gps = self._last_gps - (total_held - 1) / self.sample_rate

    def snapshot(self, start_gps: float, end_gps: float) -> np.ndarray:
        """Return a copy of data in ``[start_gps, end_gps)``.

        Parameters
        ----------
        start_gps, end_gps : float
            GPS bounds (end exclusive).

        Returns
        -------
        np.ndarray
            Copy of the requested data window.

        Raises
        ------
        ValueError
            If the requested range is outside the buffered window.
        """
        with self._lock:
            if self._start_gps is None:
                raise ValueError("RingBuffer is empty")

            n_samples = int(round((end_gps - start_gps) * self.sample_rate))

            # Offset from the start of the buffer.
            # Allow up to 1 sample of float-precision drift — clamp to 0.
            offset_samples = int(round((start_gps - self._start_gps) * self.sample_rate))
            if offset_samples < 0:
                if offset_samples >= -1:
                    # Float-precision drift (< 1 sample); clamp
                    logger.debug(
                        "Clamping snapshot offset from %d to 0 "
                        "(start=%.6f, buf_start=%.6f)",
                        offset_samples, start_gps, self._start_gps,
                    )
                    offset_samples = 0
                else:
                    raise ValueError(
                        f"Requested start {start_gps:.6f} is before buffer "
                        f"start {self._start_gps:.6f} "
                        f"(offset={offset_samples} samples)"
                    )

            total_held = min(self._write_pos, self._capacity_samples)
            if offset_samples + n_samples > total_held:
                raise ValueError(
                    f"Requested end {end_gps:.6f} exceeds buffered data "
                    f"(buffer ends at {self._last_gps:.6f})"
                )

            # Map to circular buffer indices
            oldest_idx = (self._write_pos - total_held) % self._capacity_samples
            read_start = (oldest_idx + offset_samples) % self._capacity_samples

            result = np.empty(n_samples, dtype=np.float64)
            if read_start + n_samples <= self._capacity_samples:
                result[:] = self._buf[read_start:read_start + n_samples]
            else:
                first = self._capacity_samples - read_start
                result[:first] = self._buf[read_start:]
                result[first:] = self._buf[:n_samples - first]

            self._last_snapshot_gps = end_gps
            return result

    def mark_snapshot(self) -> None:
        """Mark current time as the snapshot reference for stride tracking."""
        with self._lock:
            self._last_snapshot_gps = self._last_gps

    def _reset_unlocked(self):
        """Reset buffer state (caller must hold _lock)."""
        self._buf[:] = 0.0
        self._write_pos = 0
        self._start_gps = None
        self._last_gps = None
        self._last_snapshot_gps = None
