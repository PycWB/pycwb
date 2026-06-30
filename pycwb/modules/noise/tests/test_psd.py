"""Tests for pycwb.modules.noise.psd — PSD loading and interpolation."""
import os
import tempfile
import logging
import unittest
import numpy as np
import pytest
from pycwb.modules.noise.psd import load_psd


class TestLoadPsd(unittest.TestCase):
    """Tests for load_psd — load two-column ASD file and return PSD array."""

    def _write_asd_file(self, path: str, freqs: np.ndarray, asd: np.ndarray):
        """Write a two-column ASD text file."""
        data = np.column_stack([freqs, asd])
        np.savetxt(path, data)

    def test_basic_interpolation(self):
        """Simple ASD → squared PSD interpolation on uniform grid."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "psd.txt")
            # ASD file: 10 Hz → 1e-23, 100 Hz → 2e-23
            self._write_asd_file(path,
                                 np.array([10.0, 100.0]),
                                 np.array([1e-23, 2e-23]))

            flen = 50  # 50 bins
            delta_f = 2.0  # 0, 2, 4, ..., 98 Hz
            f_low = 5.0

            result = load_psd(path, flen, delta_f, f_low)
            assert len(result) == flen
            # Below f_low should be zero
            assert np.all(result[:3] == 0.0)  # 0, 2, 4 Hz < 5 Hz
            # At 10 Hz: ASD=1e-23 → PSD=1e-46
            idx_10 = int(10 / delta_f)
            assert result[idx_10] == pytest.approx(1e-46, rel=1e-10)
            # At 100 Hz: ASD=2e-23 → PSD=4e-46
            idx_100 = int(100 / delta_f)
            if idx_100 < flen:
                assert result[idx_100] == pytest.approx(4e-46, rel=1e-10)

    def test_zero_below_f_low(self):
        """All frequencies below f_low should be zero."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "psd.txt")
            self._write_asd_file(path,
                                 np.array([20.0, 200.0]),
                                 np.array([1e-23, 1e-23]))
            flen = 100
            delta_f = 1.0
            f_low = 30.0

            result = load_psd(path, flen, delta_f, f_low)
            # indices 0-29 should be zero (0-29 Hz < 30 Hz)
            assert np.all(result[:30] == 0.0)
            # index 30 (30 Hz) is >= f_low, should be non-zero
            assert result[30] > 0.0

    def test_beyond_file_range_gets_zero(self):
        """Frequencies beyond the file's max frequency get fill_value=0."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "psd.txt")
            self._write_asd_file(path,
                                 np.array([10.0, 50.0]),
                                 np.array([1e-23, 1e-23]))
            flen = 200
            delta_f = 1.0
            f_low = 0.0

            result = load_psd(path, flen, delta_f, f_low)
            # frequencies above 50 Hz should be 0 (bounds_error=False, fill_value=0)
            assert np.all(result[51:] == 0.0)

    def test_warns_when_request_exceeds_file(self):
        """Should log a warning when requested max freq exceeds file's max freq.

        The warning is emitted via ``logger.warning()``, not ``warnings.warn()``,
        so we capture log output instead of using ``pytest.warns``.
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "psd.txt")
            self._write_asd_file(path,
                                 np.array([10.0, 20.0]),
                                 np.array([1e-23, 1e-23]))
            flen = 1000
            delta_f = 1.0
            f_low = 10.0

            with self.assertLogs("pycwb.modules.noise.psd", level="WARNING") as cm:
                load_psd(path, flen, delta_f, f_low)
            assert any("exceeds the highest available" in msg for msg in cm.output)

    def test_invalid_file_raises(self):
        """A file with only one column should raise ValueError."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.txt")
            np.savetxt(path, np.array([1.0, 2.0, 3.0]))  # single column
            with pytest.raises(ValueError, match="at least two columns"):
                load_psd(path, 10, 1.0, 0.0)

    def test_non_existent_file_raises(self):
        """Non-existent file path should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_psd("/nonexistent/path.psd", 10, 1.0, 0.0)
