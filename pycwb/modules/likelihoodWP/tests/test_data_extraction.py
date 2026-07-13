"""Characterization tests for pixel data extraction.

Uses synthetic ``PixelArrays`` fixtures to verify the data extraction
contract before and after the split.
"""

import numpy as np
import pytest

from pycwb.types.pixel_arrays import PixelArrays


def _make_synthetic_pa(
    n_ifo: int = 2,
    n_pix: int = 3,
    tsize: int = 8,
    noise_rms_diag: float = 1.0,
    random_state: np.random.RandomState | None = None,
):
    """Create a minimal synthetic PixelArrays with dense td_amp.

    Returns (pa, n_ifo, n_pix, tsize2) where tsize2 = tsize // 2.
    """
    if random_state is None:
        rng = np.random.RandomState(42)
    else:
        rng = random_state

    tsize_half = tsize // 2

    td_dense = rng.randn(n_pix, n_ifo, tsize).astype(np.float32) * 0.1
    noise_rms = np.full((n_ifo, n_pix), noise_rms_diag, dtype=np.float32)
    noise_rms[1] *= 1.5  # make non-uniform

    pa = PixelArrays.from_arrays(
        time=np.arange(n_pix, dtype=np.int32),
        frequency=np.full(n_pix, 10, dtype=np.int32),
        layers=np.full(n_pix, 4, dtype=np.int32),
        rate=np.full(n_pix, 1024.0, dtype=np.float32),
        noise_rms=noise_rms,
        pixel_index=np.zeros((n_ifo, n_pix), dtype=np.int32),
        n_ifo=n_ifo,
        td_amp_dense=td_dense,
        core=np.ones(n_pix, dtype=bool),
    )
    return pa


class TestPixelDataExtraction:
    """Test the pixel data extraction contract."""

    def test_load_data_from_pixels_returns_expected_shapes(self):
        """Verify rms, td00, td90, td_energy shapes from PixelArrays."""
        from pycwb.modules.likelihoodWP.pixel_data import load_data_from_pixels

        n_ifo, n_pix, tsize = 2, 3, 8
        pa = _make_synthetic_pa(n_ifo=n_ifo, n_pix=n_pix, tsize=tsize)
        rms, td00, td90, td_energy = load_data_from_pixels(None, n_ifo, pixel_arrays=pa)

        tsize_half = tsize // 2
        assert rms.shape == (n_ifo, n_pix)
        assert td00.shape == (n_ifo, n_pix, tsize_half)
        assert td90.shape == (n_ifo, n_pix, tsize_half)
        assert td_energy.shape == (n_ifo, n_pix, tsize_half)

    def test_td_energy_equals_sqsum(self):
        """td_energy == td00**2 + td90**2."""
        from pycwb.modules.likelihoodWP.pixel_data import load_data_from_pixels

        n_ifo, n_pix, tsize = 2, 3, 8
        pa = _make_synthetic_pa(n_ifo=n_ifo, n_pix=n_pix, tsize=tsize)
        rms, td00, td90, td_energy = load_data_from_pixels(None, n_ifo, pixel_arrays=pa)

        np.testing.assert_allclose(td_energy, td00 ** 2 + td90 ** 2, rtol=1e-6)

    def test_extract_pixel_time_delay_data_is_alias(self):
        """load_data_from_pixels is the same as extract_pixel_time_delay_data."""
        from pycwb.modules.likelihoodWP.pixel_data import (
            load_data_from_pixels, extract_pixel_time_delay_data,
        )
        assert extract_pixel_time_delay_data is load_data_from_pixels

    def test_rms_normalization_contract(self):
        """RMS normalization matches the explicit formula.

        rms_pix = 1 / sqrt(sum(1/noise_rms²)) over IFOs
        rms_ifo_pix = (1/noise_rms) * rms_pix
        """
        from pycwb.modules.likelihoodWP.pixel_data import (
            load_data_from_pixels, _load_data_from_pixel_arrays,
        )

        n_ifo, n_pix, tsize = 2, 3, 8
        pa = _make_synthetic_pa(n_ifo=n_ifo, n_pix=n_pix, tsize=tsize)
        rms, td00, td90, td_energy = load_data_from_pixels(None, n_ifo, pixel_arrays=pa)

        inv_rms = 1.0 / pa.noise_rms.astype(np.float64)
        rms_pix_expected = 1.0 / np.sqrt(np.sum(inv_rms ** 2, axis=0))
        rms_expected = (inv_rms * rms_pix_expected[np.newaxis, :]).astype(np.float32)

        np.testing.assert_allclose(rms, rms_expected, rtol=1e-5)

    def test_alias_returns_same_output(self):
        """load_data_from_pixels and _load_data_from_pixel_arrays via aliases."""
        from pycwb.modules.likelihoodWP.pixel_data import (
            load_data_from_pixels,
            extract_pixel_time_delay_data,
        )

        n_ifo, n_pix, tsize = 2, 3, 8
        pa = _make_synthetic_pa(n_ifo=n_ifo, n_pix=n_pix, tsize=tsize)

        r1 = load_data_from_pixels(None, n_ifo, pixel_arrays=pa)
        r2 = extract_pixel_time_delay_data(None, n_ifo, pixel_arrays=pa)

        for a, b in zip(r1, r2):
            np.testing.assert_array_equal(a, b)
