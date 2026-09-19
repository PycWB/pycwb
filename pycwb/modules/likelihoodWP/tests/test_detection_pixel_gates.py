"""cWB distinguishes the outer writeback gate from cross-talk membership."""
from types import SimpleNamespace

import numpy as np
import pytest

from pycwb.modules.likelihoodWP import detection_statistics as detection


def test_satellite_is_neighbor_but_not_writeback_target(monkeypatch):
    class PixelArrays:
        core = np.array([True, True, True, False, True])

        def __len__(self):
            return len(self.core)

        def set_waveform_data(self, **kwargs):
            pass

    arrays = np.ones((2, 5), dtype=np.float32)
    sky = SimpleNamespace(**{name: arrays for name in [
        'energy_array_plus', 'energy_array_cross', 'pd', 'pD', 'ps', 'pS',
        'noise_amplitude_00', 'noise_amplitude_90', 'S_snr']})
    sky.pixel_mask = np.ones(5)
    sky.gaussian_noise_correction = np.array([1., 0., 1., 1., -1.])
    sky.coherent_energy = np.array([1., 1., 0., 1., 1.])
    sky.Rc = sky.Gn = sky.Np = sky.N_pix_effective = 1.

    class GatesChecked(Exception):
        pass

    def check_kernel(null_indices, like_indices, *args):
        np.testing.assert_array_equal(null_indices, [0, 2])
        np.testing.assert_array_equal(like_indices, [0])
        null_mask, like_mask = args[-4:-2]
        np.testing.assert_array_equal(null_mask, [True, False, True, False, False])
        np.testing.assert_array_equal(like_mask, [True, True, False, False, True])
        raise GatesChecked

    # Stop at the kernel boundary; waveform reconstruction is unrelated here.
    monkeypatch.setattr(detection, '_compute_null_likelihood_numba', check_kernel)
    with pytest.raises(GatesChecked):
        detection.populate_detection_statistics(
            sky, SimpleNamespace(), SimpleNamespace(pixel_arrays=PixelArrays()),
            2, None, 0., config=SimpleNamespace(),
            cluster_xtalk=np.zeros((0, 8)),
            cluster_xtalk_lookup=np.zeros((5, 2), dtype=np.int64))
