import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from pycwb.types import event_pixel_statistics as stats


def test_detector_bounds_undo_circular_lags_before_extrema():
    # A two-bin cluster crosses the shifted data edge in detector 1.
    # Its aligned support is [8, 9], inside analyzed interval [2, 10].
    pixels = SimpleNamespace(core=np.array([True, True]),
        rate=np.array([2., 2.]), layers=np.array([5, 5]),
        pixel_index=np.array([[16*5+1, 17*5+1], [19*5+1, 4*5+1]]))
    starts, stops = stats.aligned_pixel_bounds(pixels, 2, 2., 8., [0., 1.5])
    assert starts == [8., 8.]
    assert stops == [9., 9.]


def test_bounds_use_core_support_and_all_resolutions():
    pixels = SimpleNamespace(core=np.array([True, True, False]),
        rate=np.array([2., 4., 2.]), layers=np.array([5, 3, 5]),
        pixel_index=np.array([[16*5+1, 35*3+1, 4*5+1]]))
    assert stats.aligned_pixel_bounds(pixels, 1, 2., 8.) == ([8.], [9.])


def test_noise_zero_rms_stays_in_denominator():
    # The zero-RMS pixel contributes no squared noise but is still counted.
    expected = 10.**float(np.float32(np.log(2.)/2./np.log(10.)))
    assert stats.noise_amplitude(np.array([2., 0.])) == expected
    assert stats.noise_amplitude(np.array([0., -1.])) == 1.


def test_no_core_bounds_are_unavailable():
    pixels = SimpleNamespace(core=np.zeros(2, dtype=bool),
                              rate=np.ones(2), layers=np.ones(2))
    assert stats.aligned_pixel_bounds(pixels, 1, 2., 8.) is None


def test_padding_pixels_are_not_circularly_shifted():
    pixels = SimpleNamespace(core=np.array([True]), rate=np.array([2.]),
        layers=np.array([5]), pixel_index=np.array([[3*5+1], [3*5+1]]))
    assert stats.aligned_pixel_bounds(pixels, 2, 2., 8., [0., 1.5]) == ([1.5, 1.5], [2., 2.])
