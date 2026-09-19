"""Regression for the release's distinct total-energy correlation offset.

Golden rows were evaluated by the installed cWB 6.4.6.9 watavx.hh kernel
with XIFO=2, four padded pixels, and one active pixel. A unit-amplitude
perfectly reconstructed pixel must fall below the 0.5 sky-correlation cut.
"""

import numpy as np
import pytest
from pycwb.modules.likelihoodWP.sky_stat import avx_stat_ps
from pycwb.modules.likelihoodWP.sky_scratch import avx_stat_ps_into


@pytest.mark.parametrize("scratch", [False, True])
@pytest.mark.parametrize(
    "amplitude,expected",
    [
        (1.0, [0.49975013732910156, 1.0, 1.0, 1.0]),
        (0.03125, [0.00064541347092017531, 0.0009768183808773756, 1.0, 1.0]),
        (8.0, [0.98460769653320312, 64.0, 1.0, 1.0]),
    ],
)
def test_release_correlation_energy_offset(scratch, amplitude, expected):
    data = np.array([[amplitude, 0.0, 0.0, 0.0], [amplitude, 0.0, 0.0, 0.0]], dtype=np.float32)
    quad = np.zeros_like(data)
    sine = np.zeros(4, dtype=np.float32)
    cosine = np.ones(4, dtype=np.float32)
    mask = np.array([1.0, -1.0, -1.0, -1.0], dtype=np.float32)
    args = (data, quad, data, quad, sine, cosine, mask)
    if scratch:
        actual = avx_stat_ps_into(*args, tuple(np.empty(4, dtype=np.float32) for _ in range(3)))
    else:
        actual = avx_stat_ps(*args)
    np.testing.assert_array_equal(actual[:4], expected)
    if amplitude == 1.0:
        assert actual[0] < 0.5
