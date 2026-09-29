from types import SimpleNamespace

import numpy as np
import pytest

from pycwb.modules.read_data.data_check import check_and_resample, check_and_resample_py
from pycwb.types.time_series import TimeSeries


@pytest.mark.parametrize("resample", [check_and_resample, check_and_resample_py])
@pytest.mark.parametrize(
    "in_rate, f_resample, level, expected_rate",
    [
        pytest.param(4096, 16384, 2, 4096, id="target-equals-input-rate"),
        pytest.param(4096, 0, 2, 1024, id="level-only"),
        pytest.param(16384, 4096, 2, 1024, id="two-downsampling-steps"),
        pytest.param(4096, 16384, 0, 16384, id="resample-only"),
        pytest.param(4096, 0, 0, 4096, id="no-resampling"),
    ],
)
def test_resampling_rate_and_signal(resample, in_rate, f_resample, level, expected_rate):
    config = SimpleNamespace(inRate=in_rate, fResample=f_resample, levelR=level, dcCal=[1.0])
    # A one-second, periodic tone well below every output Nyquist frequency.
    frequency = 64.0
    signal = TimeSeries(
        data=np.sin(2 * np.pi * frequency * np.arange(in_rate) / in_rate),
        t0=1234567890.0,
        dt=1.0 / in_rate,
    )

    result = resample(signal, config, 0)

    assert result.sample_rate == expected_rate
    assert len(result.data) == expected_rate
    assert result.t0 == signal.t0
    assert result.duration == 1.0
    # Preserve cWB's sqrt(2**levelR) normalization after reducing the rate.
    expected = np.sqrt(2**level) * np.sin(
        2 * np.pi * frequency * np.arange(expected_rate) / expected_rate
    )
    np.testing.assert_allclose(result.data, expected, rtol=1e-12, atol=1e-12)
