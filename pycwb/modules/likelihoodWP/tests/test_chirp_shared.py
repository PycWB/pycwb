"""Compare the shared host algorithm to the frozen independent release loop."""
import numpy as np
import pytest
from pycwb.modules.likelihoodWP.chirp_micropixel import _bootstrap, root_uniforms
from pycwb.modules.likelihoodWP.tests.chirp_bootstrap_reference import _bootstrap as reference


@pytest.mark.parametrize('count', [13, 32, 128])
@pytest.mark.parametrize('seed', [1, 42, 7123])
@pytest.mark.parametrize('stream_size', [20, 32768])
def test_fixed_stream_is_bitwise_identical(count, seed, stream_size):
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0, 2, count))
    f = rng.uniform(20, 300, count)
    energy = rng.uniform(.1, 5, count)
    args = x, f, energy, 1 / 64, root_uniforms(seed, stream_size)
    actual, consumed = _bootstrap(*args)
    expected, expected_consumed = reference(*args)
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
    assert consumed == expected_consumed
