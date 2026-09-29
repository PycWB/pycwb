"""Cache storage and lifetime changes must preserve every stored bit."""

from pycwb.config.processing import ExecutionProfile

from types import SimpleNamespace
import numpy as np
import pytest
from pycwb.types.time_frequency_map import TimeFrequencyMap
from pycwb.utils.td_vector_batch import _build_td_inputs_single_level


def check_equal(a, b):
    for name in ("padded00", "padded90", "T0", "Tx"):
        x, y = getattr(a, name), getattr(b, name)
        assert x.dtype == y.dtype and x.shape == y.shape
        assert x.tobytes() == y.tobytes(), name
        assert y.flags.c_contiguous
    assert (a.M, a.n_coeffs, a.J) == (b.M, b.n_coeffs, b.J)


@pytest.mark.parametrize("layout", ["C", "F", "sliced", "reverse"])
@pytest.mark.parametrize("padding", [0, 1, 12])
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_direct_padding_exact(monkeypatch, layout, padding, dtype):
    rng = np.random.default_rng(741)
    data = (rng.normal(size=(17, 64)) + 1j * rng.normal(size=(17, 64))).astype(dtype)
    # Signed zeros, subnormal values and exact float32 rounding boundaries.
    data.real[0, :4] = [-0.0, 0.0, np.nextafter(np.float32(0), np.float32(1)), 1.0 + 2**-24]
    if layout == "F":
        data = np.asfortranarray(data)
    elif layout == "sliced":
        data = data[:, ::2]
    elif layout == "reverse":
        data = data[:, ::-1]
    tf = TimeFrequencyMap(data, True, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, None, None)
    filters = SimpleNamespace(
        n_coeffs=padding,
        M=16,
        max_delay=64,
        T0=rng.normal(size=(129, 2 * padding + 1)),
        Tx=rng.normal(size=(129, 2 * padding + 1)),
    )
    a = tf.prepare_td_inputs(filters)
    b = tf.prepare_td_inputs(filters, compact=True)
    check_equal(a, b)
    assert not np.shares_memory(data, b.padded00)
    assert not np.shares_memory(data, b.padded90)


@pytest.mark.parametrize("level", [3, 6, 9])
@pytest.mark.parametrize("up", [1, 4])
def test_real_wdm_cache_exact(monkeypatch, level, up):
    rng = np.random.default_rng(87)
    config = SimpleNamespace(WDM_beta_order=6, WDM_precision=10, TDSize=12, nIFO=2)
    strains = [SimpleNamespace(data=rng.normal(size=32768), sample_rate=4096, t0=1000.0) for _ in range(2)]
    ka, a = _build_td_inputs_single_level(level, config, strains, up)
    config.execution_profile = ExecutionProfile(compact_td_cache=True)
    kb, b = _build_td_inputs_single_level(level, config, strains, up)
    assert ka == kb
    for x, y in zip(a, b):
        check_equal(x, y)
