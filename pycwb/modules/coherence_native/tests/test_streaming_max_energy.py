import numpy as np
import pytest
from wdm_wavelet.wdm import WDM
from pycwb.modules.coherence_native.time_delay_numba import _time_delay_max_energy_pattern_nb


@pytest.mark.parametrize("bounded", ["0", "1"])
@pytest.mark.parametrize("M", [8, 32])
@pytest.mark.parametrize("pattern", [1, 5, 9])
@pytest.mark.parametrize("max_delay", [0, 5])
def test_streaming_exact(monkeypatch, bounded, M, pattern, max_delay):
    monkeypatch.setenv("WDM_BOUNDED_NUMBA", bounded)
    w = WDM(M=M, K=M, beta_order=6, precision=10, backend="numba")
    signal = np.random.default_rng(73).normal(size=16387)
    args = (signal, M, w.m_H, w.filter, max_delay, 2, -1, pattern, 1.0, 1024, 0.0, 512.0, 512 / M)
    a = _time_delay_max_energy_pattern_nb(*args, mode="parallel")
    b = _time_delay_max_energy_pattern_nb(*args, mode="streaming")
    assert a.shape == b.shape and a.tobytes() == b.tobytes()
