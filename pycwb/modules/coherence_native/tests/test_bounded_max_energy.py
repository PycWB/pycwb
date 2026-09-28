import numpy as np
import pytest
from wdm_wavelet.wdm import WDM
from pycwb.modules.coherence_native.time_delay_numba import _time_delay_max_energy_pattern_nb


@pytest.mark.parametrize("mode", ["parallel", "time-major"])
@pytest.mark.parametrize("M", [8, 32])
@pytest.mark.parametrize("pattern", [1, 2])
def test_bounded_max_energy_exact(monkeypatch, mode, M, pattern):
    w = WDM(M=M, K=M, beta_order=6, precision=10, backend="numba")
    signal = np.random.default_rng(432).normal(size=16384)
    args = (signal, M, w.m_H, w.filter, 3, 1, -1, pattern, 1.0, 1024, 0.0, 512.0, 512 / M)
    outputs = []
    # Switch back in the same process to test both compiled specializations.
    for enabled in ["0", "1", "0"]:
        outputs.append(_time_delay_max_energy_pattern_nb(*args, mode=mode, bounded=enabled == "1"))
    assert outputs[0].tobytes() == outputs[1].tobytes() == outputs[2].tobytes()
