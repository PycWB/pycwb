"""Test bounded transforms when embedded inside the actual delay-loop JIT."""
import numpy as np
import pytest
from wdm_wavelet.wdm import WDM
from pycwb.modules.coherence_native.time_delay_jax import _time_delay_max_energy_pattern_jit

@pytest.mark.parametrize('M',[8,16,128])
@pytest.mark.parametrize('pattern',[1,5,9])
def test_delay_loop_tiles_exact(M,pattern):
    w=WDM(M=M,K=M,beta_order=6,precision=10)
    tile=max(32,(262144//M)//32*32)
    signal=np.random.default_rng(117).normal(size=(tile+32)*M)
    args=(signal,np.float64(8192),np.float64(0),np.int32(8),np.int32(16),np.asarray(w.filter),-1,pattern,1.,8192//M,512.,4096.,4096./M,(M+1,len(signal)//M),M,int(w.m_H))
    a=np.asarray(_time_delay_max_energy_pattern_jit(*args,bounded=False))
    b=np.asarray(_time_delay_max_energy_pattern_jit(*args,bounded=True))
    c=np.asarray(_time_delay_max_energy_pattern_jit(*args,bounded=False))
    assert a.tobytes()==b.tobytes()==c.tobytes()
