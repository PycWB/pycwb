"""Expected values are exported by original cWB GetQveto, not a Python copy."""
from pathlib import Path
import numpy as np
import pytest
from pycwb.modules.qveto.qveto import get_qveto, _reference_upsample, _zero_crossing_segment_maxima

@pytest.mark.parametrize('index',range(5))
def test_get_qveto_against_original_cwb(index):
    with np.load(Path(__file__).with_name('data')/'cwb_qveto_fixture.npz') as ref:
        np.testing.assert_allclose(get_qveto(ref[f'waveform_{index}']),ref['expected'][index,1:],
            rtol=2e-7,atol=1e-7)
        # The reference upsampling has a global factor four, which cancels in Q.
        np.testing.assert_allclose(_reference_upsample(ref[f'waveform_{index}']),
            ref[f'upsampled_{index}']/4,rtol=0,atol=1e-14)


def test_zero_crossing_endpoints_match_reference_loop():
    # Zero does not count as a sign change. The crossing sample belongs to the
    # preceding peak, and the unfinished trailing peak must not be included.
    x=np.array([100.,1.,-2.,0.,-3.,4.,5.,-6.,100.])
    np.testing.assert_array_equal(_zero_crossing_segment_maxima(x),[2.,4.,6.,100.])

@pytest.mark.parametrize('wf',[[],[1.],np.ones(32),np.zeros(32)])
def test_get_qveto_returns_zero_without_valid_peaks(wf):
    assert get_qveto(wf)==(0.,0.)
