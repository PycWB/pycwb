import numpy as np
import pytest
from pycwb.modules.reconstruction.injection_timing import network_reference_times


def test_cwb_energy_squared_weighting_and_geometric_offsets():
    # Reference injection::fill_in: weights are ISNR^2, where ISNR=SNR^2.
    times=network_reference_times([100.,104.],[1.,3.],[.002,-.006])
    np.testing.assert_allclose(times,[103.6,103.592],rtol=0,atol=1e-13)


def test_captured_hf_source_72_time_cut():
    times=network_reference_times([1387222614.005074,1387222614.0188656],
        [141.0216,829.3167],[0.,-.0086482])
    cwb=[1387222614.0184872,1387222614.009839]
    np.testing.assert_allclose(times,cwb,rtol=0,atol=1e-5)
    reconstructed=1387222614.113037
    assert reconstructed-1387222614.005074>.1
    assert reconstructed-times[0]<.1

@pytest.mark.parametrize('energy',[[0.,0.],[-1.,3.],[float('nan'),3.]])
def test_invalid_energy_is_not_silently_used(energy):
    with pytest.raises(ValueError):network_reference_times([100.,101.],energy,[0.,.001])
