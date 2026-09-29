"""Reference detector::setsim values test the scaling SNR convention."""
from pathlib import Path
import numpy as np
import pytest
from pycwb.types.time_series import TimeSeries
from pycwb.modules.injection.snr_energy import cwb_snr_energy


@pytest.mark.parametrize('detector', ['L1', 'H1'])
@pytest.mark.parametrize('source', [0, 1, 19, 41])
def test_energy_matches_cwb_setsim_fixture(detector, source):
    with np.load(Path(__file__).parent/'data/cwb_snr_energy_fixture.npz') as fixture:
        key = f'{detector}_{source}'
        t0, rate, gps, expected, half_window, f_low, f_high = fixture[key+'_meta']
        signal = TimeSeries(fixture[key+'_strain'], dt=1/rate, t0=t0)
    actual = cwb_snr_energy(signal, gps, half_window, f_low, f_high)
    assert actual == pytest.approx(expected, rel=2e-10)


def test_zero_signal_has_zero_energy():
    signal = TimeSeries(np.zeros(4096), dt=1/1024, t0=100.)
    assert cwb_snr_energy(signal, 102., 1., 16., 512.) == 0.
