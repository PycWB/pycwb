from dataclasses import replace

import numpy as np
import pytest

from pycwb.types.time_frequency_map import TimeFrequencyMap
from pycwb.modules.conditioning_plugins.noise_variation import apply_noise_variation


def variation_map():
    return TimeFrequencyMap(
        data=np.full((1,4096),.5,dtype=np.float32), is_whitened=False,
        dt=1/64., df=32., start=1000., stop=1064.,
        f_low=16., f_high=48., edge=10., wavelet=None,
    )

@pytest.mark.parametrize('changes',[
    {'data':np.ones(4096)}, {'data':np.ones((2,4096))}, {'data':np.ones((1,0))},
    {'dt':0.}, {'dt':float('nan')}, {'start':float('inf')},
    {'f_low':None}, {'f_high':16.}, {'f_high':float('nan')},
    {'data':np.zeros((1,4096))}, {'data':np.full((1,4096),float('nan'))},
    {'data':np.ones((1,4096),dtype=complex)},
])
def test_invalid_variation_maps_are_rejected(changes):
    variation = replace(variation_map(),**changes)
    with pytest.raises(ValueError,match='(noise-variation|Noise variation)'):
        apply_noise_variation(np.array([2.]),np.array([1]),np.array([3840]),
                              np.array([3]),np.array([64.]),1000.,variation)
