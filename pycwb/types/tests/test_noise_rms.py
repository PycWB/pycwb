from types import SimpleNamespace
import pickle

import numpy as np
import pytest

from pycwb.types.noise_rms import NoiseRMSMap, make_noise_rms_map, lookup_pixel_noise_rms
from pycwb.types.pixel_arrays import PixelArrays


def noise_map(data, start=1000., anchor_start=1010., anchor_rate=.05, df=1.):
    return NoiseRMSMap(np.asarray(data, dtype=float), df, .125, start, 100,
                       {}, anchor_start, anchor_rate, start)


def test_anchor_metadata_is_not_the_wdm_time_spacing():
    tf = SimpleNamespace(data=np.empty((2,9760)), dt=.125, df=4.,
                         t0=1387221730., len_timeseries=9994240, wdm_params={})
    result = make_noise_rms_map(tf, np.ones((2,61)), 10.)
    assert result.noise_start == 1387221740.
    assert result.noise_rate == .05
    assert result.segment_start == tf.t0
    assert result.dt == tf.dt
    restored = pickle.loads(pickle.dumps(result))
    assert restored.noise_start == result.noise_start


def test_detector_lags_and_frequency_band_harmonic_variance():
    # Pixel at f=1, rate=4 covers noise layers 1 and 2. Two detectors
    # sample different 20-second anchors through their own lagged indices.
    noise = noise_map([[1,1,1], [2,6,10], [4,8,12], [1,1,1]])
    result = lookup_pixel_noise_rms([1], [[120,360]], 3, 4, [noise,noise])
    np.testing.assert_allclose(result[0], [np.sqrt(2/(1/4+1/16)), np.sqrt(2/(1/36+1/64))])


def test_anchor_boundary_and_last_anchor_are_not_arbitrary_clipping():
    noise = noise_map([[1,1,1], [2,3,4], [1,1,1]])
    # Rate=2, layers=3: offsets 29,30,50,70 seconds. At 70 the
    # cWB one-step endpoint adjustment selects the final anchor.
    result = lookup_pixel_noise_rms([1]*4, np.array([[174],[180],[300],[420]]), 3, 2, [noise])
    np.testing.assert_allclose(result[:,0], [2,3,4,4])
    with pytest.raises(ValueError, match='outside'):
        lookup_pixel_noise_rms([1], [[540]], 3, 2, [noise])


def test_mixed_resolution_pixel_arrays_use_detector_indices():
    noise = noise_map(np.array([[1,1,1],[2,3,4],[4,6,8],[8,9,10],[10,11,12]]))
    pa = PixelArrays.from_arrays(
        time=np.array([999,999]), frequency=np.array([1,2]),
        layers=np.array([3,5]), rate=np.array([2.,4.]),
        core=np.ones(2,dtype=bool), likelihood=np.ones(2), null=np.zeros(2),
        noise_rms=np.ones((1,2)), pixel_index=np.array([[60,600]]),
        asnr=np.ones((1,2)), n_ifo=1,
    )
    pa.populate_noise_rms([noise])
    np.testing.assert_allclose(pa.noise_rms[0], [2.,np.sqrt(2/(1/81+1/121))])


def test_invalid_maps_and_detector_shapes_are_rejected():
    noise = noise_map(np.ones((3,3)))
    with pytest.raises(ValueError, match='agree'):
        lookup_pixel_noise_rms([1], [[60,60]], 3, 2, [noise])
    noise.data[1,0] = 0
    with pytest.raises(ValueError, match='positive'):
        lookup_pixel_noise_rms([1], [[60]], 3, 2, [noise])


def test_legacy_pixel_list_uses_same_lagged_mixed_resolution_lookup():
    from pycwb.types.network_cluster import Cluster
    from pycwb.modules.likelihoodWP.likelihood_setup import populate_pixel_noise_from_maps
    noise = noise_map(np.array([[1,1,1],[2,3,4],[4,6,8],[8,9,10],[10,11,12]]))
    arrays = PixelArrays.from_arrays(
        time=np.array([999,999]), frequency=np.array([1,2]),
        layers=np.array([3,5]), rate=np.array([2.,4.]),
        core=np.ones(2,dtype=bool), likelihood=np.ones(2), null=np.zeros(2),
        noise_rms=np.ones((2,2)), pixel_index=np.array([[60,600],[180,200]]),
        asnr=np.ones((2,2)), n_ifo=2,
    )
    pixels = Cluster(pixel_arrays=arrays).pixels
    populate_pixel_noise_from_maps(pixels, [noise, noise])
    expected = [[2., np.sqrt(2/(1/81+1/121))], [3., np.sqrt(2/(1/64+1/100))]]
    for detector in range(2):
        np.testing.assert_allclose([p.data[detector].noise_rms for p in pixels], expected[detector])
