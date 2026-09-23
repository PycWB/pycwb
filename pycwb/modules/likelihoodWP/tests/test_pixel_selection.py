import numpy as np
from pycwb.types.pixel_arrays import PixelArrays
from pycwb.types.network_cluster import Cluster
from pycwb.modules.likelihoodWP.pixel_selection import select_likelihood_pixels, restore_likelihood_pixels

def cluster():
    return Cluster(pixel_arrays=PixelArrays.from_arrays(time=np.arange(6),frequency=np.ones(6),layers=np.full(6,16),rate=np.full(6,512.),noise_rms=np.ones((2,6)),pixel_index=np.tile(np.arange(6),(2,1)),n_ifo=2,core=np.ones(6,dtype=bool),likelihood=np.array([3,1,9,5,9,2]),td_amp_dense=np.arange(48,dtype=np.float32).reshape(6,2,4)))

def test_loudest_pixels_preserve_original_order_and_td_alignment():
    original=cluster();selected,state=select_likelihood_pixels(original,3)
    np.testing.assert_array_equal(selected.pixel_arrays.time,[2,3,4])
    np.testing.assert_array_equal(selected.pixel_arrays.td_amp_dense(),original.pixel_arrays.td_amp_dense()[[2,3,4]])
    assert len(original.pixel_arrays)==6

def test_unlimited_and_small_clusters_preserve_identity():
    for limit in [0,6,10]:
        original=cluster();selected,state=select_likelihood_pixels(original,limit)
        assert selected is original and state is None

def test_restoring_volume_does_not_restore_excluded_signal_pixels():
    original=cluster();fitted,state=select_likelihood_pixels(original,3)
    fitted.pixel_arrays.core[:]=[True,False,True]
    fitted.pixel_arrays.likelihood[:]=[4,-2,5]
    fitted.pixel_arrays.null[:]=[.1,0,.2]
    fitted.pixel_arrays.asnr[:]=3
    restored=restore_likelihood_pixels(fitted,state)
    assert len(restored.pixel_arrays)==6
    np.testing.assert_array_equal(np.flatnonzero(restored.pixel_arrays.core),[2,4])
    np.testing.assert_array_equal(restored.pixel_arrays.likelihood,[0,0,4,-2,5,0])
    assert not restored.pixel_arrays.null[[0,1,5]].any()
