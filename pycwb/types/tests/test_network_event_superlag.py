"""Native event output supports jobs without a superlag vector."""
import numpy as np
import pytest

from pycwb.types.job import WaveSegment
from pycwb.types.network_cluster import Cluster
from pycwb.types.network_event import Event
from pycwb.types.pixel_arrays import PixelArrays


@pytest.mark.parametrize('shift', [None, [0.0, 0.0], np.array([0.0, 1200.0])])
def test_output_py_superlag(shift):
    job = WaveSegment(index=1, ifos=['L1', 'H1'], analyze_start=1387221740,
                      analyze_end=1387222940, sample_rate=4096, seg_edge=10, shift=shift)
    pixels = PixelArrays.from_arrays(
        time=np.array([160, 176]), frequency=np.array([2, 2]),
        layers=np.array([16, 16]), rate=np.array([32., 32.]),
        noise_rms=np.ones((2, 2)), pixel_index=np.array([[160, 176], [160, 176]]),
        n_ifo=2, core=np.ones(2, dtype=bool), likelihood=np.ones(2),
    )
    event = Event()
    event.output_py(job, Cluster(pixel_arrays=pixels))
    assert event.slag == ([0.0, 0.0] if shift is None else list(shift))
