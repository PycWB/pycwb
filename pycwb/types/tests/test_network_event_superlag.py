"""Native event output supports jobs without a superlag vector."""

from types import SimpleNamespace

import numpy as np
import pytest

from pycwb.modules.job_segment.dq_segment import get_job_list
from pycwb.types.job import WaveSegment
from pycwb.types.network_cluster import Cluster
from pycwb.types.network_event import Event
from pycwb.types.pixel_arrays import PixelArrays


@pytest.fixture
def cluster():
    pixels = PixelArrays.from_arrays(
        time=np.array([160, 176]),
        frequency=np.array([2, 2]),
        layers=np.array([16, 16]),
        rate=np.array([32.0, 32.0]),
        noise_rms=np.ones((2, 2)),
        pixel_index=np.array([[160, 176], [160, 176]]),
        n_ifo=2,
        core=np.ones(2, dtype=bool),
        likelihood=np.ones(2),
    )
    return Cluster(pixel_arrays=pixels)


@pytest.mark.parametrize("shift", [None, [0.0, 0.0], [0.0, 1200.0], np.array([0.0, 1200.0])])
def test_output_py_superlag(shift, cluster):
    job = WaveSegment(
        index=1,
        ifos=["L1", "H1"],
        analyze_start=1387221740,
        analyze_end=1387222940,
        sample_rate=4096,
        seg_edge=10,
        shift=shift,
    )
    event = Event()
    event.output_py(job, cluster)
    assert event.slag == ([0.0, 0.0] if shift is None else list(shift))
    assert event.slag is not shift


def test_output_py_default_job_shift(cluster):
    jobs = get_job_list(
        ifos=["L1", "H1"],
        dq_list=([1387221730], [1387222950]),
        seg_len=1200,
        seg_mls=300,
        seg_edge=10,
        sample_rate=4096,
    )
    assert len(jobs) == 1
    job = jobs[0]
    assert isinstance(job, WaveSegment)
    assert job.shift is None

    event = Event()
    event.output_py(job, cluster)

    assert event.slag == [0.0, 0.0]
    assert job.shift is None


def test_output_py_missing_shift(cluster):
    job = SimpleNamespace(
        index=1,
        ifos=["L1", "H1"],
        physical_padded_starts={"L1": 1387221730, "H1": 1387221730},
        padded_duration=1220,
        sample_rate=4096,
        seg_edge=10,
    )
    event = Event()
    event.output_py(job, cluster)

    assert event.slag == [0.0, 0.0]
