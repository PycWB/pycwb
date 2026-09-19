"""Installed cWB 6.4.6.9 core=false moment fixtures; no ROOT at test time."""

import json
from pathlib import Path
import numpy as np
import pytest
from pycwb.types.event_pixel_statistics import pixel_moments
from pycwb.types.job import WaveSegment
from pycwb.types.network_cluster import Cluster
from pycwb.types.network_event import Event
from pycwb.types.pixel_arrays import PixelArrays


@pytest.mark.parametrize(
    "case",
    json.loads((Path(__file__).parent / "reference/pixel_moment_oracle.json").read_text()),
    ids=lambda case: case["name"],
)
def test_release_pixel_moments_include_halo_resolution(case):
    args = [np.array(case[k]) for k in ["time", "frequency", "layers", "pixel_rate", "likelihood"]]
    actual = pixel_moments(*args, case["rate"])
    expected = [case["expected"][k] for k in ["freq", "duration", "bandwidth"]]
    np.testing.assert_array_equal(actual, expected)
    pixels = PixelArrays.from_arrays(
        time=args[0],
        frequency=args[1],
        layers=args[2],
        rate=args[3],
        likelihood=args[4],
        core=np.array(case["core"]),
        n_ifo=2,
        noise_rms=np.ones((2, len(args[0]))),
        pixel_index=np.tile(args[0], (2, 1)),
    )
    job = WaveSegment(
        index=1,
        ifos=["L1", "H1"],
        analyze_start=1387221740,
        analyze_end=1387222940,
        sample_rate=case["rate"],
        seg_edge=10,
    )
    event = Event()
    event.output_py(job, Cluster(pixel_arrays=pixels))
    np.testing.assert_array_equal([event.frequency[1], event.duration[0], event.bandwidth[0]], expected)


def test_empty_pixel_moments_are_unavailable():
    integers = np.array([], dtype=np.int64)
    floats = np.array([], dtype=np.float64)
    assert pixel_moments(integers, integers, integers, floats, floats, 256.0) == (-1.0, -1.0, -1.0)
