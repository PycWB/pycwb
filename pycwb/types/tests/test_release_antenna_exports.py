"""Exported antenna values use release Float_t angles, then double arithmetic."""

from pathlib import Path
from pycwb.config import Config
import json
import numpy as np
from pycwb.types.network_cluster import Cluster, ClusterMeta
from pycwb.types.network_event import Event
from pycwb.types.job import WaveSegment
from pycwb.types.pixel_arrays import PixelArrays


def test_all_observed_release_antenna_exports():
    job = WaveSegment(
        index=1,
        ifos=["L1", "H1"],
        analyze_start=1387221740,
        analyze_end=1387222940,
        sample_rate=8192,
        seg_edge=10,
        shift=None,
    )
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
    config = _config(
        ifo=["L1", "H1"], nIFO=2, TDRate=32768, inRate=16384, levelR=1, detector_geometry={"H1": "H1:cwb", "L1": "L1:cwb"}, pattern=10, optim=False
    )
    count = 0
    rows = json.loads((Path(__file__).with_name("reference") / "release_antenna_exports.json").read_text())
    for row in rows:
        angles = np.array([row["theta"][0], row["phi"][0], row["psi"][0]], dtype=np.float32)
        # Model the higher precision internal map coordinate: the values
        # narrow back to the actual exported Float_t values.
        internal = angles.astype(np.float64) + np.spacing(angles).astype(np.float64) / 4
        np.testing.assert_array_equal(internal.astype(np.float32), angles)
        meta = ClusterMeta()
        meta.reconstructed_theta = float(internal[0])
        meta.reconstructed_phi = float(internal[1])
        meta.psi = float(internal[2])
        event = Event()
        event.output_py(job, Cluster(pixel_arrays=pixels, cluster_meta=meta), config)
        np.testing.assert_array_equal(np.array(event.bp, dtype=np.float32), np.array(row["bp"][:2], dtype=np.float32))
        np.testing.assert_array_equal(np.array(event.bx, dtype=np.float32), np.array(row["bx"][:2], dtype=np.float32))
        count += 1
    assert count == 300


def _config(**kwargs):
    result = Config()
    result.load_from_dict(kwargs)
    return result
