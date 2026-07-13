import numpy as np
import pytest

from pycwb.modules.cwb_results import CwbLiveRoot


def test_cwb_live_root_summary_and_slag_groups(tmp_path):
    uproot = pytest.importorskip("uproot")

    root_file = tmp_path / "live.root"
    lag = np.full((3, 9), -1.0, dtype=np.float32)
    lag[:, :3] = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 1.0],
        [0.0, 0.0, 0.0],
    ], dtype=np.float32)
    slag = np.full((3, 9), -1.0, dtype=np.float32)
    slag[:, :3] = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 1200.0, 1.0],
        [0.0, 1200.0, 1.0],
    ], dtype=np.float32)
    start = np.zeros((3, 8), dtype=np.float64)
    stop = np.zeros((3, 8), dtype=np.float64)
    start[:, :2] = np.array([
        [100.0, 100.0],
        [200.0, 1400.0],
        [300.0, 1500.0],
    ])
    stop[:, :2] = start[:, :2] + 1200.0

    with uproot.recreate(root_file) as root:
        root["liveTime"] = {
            "run": np.array([1, 1, 2], dtype=np.int32),
            "gps": np.array([100.0, 200.0, 300.0]),
            "live": np.array([1200.0, 1190.0, 0.0]),
            "lag": lag,
            "slag": slag,
            "start": start,
            "stop": stop,
        }

    reader = CwbLiveRoot(root_file)
    summary = reader.summary()

    assert reader.entries == 3
    assert summary.entries == 3
    assert summary.total_live_seconds == 2390.0
    assert summary.nominal_seconds == 3600.0
    assert summary.loss_seconds == 1210.0
    assert summary.live_eq_1200_count == 1
    assert summary.live_lt_1200_count == 2
    assert summary.live_zero_count == 1

    by_pair = reader.slag_summary(by="pair")
    pair_groups = {row["key"]: row for row in by_pair}
    assert pair_groups[(0, 1200)]["rows"] == 2
    assert pair_groups[(0, 1200)]["live_seconds"] == 1190.0

    by_id = reader.slag_summary(by="id")
    assert [row["key"] for row in by_id] == [0, 1]
