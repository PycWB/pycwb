"""ROOT contracts including empty exposure, superlags and strict thresholds."""

import numpy as np
import pandas as pd
import pytest

uproot = pytest.importorskip("uproot")
from pycwb.modules.postprocess.background import process_background
from pycwb.modules.postprocess.root_adapter import import_cwb_root, read_cwb_root
from pycwb.modules.postprocess.selection import trigger_selection


def fixture(path, *, duplicate=False, empty=False):
    # Third exposure has no event; fourth is lag-zero but nonzero superlag.
    lag = np.array([[0, 0, 0], [0.5, 0, 1], [1, 0, 2], [0, 0, 0]], dtype="f4")
    slag = np.array([[0, 0, 0]] * 3 + [[0, 100, 1]], dtype="f4")
    event_indices = [] if empty else [0, 1, 3]
    n = len(event_indices)
    wave = {
        "run": np.ones(n, dtype="i4"),
        "ndim": np.full(n, 2, dtype="i4"),
        "nevent": np.ones(n, dtype="i4"),
        "eventID": np.ones((n, 2), dtype="i4"),
        "rho": np.array([[4, 40], [5, 50], [6, 60]], dtype="f4")[:n],
        "netcc": np.full((n, 4), 0.8, dtype="f4"),
        "time": np.full((n, 4), 1387222477.6453857, dtype="f8"),
        "lag": lag[event_indices],
        "slag": slag[event_indices],
        "Qveto": np.ones((n, 4), dtype="f4"),
        "Lveto": np.full((n, 3), 7, dtype="f4"),
    }
    live = {
        "run": np.ones(4, dtype="i4"),
        "live": np.array([10, 20, 30, 40], dtype="f8"),
        "lag": lag,
        "slag": slag,
    }
    if duplicate:
        live["lag"][2] = live["lag"][1]
    with uproot.recreate(path) as f:
        f.mktree("waveburst", wave)
        f.mktree("liveTime", live)


def test_adapter_common_processing_and_native_selection(tmp_path):
    path = tmp_path / "wave.root"
    fixture(path)
    result = read_cwb_root(path, ["L1", "H1"], batch_size=1)
    events = result.triggers.to_pandas()
    assert events.id.nunique() == 3  # eventID/nevent reused across lags/superlags
    assert events.lag_idx.tolist() == [0, 1, 0]
    assert events.root_run.tolist() == [1, 1, 1]
    assert events.job_id.tolist() == [1, 1, 2]
    assert events.Lveto2.tolist() == [7, 7, 7]
    assert events.Qveto3.tolist() == [1, 1, 1]
    assert events.time_L1.iloc[0] == 1387222477.6453857
    paths = result.write(tmp_path / "converted")
    kwargs = {"thresholds": [5, 6], "comparison": ">"}
    memory = process_background(result.triggers, result.progress, **kwargs)
    disk = process_background(paths["catalog_file"], paths["progress_file"], **kwargs)
    assert memory["livetime"] == 90  # includes zero-event lag and shifted lag-zero
    assert memory["curve"]["count"].tolist() == [1, 0]
    pd.testing.assert_frame_equal(memory["curve"], disk["curve"])
    inclusive = process_background(result.triggers, result.progress, thresholds=[5, 6])
    assert inclusive["curve"]["count"].tolist() == [2, 1]
    selection = trigger_selection(".", paths["catalog_file"], paths["progress_file"])
    assert selection["livetime"]["seconds"] == 90
    assert len(selection["triggers"]) == 2


def test_no_events_still_retains_exposure(tmp_path):
    path = tmp_path / "empty.root"
    fixture(path, empty=True)
    result = read_cwb_root(path, ["L1", "H1"])
    out = process_background(result.triggers, result.progress, thresholds=[5])
    assert out["livetime"] == 90
    assert out["curve"]["count"].tolist() == [0]


def test_duplicate_exposure_rejected(tmp_path):
    path = tmp_path / "duplicate.root"
    fixture(path, duplicate=True)
    with pytest.raises(ValueError, match="Duplicate liveTime"):
        read_cwb_root(path, ["L1", "H1"])


def test_wrong_network_and_repeated_files_rejected(tmp_path):
    path = tmp_path / "wave.root"
    fixture(path)
    with pytest.raises(ValueError, match="ndim"):
        read_cwb_root(path, ["H1"])
    with pytest.raises(ValueError, match="duplicates"):
        read_cwb_root([path, path], ["L1", "H1"])


def test_workflow_action(tmp_path):
    fixture(tmp_path / "wave.root")
    result = import_cwb_root(tmp_path, "wave.root", ["L1", "H1"], output_dir="adapted")
    assert len(result["triggers"]) == 3
    assert len(pd.read_parquet(result["progress_file"])) == 4


def test_missing_measurements_are_not_fabricated(tmp_path):
    path = tmp_path / "wave.root"
    fixture(path)
    result = read_cwb_root(path, ["L1", "H1"])
    assert result.triggers["coherent_energy"].null_count == 3
    assert result.triggers["noise_rms_L1"].null_count == 3
    with pytest.raises(ValueError, match="Nonfinite ranking"):
        process_background(
            result.triggers,
            result.progress,
            ranking_par="coherent_energy",
            thresholds=[0],
        )


def test_missing_exposure_and_invalid_livetime_fail(tmp_path):
    path = tmp_path / "wave.root"
    fixture(path)
    result = read_cwb_root(path, ["L1", "H1"])
    events = result.triggers.to_pandas()
    live = result.progress.to_pandas()
    with pytest.raises(ValueError, match="without completed"):
        process_background(events, live.iloc[1:], thresholds=[5])
    live.loc[0, "livetime"] = np.nan
    with pytest.raises(ValueError, match="finite and nonnegative"):
        process_background(events, live, thresholds=[5])


def test_root_noise_precision_survives_canonical_catalog(tmp_path):
    path=tmp_path/'wave.root'
    fixture(path)
    # Rebuild the tree with a ROOT double RMS, beyond float32 precision.
    with uproot.open(path) as root:
        wave=root['waveburst'].arrays(library='np')
        live=root['liveTime'].arrays(library='np')
    noise=np.full((3,2),1.234567890123e-24,dtype='f8')
    wave['noise']=noise
    with uproot.recreate(path) as root:
        root.mktree('waveburst',wave)
        root.mktree('liveTime',live)
    result=read_cwb_root(path,['L1','H1']).triggers.to_pandas()
    np.testing.assert_array_equal(result.noise0,noise[:,0])
    assert result.noise0.iloc[0] != float(np.float32(noise[0,0]))
