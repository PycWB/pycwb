from types import SimpleNamespace

import numpy as np

from pycwb.modules.read_data.write_data import save_to_gwf


def test_save_to_gwf_sets_channel_and_filename(mocker, tmp_path):
    h1_gwf = mocker.MagicMock()
    l1_gwf = mocker.MagicMock()
    gwpy_time_series = mocker.patch(
        "pycwb.modules.read_data.write_data.GWpyTimeSeries",
        side_effect=[h1_gwf, l1_gwf],
    )
    signals = [
        SimpleNamespace(data=np.array([1.0]), sample_times=np.array([100.0])),
        SimpleNamespace(data=np.array([2.0]), sample_times=np.array([100.0])),
    ]

    save_to_gwf(
        signals,
        ["H1", "L1"],
        "STRAIN",
        tmp_path,
        start_time=100.9,
        duration=4.8,
        label="SIM",
    )

    assert gwpy_time_series.call_count == 2
    assert h1_gwf.channel == "H1:STRAIN"
    assert h1_gwf.name == "H1:STRAIN"
    assert l1_gwf.channel == "L1:STRAIN"
    assert l1_gwf.name == "L1:STRAIN"
    h1_gwf.write.assert_called_once_with(str(tmp_path / "H1-SIM-100-4.gwf"))
    l1_gwf.write.assert_called_once_with(str(tmp_path / "L1-SIM-100-4.gwf"))
