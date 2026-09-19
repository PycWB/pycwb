from pathlib import Path
import json
import numpy as np
import pytest
from pycwb.modules.likelihoodWP.waveform_statistics import (
    waveform_rms,
    waveform_time,
    waveform_frequency,
    network_centroids,
    final_statistics,
    sky_scale,
)

OUT = Path(__file__).with_name("reference")


def test_empty_and_invalid_waveforms():
    empty = np.empty(0, dtype=np.float64)
    assert waveform_rms(empty) == 0
    assert waveform_time(empty, 12.0, 1024.0) == 0
    assert waveform_frequency(empty, 1024.0) == 0
    with pytest.raises(ValueError, match="even length"):
        waveform_frequency(np.ones(3), 1024.0)


def test_live_release_waveform_fixtures():
    data = np.load(OUT / "waveform_oracle.npz")
    for case in json.loads((OUT / "waveform_oracle.json").read_text()):
        name = case["name"]
        values = data[name]
        rms = waveform_rms(values)
        actual = dict(
            energy=rms * rms * len(values),
            time=waveform_time(values, case["start"], case["rate"]),
            frequency=waveform_frequency(values, case["rate"]),
        )
        for key, value in actual.items():
            np.testing.assert_allclose(value, case[key], rtol=3e-14, atol=3e-13)


def test_centroid_and_final_statistic_release_fixtures():
    data = np.load(OUT / "waveform_scalar_oracle.npz")
    for name in json.loads((OUT / "waveform_scalar_oracle.json").read_text()):
        result = network_centroids(*(data[name + "__" + k] for k in ["energy", "time", "frequency"]))
        np.testing.assert_array_equal(result, data[name + "__centroid"])
        args = data[name + "__args"]
        values = final_statistics(*args[:12])
        result = [
            values[k] for k in ["ch", "cr", "cp", "norm", "null", "residual", "norm_cor", "rho_reduced", "xrho_reduced"]
        ]
        result.append(sky_scale(values["norm"] / 2.0, args[7], args[12], args[13]))
        np.testing.assert_array_equal(result, data[name + "__final"])
