import numpy as np

from pycwb.modules.reconstruction.waveform_report_metrics import (
    compute_cumulative_hrss,
    compute_hrss,
    compute_leakage,
    compute_overlap,
)
from pycwb.types.time_series import TimeSeries
from pycwb.types.waveform import Waveform


def _waveform(data):
    return Waveform(TimeSeries(np.asarray(data, dtype=float), t0=0.0, dt=0.1))


def test_compute_overlap_identical_waveform_is_one():
    reference = _waveform([0.0, 1.0, 0.0, -1.0, 0.0])

    overlap = compute_overlap(np.asarray([reference.data]), reference)

    assert np.isclose(overlap, 1.0)


def test_compute_cumulative_hrss_is_monotonic():
    cumulative = compute_cumulative_hrss(
        np.asarray([[0.0, 1.0, 2.0, 2.0]]),
        delta_t=0.25,
        axis=1,
    )

    assert np.all(np.diff(cumulative[0]) >= 0)
    assert np.isclose(cumulative[0, -1], compute_hrss([0.0, 1.0, 2.0, 2.0], 0.25))


def test_compute_leakage_runs_on_synthetic_waveform():
    reference = _waveform([0.0, 1.0, 0.0, -1.0, 0.0])
    reconstructed = [_waveform([0.0, 1.0, 0.0, -1.0, 0.0])]
    time = np.linspace(0.0, 0.2, 4)

    mean, std = compute_leakage(reconstructed, reference, time)

    assert mean.shape == time.shape
    assert std.shape == time.shape
