"""Calibration consistency between SNR estimation and final signal injection."""

from types import SimpleNamespace

import numpy as np
import pytest

from pycwb.modules.injection import snr_population
from pycwb.types.time_series import TimeSeries


@pytest.mark.parametrize("mode", ["fft", "cwb"])
def test_nonunit_calibration_matches_final_injection_convention(monkeypatch, mode):
    config = SimpleNamespace(
        injection_resampling=mode,
        inRate=16,
        fResample=0,
        levelR=0,
        dcCal=[2.0, 3.0],
        iwindow=2.0,
        fLow=0.0,
        fHigh=8.0,
    )
    source = {"gps_time": 2.0, "target_snr": 12.0, "hrss": 1e-22}
    segment = SimpleNamespace(injections=[source], sample_rate=16, ifos=["L1", "H1"])
    data = [TimeSeries(np.ones(64), dt=1 / 16, t0=0.0) for _ in range(2)]
    waveform = np.zeros(64)
    waveform[32] = 1.0
    monkeypatch.setattr(
        snr_population,
        "generate_strain_from_injection",
        lambda *args: [
            TimeSeries(waveform.copy(), dt=1 / 16, t0=0.0) for _ in range(2)
        ],
    )
    # A scalar noise RMS isolates calibration from the independently tested WDM estimator.
    monkeypatch.setattr(
        snr_population,
        "whitening_python",
        lambda config, noise, **kw: (noise, noise.data[0]),
    )
    monkeypatch.setattr(
        snr_population,
        "whiten_injection_strain",
        lambda config, signal, rms: (
            TimeSeries(signal.data / rms, dt=signal.dt, t0=signal.t0),
            None,
        ),
    )
    monkeypatch.setattr(
        snr_population, "cwb_snr_energy", lambda signal, *args: np.sum(signal.data**2)
    )
    scale = snr_population.target_snr_scales(config, segment, data)[0]
    # FFT mode adds signal before calibration; cWB mode adds uncalibrated MDC separately.
    final_energy = sum(
        (scale * (cal if mode == "fft" else 1) / cal) ** 2 for cal in config.dcCal
    )
    assert np.sqrt(final_energy) == pytest.approx(source["target_snr"])
    assert source["hrss"] == 1e-22
    assert config.dcCal == [2.0, 3.0]
    for noise in data:
        np.testing.assert_array_equal(noise.data, np.ones(64))
