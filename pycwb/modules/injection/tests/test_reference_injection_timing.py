import numpy as np
import pytest
from pycwb.modules.reconstruction.injection_timing import network_reference_times


def test_cwb_energy_squared_weighting_and_geometric_offsets():
    # Reference injection::fill_in: weights are ISNR^2, where ISNR=SNR^2.
    times=network_reference_times([100.,104.],[1.,3.],[.002,-.006])
    np.testing.assert_allclose(times,[103.6,103.592],rtol=0,atol=1e-13)


def test_captured_hf_source_72_time_cut():
    times=network_reference_times([1387222614.005074,1387222614.0188656],
        [141.0216,829.3167],[0.,-.0086482])
    cwb=[1387222614.0184872,1387222614.009839]
    np.testing.assert_allclose(times,cwb,rtol=0,atol=1e-5)
    reconstructed=1387222614.113037
    assert reconstructed-1387222614.005074>.1
    assert reconstructed-times[0]<.1

@pytest.mark.parametrize('energy',[[0.,0.],[-1.,3.],[float('nan'),3.]])
def test_invalid_energy_is_not_silently_used(energy):
    with pytest.raises(ValueError):network_reference_times([100.,101.],energy,[0.,.001])


@pytest.mark.parametrize("coordsys", ["icrs", "geo", "cwb"])
def test_strain_without_sky_retains_reconstructed_detector_centroids(coordsys):
    from types import SimpleNamespace
    from pycwb.types.time_series import TimeSeries
    from pycwb.workflow.subflow.postprocess_and_plots import reconstruct_injection_waveforms_flow

    time = np.arange(4096) / 128.
    samples = np.sin(2 * np.pi * 20 * time) * np.exp(-((time - 16) / .2) ** 2)
    waves = [
        TimeSeries(samples, t0=100., dt=1 / 128.),
        TimeSeries(samples, t0=100.01, dt=1 / 128.),
    ]
    event = SimpleNamespace(
        hash_id="strain", injection={"gps_time": 116., "coordsys": coordsys},
    )

    def reconstruct(resampling):
        return reconstruct_injection_waveforms_flow(
            ".", SimpleNamespace(injection_resampling=resampling), ["H1", "L1"],
            event, waves, waves, window=4., offset=1., inRate=128,
            save=False, plot=False,
        )

    expected = reconstruct("fft")
    actual = reconstruct("cwb")
    np.testing.assert_array_equal(actual["central_time"], expected["central_time"])
    np.testing.assert_array_equal(actual["snr"], expected["snr"])
    assert actual["central_time"][0] != actual["central_time"][1]


def test_incomplete_sky_coordinates_still_fail():
    from pycwb.modules.reconstruction.injection_timing import cwb_arrival_times

    with pytest.raises(ValueError, match="source sky coordinates"):
        cwb_arrival_times([100., 101.], [1., 2.], {"gps_time": 100., "ra": 1.},
                          ["H1", "L1"], None)
