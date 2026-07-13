from types import SimpleNamespace

import numpy as np

from pycwb.modules.read_data.simulations import (
    generate_injections,
    generate_noise_for_job_seg,
)
from pycwb.types.time_series import TimeSeries


def _time_series(value, *, length=4, dt=0.25, t0=10.0):
    return TimeSeries(data=np.full(length, value, dtype=float), dt=dt, t0=t0)


def test_generate_noise_for_job_seg_uses_padded_window_and_noise_config(mocker):
    job_seg = SimpleNamespace(
        index=7,
        ifos=["H1", "L1"],
        noise={"seeds": [11, 22], "psds": ["h1.txt", "l1.txt"]},
        padded_duration=12,
        padded_start=100,
    )
    generated = [_time_series(1), _time_series(2)]
    generate_noise = mocker.patch(
        "pycwb.modules.read_data.simulations.generate_noise",
        side_effect=generated,
    )

    result = generate_noise_for_job_seg(job_seg, 4096, f_low=16)

    assert result == generated
    assert generate_noise.call_args_list == [
        mocker.call(
            psd="h1.txt",
            f_low=16,
            sample_rate=4096,
            duration=12,
            start_time=100,
            seed=11,
        ),
        mocker.call(
            psd="l1.txt",
            f_low=16,
            sample_rate=4096,
            duration=12,
            start_time=100,
            seed=22,
        ),
    ]


def test_generate_noise_for_job_seg_combines_upstream_data(mocker):
    job_seg = SimpleNamespace(
        index=2,
        ifos=["H1"],
        noise={},
        padded_duration=1,
        padded_start=10,
    )
    mocker.patch(
        "pycwb.modules.read_data.simulations.generate_noise",
        return_value=_time_series(1),
    )

    result = generate_noise_for_job_seg(
        job_seg,
        sample_rate=4,
        data=[_time_series(2)],
    )

    np.testing.assert_allclose(result[0].data, 3)


def test_generate_injections_builds_zero_data_and_injects_each_signal(mocker):
    config = SimpleNamespace(inRate=4)
    job_seg = SimpleNamespace(
        ifos=["H1", "L1"],
        duration=1,
        analyze_start=10,
        injections=[{"id": 1}, {"id": 2}],
    )
    generate_strain = mocker.patch(
        "pycwb.modules.read_data.simulations.generate_strain_from_injection",
        side_effect=[
            [_time_series(1), _time_series(2)],
            [_time_series(3), _time_series(4)],
        ],
    )

    result = generate_injections(config, job_seg)

    np.testing.assert_allclose(result[0].data, 4)
    np.testing.assert_allclose(result[1].data, 6)
    assert generate_strain.call_count == 2
    assert all(call.args[2:] == (4.0, ["H1", "L1"])
               for call in generate_strain.call_args_list)
