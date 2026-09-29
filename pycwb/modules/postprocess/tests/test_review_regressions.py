"""Regression coverage for reviewed scoring and efficiency defects."""

import numpy as np
import pandas as pd
import pytest

from pycwb.modules.cwb_xgboost.read_data import preprocess_events
from pycwb.modules.postprocess import plot_efficiency
from pycwb.modules.postprocess.efficiency_metrics import (
    _parse_waveform_q_frequency,
    _validate_fixed_hrss_population,
)
from pycwb.modules.postprocess.prediction_cuts import prediction_mask


def test_plot_wrapper_forwards_matched_table_and_options(tmp_path):
    pd.DataFrame(
        {
            "sim_sim_idx": [0, 1],
            "id": ["a", None],
            "sim_hrss": [1e-22, 1e-22],
            "xgb_prob": [0.99, np.nan],
        }
    ).to_parquet(tmp_path / "matched.parquet")
    pd.DataFrame({"xgb_prob": [0.9]}).to_parquet(tmp_path / "bkg.parquet")
    result = plot_efficiency.plot_efficiency_vs_hrss(
        str(tmp_path),
        "unused",
        "bkg.parquet",
        100.0,
        ifar="50",
        matched_right_file="matched.parquet",
        output_file="curve.png",
    )
    assert result["efficiency_curve"][0]["efficiency"] == 0.5
    assert (tmp_path / "curve.png").is_file()


@pytest.mark.parametrize(
    "stored", [{"rho": [8.0]}, {"rho0": [8.0]}, {"rho": [[8.0, 9.0]]}]
)
def test_stored_rho_survives_derived_feature_preprocessing(stored):
    frame = pd.DataFrame({**stored, "ecor": [36.0], "penalty": [1.0], "netcc0": [0.8]})
    result = preprocess_events(frame, 2, {"readfile(vars)": [], "rho0(define)": 1}, {})
    assert result.rho0.tolist() == [6.0]
    assert prediction_mask(result, "rho[0]>7&&netcc[0]>0.7").tolist() == [True]
    assert prediction_mask(result, "rho0>7").tolist() == [False]


@pytest.mark.parametrize("name", ["SG235Q8d9", "SGE235Q8.9", "SG_Q8d9_235Hz"])
def test_fractional_q_names(name):
    assert _parse_waveform_q_frequency(name) == (8.9, 235)


def test_fractional_q_is_plotted(tmp_path):
    from pycwb.modules.postprocess.efficiency_plots import (
        _plot_efficiency_by_waveform_panels,
    )

    output = tmp_path / "fractional.png"
    _plot_efficiency_by_waveform_panels(
        [
            {
                "Q": 8.9,
                "frequency": 235,
                "waveform": "SG235Q8d9",
                "data": [{"hrss": 1e-22, "efficiency": 0.5, "n_total": 2}],
            }
        ],
        0.9,
        "50",
        50.0,
        str(output),
    )
    assert output.is_file()


@pytest.mark.parametrize(
    "column,value",
    [
        ("sim_target_snr", 10.0),
        ("sim_targeted_snr", 10.0),
        ("sim_snr_scale", 2.0),
    ],
)
@pytest.mark.parametrize(
    "action",
    [
        "compute_hrss50",
        "compute_efficiency_by_waveform",
        "compute_efficiency_vs_hrss_by_waveform",
        "simulation_efficiency",
    ],
)
def test_amplitude_reports_reject_snr_normalized_sources(
    tmp_path, column, value, action
):
    frame = pd.DataFrame(
        {
            "sim_sim_idx": [0],
            "id": [None],
            "sim_hrss": [1e-22],
            column: [value],
        }
    )
    frame.to_parquet(tmp_path / "matched.parquet")
    with pytest.raises(ValueError, match="fixed-hrss"):
        if action == "simulation_efficiency":
            from pycwb.modules.postprocess.simulation_report import (
                simulation_efficiency,
            )

            simulation_efficiency(str(tmp_path), "matched.parquet", "report")
        elif action == "compute_hrss50":
            plot_efficiency.compute_hrss50(
                str(tmp_path),
                "unused",
                "unused",
                100.0,
                matched_right_file="matched.parquet",
            )
        else:
            getattr(plot_efficiency, action)(
                str(tmp_path),
                "unused",
                "matched.parquet",
                "unused",
                100.0,
                "unused",
            )


def test_fixed_amplitudes_allow_missing_target_metadata():
    _validate_fixed_hrss_population(
        pd.DataFrame(
            {
                "sim_target_snr": [0.0, np.nan],
                "sim_snr_scale": [1.0, np.nan],
            }
        )
    )
