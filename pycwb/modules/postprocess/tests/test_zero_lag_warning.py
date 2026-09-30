"""Background inputs are used as given; unshifted rows are only reported."""

import logging

import pandas as pd

from pycwb.modules.postprocess.lag_filters import recorded_zero_lag_count
from pycwb.modules.postprocess.train_xgboost import _warn_zero_lag_background


def rows():
    # Zero lag, a regular lag (lagOff: 1 gives lag_idx 0 a shift), and lag
    # index 0 of a superlag-shifted job.
    return pd.DataFrame({
        "lag_idx": [0, 0, 0],
        "time_lag_H1": [0., 1., 0.],
        "time_lag_L1": [0., 0., 0.],
        "segment_lag_H1": [0., 0., 0.],
        "segment_lag_L1": [0., 0., 600.],
        "rho0": [8., 9., 10.],
    })


def test_only_rows_without_any_shift_count_as_zero_lag():
    assert recorded_zero_lag_count(rows()) == 1
    assert recorded_zero_lag_count(pd.DataFrame({"lag_idx": [0], "livetime": [1.]})) is None


def test_training_background_with_zero_lag_warns_without_filtering(tmp_path, caplog):
    shifted, mixed = tmp_path / "shifted.parquet", tmp_path / "mixed.parquet"
    rows().iloc[1:].to_parquet(shifted)
    rows().to_parquet(mixed)
    with caplog.at_level(logging.WARNING, logger="pycwb.modules.postprocess.train_xgboost"):
        _warn_zero_lag_background([str(shifted), str(mixed)])
    messages = [record.getMessage() for record in caplog.records]
    assert len(messages) == 1
    assert "mixed.parquet: 1 BKG rows have no time or segment shift" in messages[0]
