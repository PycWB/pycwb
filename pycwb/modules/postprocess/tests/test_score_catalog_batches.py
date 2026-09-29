"""Prediction cuts must not change the schema between Parquet batches."""

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import xgboost as xgb

from pycwb.modules.postprocess import evaluate


@pytest.fixture
def model_file(tmp_path):
    model = xgb.XGBClassifier(n_estimators=2, max_depth=1, n_jobs=1)
    model.fit(pd.DataFrame({"norm": [1., 2., 8., 9.]}), [0, 0, 1, 1])
    path = tmp_path / "model.ubj"
    model.save_model(path)
    return str(path)


def catalog_row(energy):
    row = dict(
        ifo_list=["H1", "L1"], coherent_energy=energy,
        coherent_energy_norm=999., packet_norm=9., penalty=1.,
        likelihood=100., net_cc=.9, q_veto=4., q_factor=6.,
        rho=8., Lveto2=0., lag_idx=0,
    )
    for i, ifo in enumerate(["H1", "L1"]):
        for key, value in [
            ("signal_energy", 40. + 20 * i), ("data_energy", 80. + 40 * i),
            ("noise_rms", np.float32((2. + i) * 1e-24)),
            ("central_freq", 100. + 50 * i), ("duration", .1 + i),
            ("bandwidth", 20. + i),
        ]:
            row[f"{key}_{ifo}"] = value
    return row


@pytest.mark.parametrize("energies", [
    [100., 1.], [1., 100.], [100., 1., 100.], [1., 1.], [100., 100.],
])
def test_prediction_cuts_preserve_batch_schema(tmp_path, model_file, energies):
    source = tmp_path / "input.parquet"
    pd.DataFrame([catalog_row(energy) for energy in energies]).to_parquet(source, index=False)
    output = tmp_path / "scored.parquet"
    result = evaluate.score_catalog(
        str(tmp_path), str(source), model_file, output_file=str(output),
        lag_selection="zero_lag", batch_size=1,
    )
    scored = pq.read_table(output)
    expected = sum(energy > 7.2 ** 2 for energy in energies)
    assert result["n_input"] == len(energies)
    assert result["n_scored"] == scored.num_rows == expected
    assert scored.schema.field("ifo_list").type == pa.list_(pa.string())
    assert pa.types.is_floating(scored.schema.field("xgb_prob").type)
    assert "MLstat" in scored.column_names
    assert scored["coherent_energy"].to_pylist() == [e for e in energies if e > 7.2 ** 2]


def test_already_selected_catalog_does_not_recheck_lags(tmp_path, model_file, monkeypatch):
    source = tmp_path / "selected.parquet"
    row = catalog_row(100.)
    # This can be a nonzero superlag even though the regular lag index is zero.
    row["segment_lag_H1"] = 10.
    pd.DataFrame([row]).to_parquet(source, index=False)

    def unexpected_filter(*args, **kwargs):
        pytest.fail("The selection stage already separated zero lag")

    monkeypatch.setattr(evaluate, "try_unshifted_job_ids_from_catalog", unexpected_filter)
    monkeypatch.setattr(evaluate, "zero_lag_mask", unexpected_filter)
    monkeypatch.setattr(evaluate, "nonzero_lag_mask", unexpected_filter)
    result = evaluate.score_catalog(
        str(tmp_path), str(source), model_file, output_file="scored.parquet",
        lag_selection="all", batch_size=1,
    )
    assert result["n_scored"] == 1

    # The FAR holdout in the example is already selected too.
    result = evaluate.evaluate_far_rho(
        str(tmp_path), str(source), model_file, livetime=100., exclude_zero_lag=False,
    )
    assert len(result["far_rho"]) == 1


def test_null_optional_list_in_first_batch_keeps_source_type(tmp_path, model_file):
    source = tmp_path / "input.parquet"
    frame = pd.DataFrame([catalog_row(100.), catalog_row(100.)])
    frame["optional_vector"] = [None, [1., 2.]]
    frame.to_parquet(source, index=False)
    evaluate.score_catalog(
        str(tmp_path), str(source), model_file, output_file="scored.parquet", batch_size=1,
    )
    result = pq.read_table(tmp_path / "scored.parquet")
    assert result["optional_vector"].to_pylist() == [None, [1., 2.]]


def test_inconsistent_ranking_columns_are_not_silently_dropped(tmp_path, model_file, monkeypatch):
    source = tmp_path / "input.parquet"
    pd.DataFrame([catalog_row(100.), catalog_row(200.)]).to_parquet(source, index=False)
    score = evaluate._score_catalog_dataframe

    def inconsistent_hook(frame, *args):
        result = score(frame, *args)
        if frame.coherent_energy.iloc[0] == 200.:
            result["unexpected_ranking"] = 1.
        return result

    monkeypatch.setattr(evaluate, "_score_catalog_dataframe", inconsistent_hook)
    with pytest.raises(ValueError, match="field names"):
        evaluate.score_catalog(
            str(tmp_path), str(source), model_file, output_file="scored.parquet", batch_size=1,
        )
