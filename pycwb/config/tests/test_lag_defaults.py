"""Omitted lag settings mean zero-lag; recorded nonzero runs stay protected."""

import numpy as np
import orjson
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pycwb.constants import user_parameters_schema
from pycwb.types.job import WaveSegment
from pycwb.utils.yaml_helper import load_yaml
from pycwb.workflow.subflow.prepare_job_runs import validate_run_config


BASE_YAML = "analysis: 2G\nifo: [H1, L1]\nrefIFO: H1\n"


def _segment(params):
    return WaveSegment(0, ["H1", "L1"], 1000, 1600, 4096, 8,
                       lag_size=params["lagSize"], lag_step=params["lagStep"],
                       lag_off=params["lagOff"], lag_max=params["lagMax"])


def test_omitted_lags_resolve_to_one_unshifted_trial(tmp_path):
    source = tmp_path / "parameters.yaml"
    source.write_text(BASE_YAML)
    params = load_yaml(source, user_parameters_schema)
    np.testing.assert_array_equal(_segment(params).lag_shifts, [[0., 0.]])


def test_explicit_background_lags_are_preserved(tmp_path):
    source = tmp_path / "parameters.yaml"
    source.write_text(BASE_YAML + "lagSize: 1\nlagOff: 6\nlagMax: 150\n")
    params = load_yaml(source, user_parameters_schema)
    assert params["lagOff"] == 6 and params["lagMax"] == 150
    np.testing.assert_array_equal(_segment(params).lag_shifts, [[0., 3.]])


def test_resume_requires_restoring_legacy_implicit_lags(tmp_path):
    source = tmp_path / "parameters.yaml"
    source.write_text(BASE_YAML)
    stored = load_yaml(source, user_parameters_schema)
    stored.update(lagOff=6, lagMax=150)
    catalog_dir = tmp_path / "catalog"
    catalog_dir.mkdir()
    table = pa.table({"id": pa.array([], type=pa.int64())}).replace_schema_metadata({
        b"config": orjson.dumps({"_yaml_parameters": stored}),
    })
    pq.write_table(table, catalog_dir / "catalog.parquet")
    with pytest.raises(ValueError, match="Changed settings: lagMax, lagOff") as error:
        validate_run_config(source, tmp_path)
    assert "--force-overwrite does not bypass" in str(error.value)
    source.write_text(BASE_YAML + "lagOff: 6\nlagMax: 150\n")
    validate_run_config(source, tmp_path)
