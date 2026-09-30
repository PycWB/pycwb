"""Background exposure must retain regular lag zero for shifted jobs."""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pycwb.config import Config
from pycwb.modules.catalog.catalog import Catalog
from pycwb.modules.postprocess.background import process_background


@pytest.mark.parametrize("inline", [False, True])
def test_zero_lag_selection_with_catalog_provenance(tmp_path, inline):
    path = tmp_path / "catalog.parquet"
    jobs = [{"index": 1, "shift": [0, 0]}, {"index": 2, "shift": [0, 100]}]
    catalog = Catalog.create(str(path), Config(), jobs, jobs_in_metadata=inline)
    table = pa.table({"id": ["a", "b", "c"], "job_id": [1, 2, 1], "lag_idx": [0, 0, 1], "rho": [6., 7., 8.]})
    table = table.replace_schema_metadata(pq.read_schema(path).metadata)
    pq.write_table(table, path)
    progress = pd.DataFrame({"job_id": [1, 2, 1], "lag_idx": [0, 0, 1], "livetime": [10., 20., 30.]})

    disk = process_background(path, progress, thresholds=[5])
    assert disk["livetime"] == 50.
    assert disk["triggers"].job_id.tolist() == [2, 1]
    assert disk["curve"]["count"].tolist() == [2]
    memory = catalog.triggers()
    if inline:
        result = process_background(memory, progress, thresholds=[5])
        pd.testing.assert_frame_equal(result["curve"], disk["curve"])
    else:
        # A raw Arrow table loses the path needed to resolve a relative manifest.
        with pytest.raises(ValueError, match="catalog path.*unshifted_job_ids"):
            process_background(memory, progress, thresholds=[5])
    explicit = process_background(memory, progress, thresholds=[5], unshifted_job_ids={1})
    pd.testing.assert_frame_equal(explicit["curve"], disk["curve"])
    inclusive = process_background(memory, progress, thresholds=[5], exclude_zero_lag=False)
    assert inclusive["livetime"] == 60.
    assert inclusive["n_triggers"] == 3


def test_missing_job_manifest_is_not_ignored(tmp_path):
    path = tmp_path / "catalog.parquet"
    Catalog.create(str(path), Config(), [{"index": 1, "shift": [0, 0]}])
    (tmp_path / "jobs.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="job manifest is missing"):
        process_background(path, pd.DataFrame(), thresholds=[5])
