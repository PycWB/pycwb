"""Unit tests for the ``deduplicate`` parameter of ``Catalog.triggers``."""
import os
import tempfile
import unittest

import pyarrow.parquet as pq

from pycwb.config import Config
from pycwb.modules.catalog.catalog import Catalog
from pycwb.types.trigger import Trigger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_catalog(path: str, triggers: list[Trigger]) -> Catalog:
    cat = Catalog.create(path, Config(), [])
    if triggers:
        cat.add_triggers(triggers)
    return cat


def _trigger(job_id: int, tid: str, lag_idx: int = 0, trial_idx: int = 0) -> Trigger:
    t = Trigger()
    t.job_id = job_id
    t.id = tid
    t.lag_idx = lag_idx
    t.trial_idx = trial_idx
    return t


# ---------------------------------------------------------------------------
# Test: no duplicates
# ---------------------------------------------------------------------------

class TestTriggersNoDuplicates(unittest.TestCase):
    """triggers() returns the full table when there are no duplicates."""

    def test_empty_catalog(self):
        with tempfile.TemporaryDirectory() as d:
            cat = _make_catalog(os.path.join(d, "catalog.parquet"), [])
            table = cat.triggers(deduplicate=True)
            self.assertEqual(table.num_rows, 0)

    def test_single_trigger(self):
        with tempfile.TemporaryDirectory() as d:
            cat = _make_catalog(
                os.path.join(d, "catalog.parquet"),
                [_trigger(job_id=1, tid="a")],
            )
            table = cat.triggers(deduplicate=True)
            self.assertEqual(table.num_rows, 1)

    def test_distinct_triggers_all_kept(self):
        with tempfile.TemporaryDirectory() as d:
            triggers = [
                _trigger(job_id=1, tid="a", lag_idx=0),
                _trigger(job_id=1, tid="b", lag_idx=0),
                _trigger(job_id=2, tid="a", lag_idx=0),  # same id, different job
            ]
            cat = _make_catalog(os.path.join(d, "catalog.parquet"), triggers)
            table = cat.triggers(deduplicate=True)
            self.assertEqual(table.num_rows, 3)


# ---------------------------------------------------------------------------
# Test: duplicate by (job_id, id)
# ---------------------------------------------------------------------------

class TestTriggersDedupByJobAndId(unittest.TestCase):
    """Rows with the same (job_id, id, lag_idx, trial_idx) are deduplicated."""

    def test_exact_duplicate_removed(self):
        """Two identical rows → only one kept."""
        with tempfile.TemporaryDirectory() as d:
            tr = _trigger(job_id=1, tid="x", lag_idx=0, trial_idx=0)
            cat = _make_catalog(os.path.join(d, "catalog.parquet"), [tr])
            # Manually append the same trigger a second time to simulate a re-run
            cat.add_triggers([tr])
            self.assertEqual(pq.read_metadata(cat.filename).num_rows, 2)

            table = cat.triggers(deduplicate=True)
            self.assertEqual(table.num_rows, 1)

    def test_last_occurrence_kept(self):
        """When there are two rows with the same key, the last write wins."""
        with tempfile.TemporaryDirectory() as d:
            tr1 = _trigger(job_id=1, tid="x", lag_idx=0, trial_idx=0)
            tr2 = _trigger(job_id=1, tid="x", lag_idx=0, trial_idx=0)
            tr2.rho = 9.9  # differentiate the payload

            cat = _make_catalog(os.path.join(d, "catalog.parquet"), [tr1])
            cat.add_triggers([tr2])
            self.assertEqual(pq.read_metadata(cat.filename).num_rows, 2)

            table = cat.triggers(deduplicate=True)
            self.assertEqual(table.num_rows, 1)
            self.assertAlmostEqual(table["rho"][0].as_py(), 9.9, places=5)

    def test_multiple_duplicates_collapsed(self):
        """Three copies of the same trigger → one row."""
        with tempfile.TemporaryDirectory() as d:
            tr = _trigger(job_id=1, tid="dup")
            cat = _make_catalog(os.path.join(d, "catalog.parquet"), [tr, tr, tr])
            table = cat.triggers(deduplicate=True)
            self.assertEqual(table.num_rows, 1)

    def test_mixed_unique_and_duplicates(self):
        """Three duplicates + two unique triggers → three total rows."""
        with tempfile.TemporaryDirectory() as d:
            dup = _trigger(job_id=1, tid="dup")
            uniq_a = _trigger(job_id=1, tid="uniq_a")
            uniq_b = _trigger(job_id=2, tid="dup")  # same id but different job
            cat = _make_catalog(
                os.path.join(d, "catalog.parquet"),
                [dup, dup, dup, uniq_a, uniq_b],
            )
            table = cat.triggers(deduplicate=True)
            self.assertEqual(table.num_rows, 3)


# ---------------------------------------------------------------------------
# Test: composite key includes lag_idx and trial_idx
# ---------------------------------------------------------------------------

class TestTriggersDedupCompositeKey(unittest.TestCase):
    """lag_idx and trial_idx are part of the composite key when present."""

    def test_same_id_different_lag_not_deduplicated(self):
        """Same (job_id, id) but different lag_idx → two distinct rows."""
        with tempfile.TemporaryDirectory() as d:
            triggers = [
                _trigger(job_id=1, tid="ev1", lag_idx=0, trial_idx=0),
                _trigger(job_id=1, tid="ev1", lag_idx=1, trial_idx=0),
            ]
            cat = _make_catalog(os.path.join(d, "catalog.parquet"), triggers)
            table = cat.triggers(deduplicate=True)
            self.assertEqual(table.num_rows, 2)

    def test_same_id_different_trial_not_deduplicated(self):
        """Same (job_id, id, lag_idx) but different trial_idx → two rows."""
        with tempfile.TemporaryDirectory() as d:
            triggers = [
                _trigger(job_id=1, tid="ev1", lag_idx=0, trial_idx=0),
                _trigger(job_id=1, tid="ev1", lag_idx=0, trial_idx=1),
            ]
            cat = _make_catalog(os.path.join(d, "catalog.parquet"), triggers)
            table = cat.triggers(deduplicate=True)
            self.assertEqual(table.num_rows, 2)

    def test_duplicate_across_lags_and_trials_collapsed(self):
        """Two rows sharing all four key columns → one row kept."""
        with tempfile.TemporaryDirectory() as d:
            tr = _trigger(job_id=1, tid="ev1", lag_idx=2, trial_idx=3)
            cat = _make_catalog(os.path.join(d, "catalog.parquet"), [tr])
            cat.add_triggers([tr])
            table = cat.triggers(deduplicate=True)
            self.assertEqual(table.num_rows, 1)

    def test_many_lags_no_false_dedup(self):
        """Ten triggers with the same id but different lags → all ten kept."""
        with tempfile.TemporaryDirectory() as d:
            triggers = [
                _trigger(job_id=1, tid="ev", lag_idx=i, trial_idx=0)
                for i in range(10)
            ]
            cat = _make_catalog(os.path.join(d, "catalog.parquet"), triggers)
            table = cat.triggers(deduplicate=True)
            self.assertEqual(table.num_rows, 10)


# ---------------------------------------------------------------------------
# Test: deduplicate=False skips deduplication
# ---------------------------------------------------------------------------

class TestTriggersDeduplicateFalse(unittest.TestCase):
    """deduplicate=False returns the raw table including duplicates."""

    def test_duplicates_present_when_disabled(self):
        with tempfile.TemporaryDirectory() as d:
            tr = _trigger(job_id=1, tid="x")
            cat = _make_catalog(os.path.join(d, "catalog.parquet"), [tr])
            cat.add_triggers([tr])

            raw = cat.triggers(deduplicate=False)
            deduped = cat.triggers(deduplicate=True)

            self.assertEqual(raw.num_rows, 2)
            self.assertEqual(deduped.num_rows, 1)

    def test_no_duplicates_same_result(self):
        """When there are no duplicates, both modes return the same count."""
        with tempfile.TemporaryDirectory() as d:
            triggers = [
                _trigger(job_id=1, tid="a"),
                _trigger(job_id=1, tid="b"),
            ]
            cat = _make_catalog(os.path.join(d, "catalog.parquet"), triggers)
            self.assertEqual(
                cat.triggers(deduplicate=True).num_rows,
                cat.triggers(deduplicate=False).num_rows,
            )


if __name__ == "__main__":
    unittest.main()
