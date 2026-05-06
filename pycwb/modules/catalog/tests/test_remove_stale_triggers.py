"""Unit tests for Catalog.remove_stale_triggers."""
import tempfile
import os
import unittest

from pycwb.modules.catalog.catalog import Catalog
from pycwb.config import Config
from pycwb.types.trigger import Trigger


def _make_catalog(path: str, triggers: list[Trigger]) -> Catalog:
    """Create a minimal Catalog with the given triggers."""
    config = Config()
    cat = Catalog.create(path, config, [])
    if triggers:
        cat.add_triggers(triggers)
    return cat


def _trigger(job_id: int, lag_idx: int, trial_idx: int, tid: str = None) -> Trigger:
    t = Trigger()
    t.job_id = job_id
    t.lag_idx = lag_idx
    t.trial_idx = trial_idx
    t.id = tid or f"{job_id}-{lag_idx}-{trial_idx}"
    return t


class TestRemoveStaleTriggersEmpty(unittest.TestCase):
    """remove_stale_triggers on a catalog with no trigger rows."""

    def test_empty_catalog_returns_zero(self):
        with tempfile.TemporaryDirectory() as d:
            cat = _make_catalog(os.path.join(d, "catalog.parquet"), [])
            n = cat.remove_stale_triggers(job_id=0, completed_lags={})
            self.assertEqual(n, 0)


class TestRemoveStaleTriggersNothingStale(unittest.TestCase):
    """When all triggers belong to completed lags, nothing should be removed."""

    def test_all_completed_nothing_removed(self):
        with tempfile.TemporaryDirectory() as d:
            triggers = [
                _trigger(job_id=1, lag_idx=0, trial_idx=0),
                _trigger(job_id=1, lag_idx=1, trial_idx=0),
            ]
            cat = _make_catalog(os.path.join(d, "catalog.parquet"), triggers)
            completed = {0: {0, 1}}
            n = cat.remove_stale_triggers(job_id=1, completed_lags=completed)
            self.assertEqual(n, 0)
            self.assertEqual(cat.triggers().num_rows, 2)

    def test_different_job_not_touched(self):
        """Triggers for a different job_id must never be removed."""
        with tempfile.TemporaryDirectory() as d:
            triggers = [
                _trigger(job_id=2, lag_idx=0, trial_idx=0),
                _trigger(job_id=2, lag_idx=1, trial_idx=0),
            ]
            cat = _make_catalog(os.path.join(d, "catalog.parquet"), triggers)
            # Tell it to clean job 1 (which has no rows); job 2 must survive.
            n = cat.remove_stale_triggers(job_id=1, completed_lags={})
            self.assertEqual(n, 0)
            self.assertEqual(cat.triggers().num_rows, 2)


class TestRemoveStaleTriggersAllStale(unittest.TestCase):
    """When completed_lags is empty, all triggers for that job are stale."""

    def test_all_stale_removed(self):
        with tempfile.TemporaryDirectory() as d:
            triggers = [
                _trigger(job_id=1, lag_idx=0, trial_idx=0),
                _trigger(job_id=1, lag_idx=1, trial_idx=0),
                _trigger(job_id=1, lag_idx=2, trial_idx=0),
            ]
            cat = _make_catalog(os.path.join(d, "catalog.parquet"), triggers)
            n = cat.remove_stale_triggers(job_id=1, completed_lags={})
            self.assertEqual(n, 3)
            self.assertEqual(cat.triggers().num_rows, 0)

    def test_other_job_preserved_when_all_stale(self):
        with tempfile.TemporaryDirectory() as d:
            triggers = [
                _trigger(job_id=1, lag_idx=0, trial_idx=0),
                _trigger(job_id=2, lag_idx=0, trial_idx=0),
            ]
            cat = _make_catalog(os.path.join(d, "catalog.parquet"), triggers)
            n = cat.remove_stale_triggers(job_id=1, completed_lags={})
            self.assertEqual(n, 1)
            remaining = cat.triggers()
            self.assertEqual(remaining.num_rows, 1)
            self.assertEqual(remaining["job_id"][0].as_py(), 2)


class TestRemoveStaleTriggersPartial(unittest.TestCase):
    """Only incomplete lag-trial pairs should be removed."""

    def test_partial_removal(self):
        """lag 0 is complete, lag 1 is stale — only lag 1 should be removed."""
        with tempfile.TemporaryDirectory() as d:
            triggers = [
                _trigger(job_id=1, lag_idx=0, trial_idx=0, tid="keep"),
                _trigger(job_id=1, lag_idx=1, trial_idx=0, tid="remove"),
            ]
            cat = _make_catalog(os.path.join(d, "catalog.parquet"), triggers)
            n = cat.remove_stale_triggers(job_id=1, completed_lags={0: {0}})
            self.assertEqual(n, 1)
            remaining = cat.triggers()
            self.assertEqual(remaining.num_rows, 1)
            self.assertEqual(remaining["id"][0].as_py(), "keep")

    def test_multiple_trials_partial_removal(self):
        """trial 0 lag 0 complete; trial 1 lag 0 stale → only trial 1 removed."""
        with tempfile.TemporaryDirectory() as d:
            triggers = [
                _trigger(job_id=1, lag_idx=0, trial_idx=0, tid="t0-keep"),
                _trigger(job_id=1, lag_idx=0, trial_idx=1, tid="t1-remove"),
            ]
            cat = _make_catalog(os.path.join(d, "catalog.parquet"), triggers)
            n = cat.remove_stale_triggers(job_id=1, completed_lags={0: {0}})
            self.assertEqual(n, 1)
            remaining = cat.triggers()
            self.assertEqual(remaining.num_rows, 1)
            self.assertEqual(remaining["id"][0].as_py(), "t0-keep")

    def test_multiple_triggers_per_lag(self):
        """Multiple triggers sharing the same stale (lag, trial) are all removed."""
        with tempfile.TemporaryDirectory() as d:
            triggers = [
                _trigger(job_id=1, lag_idx=0, trial_idx=0, tid="a"),
                _trigger(job_id=1, lag_idx=0, trial_idx=0, tid="b"),
                _trigger(job_id=1, lag_idx=1, trial_idx=0, tid="c"),
            ]
            cat = _make_catalog(os.path.join(d, "catalog.parquet"), triggers)
            # lag 1 complete, lag 0 stale
            n = cat.remove_stale_triggers(job_id=1, completed_lags={0: {1}})
            self.assertEqual(n, 2)
            remaining = cat.triggers()
            self.assertEqual(remaining.num_rows, 1)
            self.assertEqual(remaining["id"][0].as_py(), "c")


class TestRemoveStaleTriggersIdempotent(unittest.TestCase):
    """Calling remove_stale_triggers twice should be safe (idempotent)."""

    def test_idempotent(self):
        with tempfile.TemporaryDirectory() as d:
            triggers = [
                _trigger(job_id=1, lag_idx=0, trial_idx=0),
                _trigger(job_id=1, lag_idx=1, trial_idx=0),
            ]
            cat = _make_catalog(os.path.join(d, "catalog.parquet"), triggers)
            n1 = cat.remove_stale_triggers(job_id=1, completed_lags={0: {0}})
            n2 = cat.remove_stale_triggers(job_id=1, completed_lags={0: {0}})
            self.assertEqual(n1, 1)
            self.assertEqual(n2, 0)  # nothing left to remove
            self.assertEqual(cat.triggers().num_rows, 1)


if __name__ == "__main__":
    unittest.main()
