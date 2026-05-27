"""Unit tests for permission handling in ``_write_table_atomic``."""
import os
import stat
import tempfile
import unittest

import pyarrow as pa

from pycwb.modules.catalog.catalog import _write_table_atomic


def _simple_table() -> pa.Table:
    return pa.table({"x": pa.array([1, 2, 3], type=pa.int32())})


class TestWriteTableAtomicPermissions(unittest.TestCase):
    """_write_table_atomic must respect the process umask, not fix at 0o600."""

    def _file_mode(self, path: str) -> int:
        return stat.S_IMODE(os.stat(path).st_mode)

    def _call_with_umask(self, umask: int) -> int:
        """Write a table with a specific umask and return the resulting file mode."""
        with tempfile.TemporaryDirectory() as d:
            dest = os.path.join(d, "catalog.parquet")
            old_umask = os.umask(umask)
            try:
                _write_table_atomic(_simple_table(), dest)
            finally:
                os.umask(old_umask)
            return self._file_mode(dest)

    def test_permissions_with_umask_022(self):
        """File mode should be 0o644 when umask is 0o022."""
        mode = self._call_with_umask(0o022)
        expected = 0o666 & ~0o022  # 0o644
        self.assertEqual(
            mode, expected,
            f"Expected 0o{expected:03o} but got 0o{mode:03o} (umask=0o022)",
        )

    def test_permissions_with_umask_002(self):
        """File mode should be 0o664 when umask is 0o002."""
        mode = self._call_with_umask(0o002)
        expected = 0o666 & ~0o002  # 0o664
        self.assertEqual(
            mode, expected,
            f"Expected 0o{expected:03o} but got 0o{mode:03o} (umask=0o002)",
        )

    def test_permissions_with_umask_077(self):
        """File mode should be 0o600 when umask is 0o077."""
        mode = self._call_with_umask(0o077)
        expected = 0o666 & ~0o077  # 0o600
        self.assertEqual(
            mode, expected,
            f"Expected 0o{expected:03o} but got 0o{mode:03o} (umask=0o077)",
        )

    def test_permissions_not_hardcoded_600(self):
        """File mode must NOT be 0o600 when umask allows more permissions."""
        mode = self._call_with_umask(0o022)
        self.assertNotEqual(
            mode, 0o600,
            "File mode was 0o600, indicating mkstemp permissions were not corrected.",
        )

    def test_no_temp_file_left_after_success(self):
        """No temporary .catalog_tmp_* files should remain after a successful write."""
        with tempfile.TemporaryDirectory() as d:
            dest = os.path.join(d, "catalog.parquet")
            _write_table_atomic(_simple_table(), dest)
            tmp_files = [
                f for f in os.listdir(d) if f.startswith(".catalog_tmp_")
            ]
            self.assertEqual(tmp_files, [], f"Stale temp files found: {tmp_files}")

    def test_destination_file_created(self):
        """The destination file must exist after a successful write."""
        with tempfile.TemporaryDirectory() as d:
            dest = os.path.join(d, "catalog.parquet")
            _write_table_atomic(_simple_table(), dest)
            self.assertTrue(os.path.exists(dest))

    def test_destination_dir_created(self):
        """Parent directory is created if it does not exist."""
        with tempfile.TemporaryDirectory() as d:
            dest = os.path.join(d, "nested", "subdir", "catalog.parquet")
            _write_table_atomic(_simple_table(), dest)
            self.assertTrue(os.path.exists(dest))


if __name__ == "__main__":
    unittest.main()
