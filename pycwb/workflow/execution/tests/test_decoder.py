"""Exercise the real subprocess boundary, including malformed backend inputs."""

from pathlib import Path

import pytest

from pycwb.workflow.execution.cache import FrameCache
from pycwb.workflow.execution.planner import FrameRequest
from pycwb.workflow.execution.resources import MemoryBudget


def test_corrupt_frame_decoder_failure_removes_partial_files(tmp_path):
    source = tmp_path / "corrupt.gwf"
    source.write_bytes(b"not a frame")
    request = FrameRequest(str(source), "H1:TEST", 100, 104, 128)
    budget = MemoryBudget(3 * 1024**3, 0, 64 * 1024**2, 0, 1024**2, 1)
    cache = FrameCache(tmp_path, budget)
    directory = Path(cache._temporary.name)
    try:
        with pytest.raises(RuntimeError, match="decoder failed"):
            cache.acquire((request,))
        assert cache.bytes == 0
        assert not list(directory.glob("*.npy"))
    finally:
        cache.close()
    assert not directory.exists()
