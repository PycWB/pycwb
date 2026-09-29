"""Exercise the real subprocess boundary, including malformed backend inputs."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from pycwb.workflow.execution.cache import FrameCache
from pycwb.workflow.execution.planner import FrameRequest
from pycwb.workflow.execution.resources import MemoryBudget


@pytest.mark.parametrize("result", [(128, None), EOFError()])
def test_decoder_exits_during_memory_check(monkeypatch, tmp_path, result):
    from pycwb.workflow.execution import decoder, resources

    parent, child, process = Mock(), Mock(), Mock()
    parent.poll.side_effect = [False, True]
    if isinstance(result, Exception):
        parent.recv.side_effect = result
    else:
        parent.recv.return_value = result
    process.is_alive.return_value = False
    process.exitcode = 0
    context = SimpleNamespace(Pipe=Mock(return_value=(parent, child)),
                              Process=Mock(return_value=process))
    monkeypatch.setattr(decoder.multiprocessing, "get_context", lambda _: context)
    monkeypatch.setattr(resources, "process_tree_memory", lambda: (0, 0))
    monkeypatch.setattr(resources, "available_memory", lambda: 1024**3)
    request = FrameRequest("frame.gwf", "H1:TEST", 100, 104, 128)
    budget = MemoryBudget(1024**3, 0, 1024**2, 0, 1024**2, 1)
    args = ("frame.gwf", request, (), str(tmp_path / "cache.npy"), budget, 0, None)
    if isinstance(result, Exception):
        with pytest.raises(RuntimeError, match="without a result"):
            decoder.decode_to_file(*args)
    else:
        assert decoder.decode_to_file(*args) == 128
        process.terminate.assert_not_called()
    assert parent.poll.call_count == 2
    parent.recv.assert_called_once()


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
