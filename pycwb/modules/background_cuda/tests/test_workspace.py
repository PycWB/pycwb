"""Bounded reusable device workspace: validation without a GPU, transfers with one."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pycwb.modules.background_cuda.workspace import DEFAULT_BUDGET, Workspace


def test_default_budget_is_64_mib() -> None:
    assert Workspace().budget == DEFAULT_BUDGET == 64 * 1024**2
    assert Workspace(budget=1.5e3).budget == 1500


def test_reserve_rejects_negative_count_before_allocation() -> None:
    with pytest.raises(ValueError, match="Invalid device workspace request"):
        Workspace().reserve("slot", -1, np.float32)


def test_reserve_rejects_object_dtype_before_allocation() -> None:
    with pytest.raises(ValueError, match="Invalid device workspace request"):
        Workspace().reserve("slot", 4, object)


def test_reserve_enforces_budget_before_allocation() -> None:
    workspace = Workspace(budget=8)
    with pytest.raises(MemoryError, match="resident byte budget"):
        workspace.reserve("slot", 2, np.float64)
    assert workspace.slots == {}


def test_budget_accounts_for_other_retained_slots() -> None:
    workspace = Workspace(budget=16)
    workspace.slots["other"] = SimpleNamespace(nbytes=12, dtype=np.dtype(np.uint8), size=12)
    with pytest.raises(MemoryError):
        workspace.reserve("slot", 5, np.uint8)  # 8-byte power-of-two slot + 12 retained > 16
    # Replacing the retained slot itself does not count its old bytes.
    workspace.slots["other"] = SimpleNamespace(nbytes=12, dtype=np.dtype(np.uint8), size=12)
    with pytest.raises(MemoryError):
        workspace.reserve("other", 17, np.uint8)


def test_reserve_returns_existing_slot_without_allocation() -> None:
    workspace = Workspace(budget=64)
    existing = SimpleNamespace(nbytes=32, dtype=np.dtype(np.float32), size=8)
    workspace.slots["slot"] = existing
    assert workspace.reserve("slot", 8, np.float32) is existing
    assert workspace.reserve("slot", 3, "float32") is existing
    assert workspace.reserve("slot", 0, np.float32) is existing


def test_download_rejects_oversized_shape_before_transfer() -> None:
    source = SimpleNamespace(dtype=np.dtype(np.float32), nbytes=16)
    with pytest.raises(ValueError, match="exceeds its device allocation"):
        Workspace.download(source, (5,))


def test_download_of_empty_shape_needs_no_transfer() -> None:
    source = SimpleNamespace(dtype=np.dtype(np.int64), nbytes=64)
    result = Workspace.download(source, (0,))
    assert result.shape == (0,) and result.dtype == np.int64


@pytest.mark.gpu
class TestDeviceWorkspace:
    """Actual-device ownership, byte bounds and transfer exactness."""

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64])
    def test_upload_reuses_pointer_within_capacity(self, dtype: type) -> None:
        workspace = Workspace(budget=4096)
        previous = None
        for n in (200, 5, 128, 255, 0, 12):
            source = np.arange(n, dtype=dtype)
            allocation = workspace.upload("data", source)
            if previous is not None:
                assert allocation.device_ctypes_pointer.value == previous
            previous = allocation.device_ctypes_pointer.value
            restored = workspace.download(allocation, source.shape)
            assert restored.dtype == source.dtype
            np.testing.assert_array_equal(source.view(np.uint8), restored.view(np.uint8))

    def test_slot_grows_to_power_of_two_and_reallocates(self) -> None:
        workspace = Workspace(budget=1 << 20)
        small = workspace.reserve("out", 100, np.float32)
        assert small.size == 128
        same = workspace.reserve("out", 128, np.float32)
        assert same is small
        grown = workspace.reserve("out", 129, np.float32)
        assert grown is not small and grown.size == 256
        assert workspace.reserve("out", 1, np.float32) is grown
        assert workspace.reserve("single", 0, np.float32).size == 1

    def test_dtype_change_reallocates_slot(self) -> None:
        workspace = Workspace(budget=1 << 20)
        first = workspace.reserve("slot", 16, np.float32)
        second = workspace.reserve("slot", 16, np.int32)
        assert second is not first
        assert second.dtype == np.int32
        assert set(workspace.slots) == {"slot"}

    def test_budget_and_download_bounds_on_device(self) -> None:
        workspace = Workspace(budget=4096)
        allocation = workspace.upload("data", np.arange(12, dtype=np.int64))
        with pytest.raises(MemoryError):
            workspace.reserve("too_large", 4097, np.uint8)
        with pytest.raises(ValueError):
            workspace.download(allocation, (4097,))
        assert sum(slot.nbytes for slot in workspace.slots.values()) <= 4096

    def test_download_returns_only_requested_prefix(self) -> None:
        workspace = Workspace(budget=4096)
        source = np.arange(10, dtype=np.float64)
        allocation = workspace.upload("data", source)
        assert allocation.size == 16
        np.testing.assert_array_equal(workspace.download(allocation, (2, 3)), source[:6].reshape(2, 3))

    def test_upload_of_non_contiguous_input(self) -> None:
        workspace = Workspace(budget=4096)
        source = np.arange(20, dtype=np.float32).reshape(4, 5)[:, ::2]
        allocation = workspace.upload("data", source)
        np.testing.assert_array_equal(workspace.download(allocation, source.shape), source)


@pytest.mark.gpu
class TestDeviceBuffers:
    """``DeviceBuffers`` gives uniform transfers with and without a workspace."""

    @pytest.mark.parametrize("with_workspace", [False, True])
    def test_round_trip(self, with_workspace: bool) -> None:
        from pycwb.modules.background_cuda.cuda_runtime import DeviceBuffers

        buffers = DeviceBuffers(Workspace(budget=1 << 16) if with_workspace else None)
        source = np.arange(6, dtype=np.float64).reshape(2, 3)
        uploaded = buffers.upload("x", source, np.float32)
        assert uploaded.dtype == np.float32
        np.testing.assert_array_equal(buffers.download(uploaded, source.shape), source.astype(np.float32))
        output = buffers.output("y", (4, 2), np.int32)
        assert output.dtype == np.int32
        assert output.size >= 8
        assert buffers.download(output, (4, 2)).shape == (4, 2)
        if with_workspace:
            assert set(buffers.workspace.slots) == {"x", "y"}
