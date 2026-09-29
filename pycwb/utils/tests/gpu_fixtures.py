"""Shared fixtures and GPU availability gating for the modular GPU tests.

Every test that touches a CUDA device carries the ``gpu`` marker; tests that
additionally need a JAX GPU backend carry ``jax_gpu``. Both markers are turned
into clean skips when the corresponding device is not available, so the CPU-only
CI selection passes without a GPU.
"""

from __future__ import annotations

import os
from functools import lru_cache

# JAX must see x64 before its first import anywhere in the test process.
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pytest


@lru_cache(maxsize=1)
def _cuda_available() -> bool:
    """Return whether the package's CUDA stages can run in this process.

    A Numba driver context must be creatable and the bundled NVRTC compiler
    must be loadable: a machine with a driver but without the CUDA wheels
    (the CPU-only development environment) cannot compile any kernel.
    """
    try:
        from numba import cuda

        cuda.current_context()
        from pycwb.utils.gpu.cuda_runtime import _nvrtc_library

        _nvrtc_library()
    except Exception:  # noqa: BLE001 - any driver/library failure means "no GPU"
        return False
    return True


@lru_cache(maxsize=1)
def _jax_gpu_available() -> bool:
    """Return whether JAX exposes a GPU backend with x64 enabled."""
    if not _cuda_available():
        return False
    try:
        import jax

        if not jax.config.x64_enabled:
            return False
        return len(jax.devices("gpu")) > 0
    except Exception:  # noqa: BLE001 - missing backend or plugin means "no GPU"
        return False


def pytest_configure(config: pytest.Config) -> None:
    """Register the GPU markers so ``--strict-markers`` runs stay warning-free."""
    config.addinivalue_line(
        "markers", "gpu: test needs a CUDA device (Numba driver context)"
    )
    config.addinivalue_line(
        "markers", "jax_gpu: test needs a JAX GPU backend with x64 enabled"
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip ``gpu``/``jax_gpu`` marked tests when the device is unavailable."""
    for item in items:
        if item.get_closest_marker("gpu") is not None:
            item.add_marker(pytest.mark.skipif(
                not _cuda_available(), reason="requires a CUDA device visible to Numba"
            ))
        if item.get_closest_marker("jax_gpu") is not None:
            item.add_marker(pytest.mark.skipif(
                not _jax_gpu_available(), reason="requires a JAX GPU backend with JAX_ENABLE_X64=1"
            ))


@pytest.fixture
def rng() -> np.random.Generator:
    """Deterministic generator shared by the synthetic parity tests."""
    return np.random.default_rng(20260921)


@pytest.fixture(params=[False, True], ids=["fresh_buffers", "reuse_workspace"])
def reuse_workspace(request):
    return {"reuse_workspace": request.param}
