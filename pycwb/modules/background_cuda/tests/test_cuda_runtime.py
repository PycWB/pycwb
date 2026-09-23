"""NVRTC compilation, kernel launch and per-process module caching."""

from __future__ import annotations

import ctypes as ct
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.gpu

TRIVIAL_SOURCE = """
extern "C" __global__ void fill_scaled(const double* input, double* output, int count, double scale) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    output[index] = input[index] * scale + 1.0;
}
"""

BROKEN_SOURCE = """
extern "C" __global__ void broken(double* output) {
    output[0] = undefined_symbol;
}
"""


def test_compile_and_launch_trivial_kernel() -> None:
    from numba import cuda

    from pycwb.modules.background_cuda.cuda_runtime import DEFAULT_THREADS, CUDAModule

    module = CUDAModule(TRIVIAL_SOURCE, "trivial.cu")
    assert any(option.startswith("--gpu-architecture=sm_") for option in module.options)
    assert "--fmad=false" in module.options
    count = DEFAULT_THREADS * 3 + 5
    values = np.arange(count, dtype=np.float64)
    source = cuda.to_device(values)
    output = cuda.device_array(count, np.float64)
    module.launch("fill_scaled", count, [source, output, ct.c_int(count), ct.c_double(0.5)])
    result = output.copy_to_host()
    np.testing.assert_array_equal(result.view(np.uint64), (values * 0.5 + 1.0).view(np.uint64))


def test_launch_with_custom_block_size() -> None:
    from numba import cuda

    from pycwb.modules.background_cuda.cuda_runtime import CUDAModule

    module = CUDAModule(TRIVIAL_SOURCE, "trivial.cu")
    values = np.ones(33, dtype=np.float64)
    output = cuda.device_array(33, np.float64)
    module.launch("fill_scaled", 33, [cuda.to_device(values), output, ct.c_int(33), ct.c_double(2.0)], threads=32)
    np.testing.assert_array_equal(output.copy_to_host(), np.full(33, 3.0))


def test_compile_failure_reports_log() -> None:
    from pycwb.modules.background_cuda.cuda_runtime import CUDAModule

    with pytest.raises(RuntimeError, match="undefined_symbol"):
        CUDAModule(BROKEN_SOURCE, "broken.cu")


def test_load_module_caches_by_source_hash(tmp_path: Path) -> None:
    from pycwb.modules.background_cuda import cuda_runtime
    from pycwb.modules.background_cuda.cuda_runtime import CUDAModule, load_module

    first_path = tmp_path / "a.cu"
    second_path = tmp_path / "b.cu"
    first_path.write_text(TRIVIAL_SOURCE)
    second_path.write_text(TRIVIAL_SOURCE)
    first = load_module(first_path)
    assert isinstance(first, CUDAModule)
    assert load_module(first_path) is first
    # Same source under a different name is the same compiled module.
    assert load_module(second_path) is first
    third_path = tmp_path / "c.cu"
    third_path.write_text(TRIVIAL_SOURCE.replace("+ 1.0", "+ 2.0"))
    third = load_module(third_path)
    assert third is not first
    assert any(key[0] for key in cuda_runtime._modules)


def test_package_kernel_sources_compile_once_each() -> None:
    from pycwb.modules.background_cuda import cuda_runtime
    from pycwb.modules.background_cuda.cuda_runtime import load_module

    package = Path(cuda_runtime.__file__).parent
    for name in ("dpf_regulator", "likelihood_scan", "subnet_scan", "selection_cuda", "chirp_bootstrap", "td_vectors"):
        path = package / f"{name}.cu"
        module = load_module(path)
        assert load_module(path) is module, name
