"""Compile CUDA C with the NVRTC library bundled in the JAX CUDA wheels.

Compiling to a native cubin for the current device avoids the driver's PTX
version limit when the bundled compiler is newer than the installed driver.
Numba is used only for the context, device buffers and kernel launches, never
for NVVM compilation, so ``libNVVM`` does not need to be discoverable.

Compiled modules are cached per process by source hash and compute capability:
every wrapper class that shares a kernel file reuses one cubin instead of
recompiling it, which is numerically neutral because the same source and
options produce the same code.
"""

from __future__ import annotations

import ctypes as ct
import hashlib
import importlib.util
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numba import cuda
from numba.cuda.cudadrv import driver

DEFAULT_THREADS = 128
"""Default one-dimensional block size used by :meth:`CUDAModule.launch`."""

_NVRTC_OPTIONS = (
    "--std=c++11",
    "--fmad=false",
    "--ftz=false",
    "--prec-div=true",
    "--prec-sqrt=true",
)
"""Options that keep IEEE semantics: no fused multiply-add, no flush-to-zero."""

_modules: dict[tuple[str, tuple[int, int]], CUDAModule] = {}


def _nvrtc_library() -> ct.CDLL:
    """Load the newest bundled ``libnvrtc`` by parsed version number."""
    spec = importlib.util.find_spec("nvidia")
    roots = spec.submodule_search_locations if spec is not None else None
    if not roots:
        raise RuntimeError("No bundled NVIDIA CUDA wheels found on sys.path")
    paths = [p for root in roots for p in Path(root).glob("cu*/lib/libnvrtc.so.*")]
    if not paths:
        raise RuntimeError("No bundled CUDA NVRTC library found")

    def version(path: Path) -> tuple[int, ...]:
        suffix = path.name.split("libnvrtc.so.", 1)[1]
        return tuple(int(part) for part in suffix.split(".") if part.isdigit())

    return ct.CDLL(str(max(paths, key=version)))


class CUDAModule:
    """One compiled CUDA module and its launch helper.

    Parameters
    ----------
    source : str
        CUDA C source text.
    name : str, optional
        File name reported in compiler diagnostics.

    Attributes
    ----------
    module
        Numba module handle holding the loaded cubin.
    options : list[str]
        NVRTC options used for the compilation, including the architecture.

    Raises
    ------
    RuntimeError
        If NVRTC is unavailable or compilation fails; the compiler log is the
        error message.
    """

    def __init__(self, source: str, name: str = "background.cu") -> None:
        nvrtc = _nvrtc_library()
        program = ct.c_void_p()

        def check(code: int) -> None:
            if code:
                raise RuntimeError(f"NVRTC operation failed with status {code}")

        check(nvrtc.nvrtcCreateProgram(ct.byref(program), source.encode(), name.encode(), 0, None, None))
        try:
            major, minor = cuda.get_current_device().compute_capability
            options = [f"--gpu-architecture=sm_{major}{minor}", *_NVRTC_OPTIONS]
            packed = (ct.c_char_p * len(options))(*(option.encode() for option in options))
            status = nvrtc.nvrtcCompileProgram(program, len(options), packed)
            size = ct.c_size_t()
            check(nvrtc.nvrtcGetProgramLogSize(program, ct.byref(size)))
            log = ct.create_string_buffer(size.value)
            check(nvrtc.nvrtcGetProgramLog(program, log))
            if status:
                raise RuntimeError(log.value.decode())
            check(nvrtc.nvrtcGetCUBINSize(program, ct.byref(size)))
            cubin = ct.create_string_buffer(size.value)
            check(nvrtc.nvrtcGetCUBIN(program, cubin))
            self.module = cuda.current_context().create_module_image(cubin.raw)
            self.options = options
        finally:
            nvrtc.nvrtcDestroyProgram(ct.byref(program))

    def launch(self, name: str, count: int, args: Sequence[Any], threads: int = DEFAULT_THREADS) -> None:
        """Launch ``name`` on a one-dimensional grid covering ``count`` threads.

        Parameters
        ----------
        name : str
            ``extern "C"`` kernel symbol.
        count : int
            Number of logical threads; the grid is ``ceil(count / threads)``
            blocks. Kernels must guard ``index >= count`` themselves.
        args : sequence
            Kernel arguments in order: Numba device arrays are passed as raw
            device pointers, everything else must already be a ``ctypes`` scalar.
        threads : int, optional
            Block size. Kernels that use fixed-size shared memory must be
            launched with the block size they were written for.
        """
        function = self.module.get_function(name)
        packed = [
            ct.c_void_p(x.device_ctypes_pointer.value) if hasattr(x, "device_ctypes_pointer") else x for x in args
        ]
        driver.launch_kernel(
            function.handle,
            (count + threads - 1) // threads,
            1,
            1,
            threads,
            1,
            1,
            0,
            0,
            packed,
        )


def load_module(path: Path) -> CUDAModule:
    """Return the process-wide compiled module for a kernel source file.

    Parameters
    ----------
    path : pathlib.Path
        Path of the ``.cu`` file. Wrapper classes pass
        ``Path(__file__).with_suffix(".cu")``.

    Returns
    -------
    CUDAModule
        Cached per (source SHA-256, compute capability) in this process. The
        cache is deliberately in-process only: spawned lag workers own their
        CUDA context and compile once each.
    """
    source = path.read_text()
    capability = tuple(cuda.get_current_device().compute_capability)
    key = (hashlib.sha256(source.encode()).hexdigest(), capability)
    module = _modules.get(key)
    if module is None:
        module = CUDAModule(source, path.name)
        _modules[key] = module
    return module


class DeviceBuffers:
    """Uniform upload/output/download through an optional reusable workspace.

    Parameters
    ----------
    workspace : Workspace or None
        When given, named slots of the workspace are reused across calls and
        only the active bytes are transferred. When ``None`` every call
        allocates fresh device arrays through Numba, which is the behaviour of
        the original stage implementations.
    """

    def __init__(self, workspace: Any | None) -> None:
        self.workspace = workspace

    def upload(self, name: str, value: np.ndarray, dtype: Any) -> Any:
        """Copy ``value`` to the device as a contiguous array of ``dtype``."""
        data = np.ascontiguousarray(value, dtype=dtype)
        if self.workspace is None:
            return cuda.to_device(data)
        return self.workspace.upload(name, data)

    def output(self, name: str, shape: tuple[int, ...], dtype: Any) -> Any:
        """Return a device buffer able to hold ``shape`` values of ``dtype``."""
        if self.workspace is None:
            return cuda.device_array(shape, dtype)
        count = int(np.prod(shape)) if shape else 1
        return self.workspace.reserve(name, count, dtype)

    def download(self, buffer: Any, shape: tuple[int, ...]) -> np.ndarray:
        """Copy the first ``shape`` values of ``buffer`` back to the host."""
        if self.workspace is None:
            return buffer.copy_to_host()
        return self.workspace.download(buffer, shape)
