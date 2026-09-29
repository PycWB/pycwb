"""Context lifetime regressions do not require a CUDA device."""
import weakref
from types import SimpleNamespace

from pycwb.utils.gpu import cuda_runtime as runtime


class Context:
    """Own simulated driver handles until reset."""
    __hash__ = None  # Like Numba Context: identity must not rely on hashability.

    def __init__(self):
        self.modules = []


class Handle:
    handle = 123


def test_module_cache_is_context_owned_and_reloads_after_reset(tmp_path, monkeypatch):
    contexts = [Context(), Context()]
    active = [contexts[0]]
    monkeypatch.setattr(runtime.cuda, 'current_context', lambda: active[0])
    monkeypatch.setattr(runtime.cuda, 'get_current_device', lambda: SimpleNamespace(compute_capability=(8, 9)))
    monkeypatch.setattr(runtime, '_modules', {})

    def compile_module(source, name):
        handle = Handle()
        active[0].modules.append(handle)
        return SimpleNamespace(module=weakref.proxy(handle))

    monkeypatch.setattr(runtime, 'CUDAModule', compile_module)
    path = tmp_path / 'kernel.cu'
    path.write_text('kernel')
    first = runtime.load_module(path)
    assert runtime.load_module(path) is first
    active[0] = contexts[1]
    second = runtime.load_module(path)
    assert second is not first
    active[0] = contexts[0]
    assert runtime.load_module(path) is first
    contexts[0].modules.clear()
    assert runtime.load_module(path) is not first


def test_cache_does_not_keep_context_alive(tmp_path, monkeypatch):
    import gc
    context = Context()
    active = [context]
    monkeypatch.setattr(runtime.cuda, 'current_context', lambda: active[0])
    monkeypatch.setattr(runtime.cuda, 'get_current_device', lambda: SimpleNamespace(compute_capability=(8, 9)))
    monkeypatch.setattr(runtime, '_modules', {})
    monkeypatch.setattr(runtime, 'CUDAModule', lambda *_: SimpleNamespace(module=Handle()))
    path = tmp_path / 'kernel.cu'
    path.write_text('kernel')
    runtime.load_module(path)
    reference = weakref.ref(context)
    del context
    active.clear()
    gc.collect()
    assert reference() is None
    assert runtime._modules == {}
