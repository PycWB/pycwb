import ctypes
import gc


def release_memory():
    """Run the garbage collector and return fragmented heap pages to the OS.

    Calls ``gc.collect()`` followed by ``malloc_trim(0)`` on Linux (glibc).
    On non-Linux platforms or systems using musl libc, only the GC pass runs.
    """
    gc.collect()
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except (OSError, AttributeError):
        pass  # Non-Linux or musl libc
