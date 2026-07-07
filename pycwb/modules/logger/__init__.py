"""
pycwb.modules.logger — Structured logging initialization.

Configures log format, output destination (file/stdout), and log level.
Pins noisy external libraries (JAX, Numba, matplotlib) to WARNING to
keep logs readable.
"""

from .logger import logger_init, log_prints