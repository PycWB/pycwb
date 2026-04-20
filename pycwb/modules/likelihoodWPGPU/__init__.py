"""
likelihoodWPGPU — JAX-accelerated coherent likelihood for gravitational wave bursts.

This module is a GPU-optimized reimplementation of ``likelihoodWP`` using JAX.
It provides the same external interface (``setup_likelihood``, ``likelihood``,
``likelihood_wrapper``) so it can be used as a drop-in replacement.

Key design differences from the CPU module:

- All inner kernels are written in JAX (``jax.numpy`` + ``jax.jit``).
- The sky scan uses ``jax.vmap`` over sky directions instead of ``numba.prange``.
- Array layouts are GPU-optimal: pixel axis last (contiguous), IFO as a small
  explicit dimension, sky as the batch/grid axis.
- Mathematical variable names follow the notation in ``docs/likelihood/likelihoodWP.md``
  rather than the C++ AVX register names.
- All computation uses float32 (FP32).
"""

from .likelihood import setup_likelihood, likelihood, likelihood_wrapper
