"""
likelihoodWPGPU — JAX-accelerated coherent likelihood for gravitational wave bursts.

This module is a GPU-optimized reimplementation of ``likelihoodWP`` using JAX.
It shares input preparation with CPU likelihood and exports ``prepare_likelihood_inputs``,
``likelihood``, and ``likelihood_wrapper`` for GPU cluster evaluation.

Key design differences from the CPU module:

- All inner kernels are written in JAX (``jax.numpy`` + ``jax.jit``).
- The sky scan uses ``jax.vmap`` over sky directions instead of ``numba.prange``.
- Array layouts are GPU-optimal: pixel axis last (contiguous), IFO as a small
  explicit dimension, sky as the batch/grid axis.
- Mathematical variable names follow the notation in ``docs/likelihood/likelihoodWP.md``
  rather than the C++ AVX register names.
- All computation uses float32 (FP32).
"""

from .likelihood import prepare_likelihood_inputs, likelihood, likelihood_wrapper

__all__ = ["likelihood", "likelihood_wrapper", "prepare_likelihood_inputs"]
