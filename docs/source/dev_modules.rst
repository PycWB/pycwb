.. _dev_modules:

Module Development
==================

How to add, modify, and test pycWB modules following project conventions.

.. contents:: Table of Contents
   :depth: 2
   :local:


Module Structure
----------------

Every module follows this layout:

.. code-block:: text

   pycwb/modules/<module_name>/
   ├── __init__.py
   ├── <module_name>.py       # Main implementation
   ├── <sub_feature>.py       # Optional sub-modules
   ├── utils.py               # Optional helpers
   └── tests/
       ├── __init__.py
       └── test_<feature>.py  # Unit tests


Adding a New Module
-------------------

1. **Create the directory** under ``pycwb/modules/``:

   .. code-block:: bash

      mkdir -p pycwb/modules/my_module/tests
      touch pycwb/modules/my_module/__init__.py
      touch pycwb/modules/my_module/tests/__init__.py

2. **Write the module** — a single entry function that takes a
   :py:class:`~pycwb.config.Config` and returns results:

   .. code-block:: python

      # pycwb/modules/my_module/my_module.py
      import logging

      logger = logging.getLogger(__name__)

      def process(config):
          """One-line summary of what this module does."""
          # Read config params
          param = config.some_param

          # Do work
          result = ...

          return result

3. **Wire into the pipeline** — add the call to
   ``pycwb/workflow/subflow/process_job_segment.py`` in the correct stage.

4. **Add tests** — see below.

5. **Register config parameters** — if your module needs new YAML parameters,
   add them to ``pycwb/constants/user_parameters_schema.py`` with defaults,
   types, and descriptions.


Module Communication
--------------------

Modules must **not** import sideways from each other. Communication flows
through:

- **Config**: parameters flow downward from :py:class:`~pycwb.config.Config`.
- **Return values**: each module returns plain Python objects or NumPy arrays.
- **Types**: shared data classes live in ``pycwb/types/`` (e.g.,
  ``WaveSegment``, ``Cluster``, ``PixelArrays``).

This keeps modules independently testable and avoidable of circular imports.


Choosing Numba vs JAX
---------------------

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Criterion
     - Use Numba
     - Use JAX
   * - Data pattern
     - Loops over time-delay batches
     - Batched vector/matrix operations
   * - Compilation
     - ``@njit`` — eager, first-call overhead
     - ``jit`` + ``vmap`` — cached in ``~/.cache/pycwb/jax_compilation_cache/``
   * - Parallelism
     - ``prange`` for CPU threads
     - Device-agnostic (CPU or GPU)
   * - Example
     - ``pycwb/utils/td_vector_batch.py``
     - ``pycwb/modules/coherence/coherence.py``
   * - Memory
     - Standard Python/NumPy
     - **Must explicitly free JAX device buffers after each lag**

**Rule of thumb**: prefer Numba for CPU-bound inner loops on small arrays;
prefer JAX for large batched operations that will benefit from GPU
acceleration in the future. Write JAX code in a device-agnostic way.
Never use pure NumPy for hot paths.


Writing Tests
-------------

.. code-block:: python

   # pycwb/modules/my_module/tests/test_my_module.py
   import unittest
   from pycwb.modules.my_module import process

   class TestMyModule(unittest.TestCase):

       def setUp(self):
           # Create minimal config for testing
           from pycwb.config import Config
           self.config = Config()
           self.config.some_param = 42

       def test_process_basic_returns_expected(self):
           result = process(self.config)
           self.assertIsNotNone(result)

       def test_process_edge_case_empty_input(self):
           self.config.some_param = 0
           result = process(self.config)
           self.assertEqual(result, [])


Config Schema Conventions
-------------------------

When adding parameters to ``user_parameters_schema.py``:

.. code-block:: python

   "my_new_param": {
       "type": "number",
       "default": 3.0,
       "minimum": 0.0,
       "maximum": 10.0,
       "description": "My new parameter controlling widget size [arbitrary units]"
   }

- Use descriptive names matching the module (``my_module_param`` not ``x``).
- Provide sensible defaults — users shouldn't need to set every parameter.
- Document units in the description.
- Auto-derived fields (``rateANA``, ``nRES``, ``WDM_level``, ``max_delay``)
  must **not** be set manually by users — they are computed from other
  parameters.


Deprecation Policy
------------------

- ROOT and C++ bindings are being phased out. Do not write new code that
  depends on them.
- Existing ROOT-dependent paths are guarded by
  :py:func:`pycwb.utils.check_ROOT.has_ROOT`.
- New wavelet/WDM code must use the pure-Python ``wdm-wavelet`` package.
