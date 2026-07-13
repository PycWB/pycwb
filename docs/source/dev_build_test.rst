.. _dev_build_test:

Build & Test
============

How to build, test, and verify pycWB during development.

.. contents:: Table of Contents
   :depth: 2
   :local:


Building
--------

**Pure-Python (no C++ compilation needed):**

.. code-block:: bash

   pip install -e .

**With C++ core:**

.. code-block:: bash

   make build_cwb
   # or
   python setup.py build_cwb

The C++ core (``cwb-core/``) is built via CMake → ``build.sh`` → ROOT/PyROOT
bindings. This step is optional and only needed for the ROOT-backed wavelet
extension or legacy ROOT I/O.


Running Tests
-------------

**All tests:**

.. code-block:: bash

   pytest
   # or
   python -m unittest discover tests/

**Unit tests only** (per module):

.. code-block:: bash

   pytest pycwb/modules/skymask/tests/
   pytest pycwb/modules/super_cluster_native/tests/
   pytest pycwb/modules/likelihoodWP/tests/

**Specific test file:**

.. code-block:: bash

   pytest pycwb/modules/catalog/tests/test_catalog.py -v

**With coverage:**

.. code-block:: bash

   pip install pytest-cov
   pytest --cov=pycwb --cov-report=html


Test Categories
---------------

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Category
     - Location
     - Purpose
   * - Unit tests
     - ``pycwb/modules/*/tests/``
     - Test individual module functions in isolation
   * - Integration tests
     - ``tests/``
     - End-to-end pipeline with synthetic data
   * - Numerical parity
     - ``tests/compare_with_cwb/``
     - Compare pycWB native vs. cWB ROOT results
   * - Performance benchmarks
     - ``benchmark/``, ``_test_njit.py``
     - Numba/JAX warm-up and throughput benchmarks

Unit tests use Python's ``unittest`` framework by convention.


Continuous Integration
----------------------

CI runs on LIGO GitLab via ``.gitlab-ci.yml``. The pipeline includes:

- Build (pure-Python and ROOT variants)
- Unit tests (multiple Python versions)
- Integration tests
- Linting / static analysis

Badges in the README show current build and test status.


Test Conventions
----------------

When adding tests:

- Place unit tests in ``pycwb/modules/<module>/tests/`` alongside the code.
- Use ``unittest.TestCase`` for new unit tests.
- Name test files ``test_<feature>.py``.
- Use descriptive test method names: ``test_<function>_<scenario>_<expected>``.
- Mock external dependencies (ROOT, gwdatafind, GraceDB) rather than requiring
  real services.
- For Numba/JAX functions, test both the Python and compiled paths.


Verifying Before PR
-------------------

.. code-block:: bash

   # Full check
   pytest
   python -m unittest discover tests/

   # Build docs (optional — check for warnings)
   cd docs && make html

   # Check for import issues
   python -c "import pycwb; print('OK')"
