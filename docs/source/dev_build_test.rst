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

   python -m pip install -e ".[test]"
   python -m pytest pycwb/ tests/ --ignore=tests/sample --ignore=tests/injection_consistency -m "not slow"

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

The test suite includes pytest tests and unittest test cases; pytest runs both.


Lint and Type Checks
--------------------

PycWB requires Python 3.11 or newer. Run the quality checks in a separate
Python 3.11 environment so mypy's target and the installed dependency stubs
describe the same Python version:

.. code-block:: bash

   python3.11 -m venv .venv-quality
   .venv-quality/bin/python -m pip install -r tools/quality/requirements.txt
   .venv-quality/bin/python tools/check_quality.py
   .venv-quality/bin/python -m mypy

The quality script rejects new lint, public-contract and import-boundary debt
against the reviewed baseline. Mypy checks the 13 boundary files listed in
``pyproject.toml``. This environment includes NumPy's real type stubs; the
NumPy pin in ``tools/quality/requirements.txt`` applies only to these checks.
The scientific runtime retains its separate ``numpy>=2`` requirement.
These static checks do not require importing or installing the full scientific
stack and do not replace runtime tests on the minimum supported Python.


Continuous Integration
----------------------

CI runs on LIGO GitLab via ``.gitlab-ci.yml``. The native Linux image uses
Python 3.13 and builds without the optional ROOT wavelet extension. The pipeline
runs unit/integration tests excluding slow and fixture-dependent cases, onboarding
tests and the standalone synthetic example against the installed package.
The independent quality job uses Python 3.11 and the pinned requirements above.
The ``installed-cli-smoke`` job runs pytest with ``python -I`` so both the CLI
and recovery checks import the installed wheel instead of the source checkout.
Documentation builds run on Read the Docs using ``.readthedocs.yaml``, with
Sphinx warnings treated as errors. Use ``make doc-check`` for a local check.
The injection-consistency reference job is manual. A multi-Python/OS matrix and
ROOT validation are not implied by the native test badge.


Test Conventions
----------------

When adding tests:

- Place unit tests in ``pycwb/modules/<module>/tests/`` alongside the code.
- Follow the conventions of the surrounding tests; pytest discovers both styles.
- Name test files ``test_<feature>.py``.
- Use descriptive test method names: ``test_<function>_<scenario>_<expected>``.
- Mock external dependencies (ROOT, gwdatafind, GraceDB) rather than requiring
  real services.
- For Numba/JAX functions, test both the Python and compiled paths.


Verifying Before PR
-------------------

.. code-block:: bash

   python -m pytest tests/test_onboarding.py
   python -m pip install -r docs/requirements.txt
   make doc-check
   python -m build --wheel
   python -m pip install --force-reinstall --no-deps dist/*.whl
   python -m pytest tests/test_demo_e2e.py -m slow

The test invokes ``python -I -m pycwb run`` outside the checkout, so it
imports PycWB from the installed package with the example YAML copied from
the source checkout. The checkout and ``PYTHONPATH`` cannot replace installed
pipeline modules. Use a separate
environment for this wheel check. Set ``PYCWB_DEMO_XTALK`` to the absolute path
of a compatible local catalog to avoid downloading it.

Use a new demo directory for every verification run. Tests marked ``slow`` are
opt-in and require the documented reference data or network access.
