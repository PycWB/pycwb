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


.. _building_documentation:

Build the documentation
-----------------------

From a source checkout with its runtime dependencies installed:

.. code-block:: bash

   python -m pip install -r docs/requirements.txt
   python -m pip install --no-deps -e .
   make doc-check

Installing the checkout generates ``pycwb/_version.py``, which is not tracked
in Git and is required when Sphinx imports the package.

HTML is written to ``docs/build/html``. API pages and CLI help are generated
automatically for both local and hosted builds; edit the source docstrings and
parser definitions rather than generated ``pycwb*.rst``, ``modules.rst`` or
``_cli_help.rst.inc`` files.

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
against the reviewed baseline. Mypy checks the boundary files listed in
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

Backend checks
--------------

The output combinations in :doc:`backends` are checked by
``tests/test_runtime_validation.py`` and the GPU output, pipeline, resume and
numerical tests. Numerical tests skip when CUDA is unavailable, so run them on
a CUDA host for GPU changes. Whole-job comparisons cover catalog-only LF/HF/LD
background; injection behavior needs its own end-to-end checks.

A historical 128-lag LF comparison of worker-side output found matching
records but slower execution. The worker-output path was retired; the parent
now writes output. See ``docs/dev/quality_cleanup.md`` for the measurements.

.. _detector_geometry_reference:

Detector-geometry reference
---------------------------

The reference vectors come from ``wat/detector.cc`` in cWB release 6.4.6.9,
commit ``e03cf7f``. This revision identifies the test oracle; configuration uses
``H1:cwb`` and ``L1:cwb``.

The cWB definition was added to reproduce release outputs. PycWB's bundled
LAL-derived definition reconstructs vectors from geographic angles, whereas
cWB uses literal rounded vectors. The geometry audit measured:

.. list-table::
   :header-rows: 1
   :widths: 15 30 55

   * - Detector
     - Vertex displacement
     - Maximum absolute antenna-response difference in sampled sky
   * - H1
     - 5.367 mm
     - 3.006 × 10\ :sup:`−6`
   * - L1
     - 3.524 mm
     - 2.921 × 10\ :sup:`−6`

The antenna comparison covered 4,099 directions. These are absolute differences,
not relative errors or bounds over the continuous sky. With matched literal
vectors, differences from the cWB antenna oracle were below 1.6 × 10\ :sup:`−15`.
The stored oracle and its provenance are under
``pycwb/types/tests/reference/RELEASE_GEOMETRY.md``.

These checks establish reproduction of cWB, not which set is physically closer
to the surveyed instrument. More decimal digits alone do not establish physical
accuracy. Keep the bundled default to preserve existing PycWB geometry; use
``:cwb`` when comparing against the validated cWB reference. Neither choice is
a performance preset.

Regression target and witness
-----------------------------

Both native regression engines (Numba and JAX) preserve the original target
transform and transform a separate, mean-subtracted self-witness, following
cWB's witness preparation. Cross-correlations use both transforms; the filter
matrix and capped filter input use the witness. Predicted noise is restored
with the target normalization. The caller's strain array is not modified.

This correction can change regression trim decisions and downstream event
parameters for nonzero-mean input. Use a new run directory when comparing with
results generated before this correction and retain the source revision.
The independent cWB sliced-RMS normalization discrepancy is not emulated.
