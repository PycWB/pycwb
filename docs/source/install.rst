.. _installing_pycwb:

Installation and Versions
=========================

This documentation describes PycWB |release|. Run ``pycwb --version`` and
check the documentation version before copying a configuration or command.
The ``latest`` documentation tracks development and can contain unreleased
features. A plain ``pip install pycwb`` normally selects a stable release.

Choose a release channel
------------------------

**Stable release:** install the version appropriate for your analysis, then
select that version in the documentation. The PyPI release description contains
its installation requirements; older ROOT-based releases have different
requirements from the native Python path described below.

.. code-block:: bash

   python -m pip install pycwb
   pycwb --version

**Prerelease:** alpha releases require an explicit version or ``--pre``. An
explicit version is preferable when preserving an analysis environment.

.. code-block:: bash

   python -m pip install --pre pycwb
   pycwb --version

Check the `PyPI release history <https://pypi.org/project/PycWB/#history>`_
and :ref:`release_policy`. Do not assume a prerelease contains every feature
shown in the development documentation.

Source installation for this development guide
----------------------------------------------

The current native Python path requires Python 3.11 or newer. The Linux CI
container uses Python 3.13. Use a separate environment:

.. code-block:: bash

   conda create -n pycwb -c conda-forge python=3.13 pip nds2-client python-nds2-client lalsuite python-ligo-lw
   conda activate pycwb
   git clone https://git.ligo.org/yumeng.xu/pycwb.git
   cd pycwb
   python -m pip install .
   pycwb --version
   pycwb doctor

To reproduce an existing analysis, check out its recorded tag or commit before
installing. To develop code, use ``python -m pip install -e ".[test]"``.
See :ref:`dev_setup` for contributor details.

ROOT is optional for the native search. Native dependencies still include
compiled scientific libraries; "native Python path" does not mean every
dependency is implemented solely in Python.

Platform coverage
-----------------

.. list-table::
   :header-rows: 1

   * - Environment
     - Evidence and limitations
   * - Linux x86_64, Python 3.13, CPU
     - Native CI container and automated tests. Recommended starting point.
   * - Other Python versions >=3.11
     - Permitted by package metadata; no full multi-version CI matrix yet.
   * - macOS Intel / Apple Silicon, CPU
     - Dependency availability must be checked locally; not covered by the current CI.
   * - Windows / WSL2
     - Native Windows is not tested. A Linux environment under WSL2 is a possible route, not a validated configuration.
   * - GPU backends
     - Require backend-specific dependencies and validation. The beginner demo uses the CPU workflow.
   * - ROOT interoperability
     - Separate optional build and reference tests; not implied by native CI success.

Optional components
-------------------

.. code-block:: bash

   python -m pip install 'pycwb[xgboost]'  # postproduction classification
   python -m pip install 'pycwb[pycbc]'    # PyCBC waveform integrations

For a source checkout, use ``'.[xgboost]'`` or ``'.[pycbc]'`` to retain the same
code version. NDS2 client libraries are available through conda-forge rather
than a complete pip-only installation route.

For ROOT/C++ interoperability, install compatible ``root`` and ``healpix_cxx``
packages in a dedicated environment before building the optional extension.
See :ref:`dev_cxx_core`. This is not needed for :ref:`start_here`.

Build the documentation
-----------------------

From a source checkout with its runtime dependencies installed:

.. code-block:: bash

   python -m pip install -r docs/requirements.txt
   make doc-check

HTML is written to ``docs/build/html``. API pages and CLI help are generated
automatically for both local and hosted builds; edit the source docstrings and
parser definitions rather than generated ``pycwb*.rst``, ``modules.rst`` or
``_cli_help.rst.inc`` files.
