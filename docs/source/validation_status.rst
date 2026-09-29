.. _validation_status:

Validation Scope and Limitations
================================

Different checks support different claims. A successful installation or demo
does not establish equivalence to every cWB production configuration.

.. list-table::
   :header-rows: 1

   * - Check
     - What it establishes
     - What it does not establish
   * - ``pycwb validate``
     - YAML/schema validity, sky units, execution/GPU settings and detector definitions
     - Data availability, all cross-field constraints or scientific suitability
   * - Synthetic CLI recovery test (``tests/test_demo_e2e.py``)
     - When passing: one loud SGE injection is recovered through the installed CLI
     - FAR calibration, sensitivity, other networks or GPU equivalence
   * - Regular CI tests
     - Regression checks selected in ``.gitlab-ci.yml`` on the native Linux image
     - A complete operating-system/Python/ROOT matrix
   * - Injection consistency job
     - The explicitly invoked reference fixtures and comparison tolerances
     - Unexamined configurations; this CI job is currently manual

.. _platform_coverage:

Platform coverage
-----------------

.. list-table::
   :header-rows: 1

   * - Environment
     - Evidence and limitations
   * - Linux x86_64, Python 3.13, CPU
     - Native CI container and automated tests. Recommended starting point.
   * - Python 3.11
     - Minimum supported version; lint/type checks run in a separate 3.11 environment. Full runtime tests currently run on 3.13.
   * - Other Python versions >=3.11
     - Permitted by package metadata; no full multi-version runtime CI matrix yet.
   * - macOS Intel / Apple Silicon, CPU
     - Dependency availability must be checked locally; not covered by the current CI.
   * - Windows / WSL2
     - Native Windows is not tested. A Linux environment under WSL2 is a possible route, not a validated configuration.
   * - GPU backends
     - Require backend-specific dependencies and validation. The beginner demo uses the CPU workflow.
   * - ROOT interoperability
     - Legacy backend reference tests are separate from native CI. For ROOT-based analyses, use the original cWB; see :ref:`cwb_heritage`.

Reference comparisons
---------------------

``tests/injection_consistency/`` contains end-to-end comparison fixtures and
``tests/compare_with_cwb/`` contains reference-comparison tools. Additional
numerical tests live alongside the relevant scientific modules. Read the fixture
README/configuration and comparison code for the exact reference and tolerance.

For a scientific release, attach a validation record with:

* PycWB and cWB revisions, dependency versions and reference input checksums;
* detector networks, search bands, waveform families and processing backends;
* compared quantities, tolerances, measured differences and test commands;
* known exceptions and the release/configurations to which the record applies.

The presence of a comparison script is not evidence that it passed for the
release being used. No comprehensive release validation matrix is claimed by
this page. Experimental modules are identified in :ref:`modules_guide`.
