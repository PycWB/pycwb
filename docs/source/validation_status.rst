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
   * - ``pycwb doctor``
     - Selected dependencies import and JAX exposes a device
     - End-to-end execution or numerical agreement
   * - ``pycwb validate``
     - YAML/schema validity, sky units, execution/GPU settings and detector definitions
     - Data availability, all cross-field constraints or scientific suitability
   * - ``pycwb demo --run``
     - One loud SGE injection is recovered by the native H1/L1 CPU workflow
     - FAR calibration, sensitivity, other networks or GPU equivalence
   * - Regular CI tests
     - Regression checks selected in ``.gitlab-ci.yml`` on the native Linux image
     - A complete operating-system/Python/ROOT matrix
   * - Injection consistency job
     - The explicitly invoked reference fixtures and comparison tolerances
     - Unexamined configurations; this CI job is currently manual

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
