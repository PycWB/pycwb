.. _migration:

Upgrading Existing Analyses
===========================

The unreleased native-processing changes include Python API and numerical
changes. Loading the same YAML does not promise the same scientific results.
Record the exact software revision and compare a fresh run before combining
productions. See :ref:`reproducibility` and :ref:`validation_status` for the
archiving procedure and the scope of validation.

Environment and prepared runs
-----------------------------

* Install Python 3.11 or newer, ``wdm-wavelet>=0.4.0`` and
  ``burst-waveform>=0.5.0``. ``joblib>=1.3`` is now a direct dependency.
* Complete old prepared runs in their original environment if their catalogs
  lack a YAML snapshot or execution profile. Otherwise regenerate in a new
  directory. Neither overwrite flags nor copying a new YAML into the old
  directory supplies the missing provenance.
* Current workers load YAML and check analysis settings against the saved
  snapshot. Submission memory, disk and walltime can change on resubmission.
* Move retired processing environment switches into YAML. For example,
  ``PYCWB_REGRESSION_ENGINE`` becomes ``execution_profile.regression_engine``,
  ``WDM_BOUNDED_NUMBA`` becomes ``execution_profile.wdm_bounded_numba``, and
  ``PYCWB_GPU_LIKELIHOOD`` becomes ``gpu.likelihood``. Config loading warns
  about recognized retired switches but never applies their values. See
  :ref:`execution_profile_options` for the available settings.
* Scheduler scripts allocate parallel batch workers, with one processing thread
  per job by default. ``batch-runner --n-proc 0`` uses YAML ``nproc``; account
  for both worker count and per-job threads when allocating CPUs.

Master and merged catalogs reference ``jobs.parquet`` using a relative path
and a manifest identity. Copy the catalog, referenced manifest and progress
together. Read job metadata through ``Catalog.jobs`` with the current software;
older readers that inspect only inline ``jobs`` metadata can silently lose
exposure. Old inline-job catalogs remain readable for postproduction.

When calling ``process_background``, pass a catalog filename to resolve this
reference. If passing an in-memory Arrow table with a manifest reference,
provide ``unshifted_job_ids`` explicitly. The table alone has no source directory
from which to resolve the manifest. Missing job information must not turn a
shifted job's regular lag zero into physical zero lag.

Python callers and notebooks
----------------------------

The native API renames are intentional. Update imports and calls together:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Previous name
     - Current name
   * - ``data_conditioning`` / ``data_conditioning_single``
     - ``condition_strains`` / ``condition_strain``
   * - ``regression_python``
     - ``apply_regression``
   * - ``whitening_python`` / ``whitening_mesa_python``
     - ``whiten_wavelet`` / ``whiten_mesa``
   * - ``whitening_mdc``
     - ``whiten_injection_strain`` with an existing noise map
   * - ``PSD_correction.psd_correction_python``
     - ``psd_correction.apply_psd_correction``
   * - ``setup_likelihood``
     - ``prepare_likelihood_inputs``
   * - ``likelihood`` / ``likelihood_wrapper`` in ``likelihoodWP``
     - ``evaluate_cluster_likelihood`` / ``evaluate_fragment_clusters``
   * - ``likelihoodWP.typing``
     - ``likelihoodWP.results``

Conditioning entries above are relative to ``pycwb.modules.data_conditioning``;
likelihood entries are relative to ``pycwb.modules.likelihoodWP``. The module
filenames ``data_conditioning.py`` and ``likelihood.py`` still exist, so importing
an old function name can return a module instead of a callable.
``condition_strains`` no longer accepts the unused ``nproc`` argument.
Use the current :doc:`tutorial_search` and :doc:`tutorial_injection` examples.

For direct sky scans, use the Python ``likelihoodWP.sky_scan.scan_sky`` wrapper
with geometry, cluster-data and settings tuples. The compiled ``scan_sky_kernel``
requires delay-group arrays; it is not a signature-compatible rename of
``scan_sky_for_best_fit``.

``max_delay``, ``compute_sky_delay_and_patterns`` and ``project_to_detector``
now take initialized detector objects. Pass ``config.detectors`` to retain
configured custom geometry, and use the ``detectors=`` keyword in place of
``ifos=``. For standalone standard geometry, construct ``Detector("H1")`` and
``Detector("L1")`` explicitly. Default geometry remains LAL; selecting a cWB
geometry is a separate explicit choice.

Numerical and postproduction changes
------------------------------------

* Delay splitting follows cWB; subnet delays are rounded at the analysis rate
  before upsampling. TD buffers cover both the rounded subnet and likelihood
  grids. Defragmentation frequency gaps are measured in Hz. Pixel noise RMS
  uses the noise maps, packet norms use float32 arithmetic, and reduced
  correlation uses the cWB ``0.001`` offset. The ``BATCH`` pixel cap is enforced;
  set ``BATCH: 0`` explicitly for an uncapped likelihood.
* Regression demeans its self-witness. Reconstructed sky position, pixel
  statistics, waveform bounds and Q-veto values can change. Additional
  ``native_chirp`` and ``release_waveform_stats`` execution-profile options
  remain opt-in; their defaults are false.
* Omitted ``lagOff`` and ``lagMax`` now default to zero, replacing 6 and 150.
  Preserve explicit recorded settings when continuing an existing run.
* Target-SNR injections are scaled and placed at fractional-sample offsets.
  The new ``injection_resampling: cwb`` default uses the cWB target-SNR
  resampling and arrival-time conventions. Mixed target-SNR/fixed-hrss trials
  require separate trials in this mode. ``fft`` selects the FFT path but does
  not revert waveform-library or other numerical fixes. Burst waveform
  conventions and signal-only whitening have also changed; see
  :doc:`tutorial_injection`.
* Retrain native-catalog XGBoost models: ``norm``, ``sSNR`` and detector-order
  feature definitions changed. Prediction cuts are applied, and model
  preprocessing compatibility is checked before scoring. Changing a metadata
  declaration does not convert a model's feature definitions.
* ``compute_hrss50`` and ``plot_efficiency_vs_hrss`` require the right-matched
  simulation table, including missed injections. ``use_unique_sim`` must be
  true; duplicate simulation IDs and target-SNR populations are rejected for
  fixed-hrss efficiency. cWB XGB ranking requires coherent energy and penalty.
  IFAR labels accept ``s``, ``day``, ``wk``, ``mo`` and ``yr`` or positive seconds;
  replace ``1year``, ``12h`` and ``1d`` with ``1yr``, ``43200s`` and ``1day``.
* Efficiency uses the specified detection boundary, unbracketed 50-percent
  crossings return no point estimate, tied rankings share a FAR, and report
  background exposure excludes physical zero lag. See
  :doc:`postproduction_efficiency` for the statistical conventions.
