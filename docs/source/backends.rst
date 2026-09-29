.. _backends:

Scientific backends and execution support
=========================================

Choose a scientific implementation separately from the workflow scheduler.
``execution.profile`` controls job admission and input reuse; ``execution_profile``
controls numerical conventions. Neither automatically selects CUDA.

.. list-table:: Backend contracts
   :header-rows: 1
   :widths: 18 28 27 27

   * - Backend
     - Modules and dependencies
     - Scheduling and resume
     - Injection and output scope
   * - Native CPU
     - ``coherence_native``, ``super_cluster_native``, ``likelihoodWP``;
       NumPy, Numba, JAX CPU, WDM
     - Native simple workflow; scalable executor supports input reuse.
       Shared-input lag workers fall back to serial for injections.
       Catalog progress supports resume.
     - Injection trials, catalog records, saved waveforms and plots follow
       native configuration. Numerical reference and end-to-end tests apply.
   * - JAX likelihood (experimental)
     - ``likelihoodWPGPU``; JAX device backend, mostly FP32 kernels
     - A separate likelihood implementation; it is not the CUDA workflow
       selected below. Scheduler/resume guarantees require a compatible
       processor adapter; no equivalent whole-job matrix is claimed.
     - Do not infer CPU/CUDA parity or injection support from the package name.
       Its kernel tests do not certify every full-pipeline configuration.
   * - CUDA workflow (experimental)
     - ``coherence_gpu``, ``super_cluster_gpu``, ``likelihood_gpu``,
       ``pycwb.utils.gpu``; NVIDIA driver, bundled NVRTC and JAX CUDA wheels, x64 enabled
     - Simple or scalable job scheduling; spawned lag workers own device state.
       Parent commits catalog progress. Resume requires matching recorded options.
     - Whole-job performance/parity evidence covers catalog-only LF/HF/LD
       background. Injections use serial lag dispatch, but this is not equivalent
       to full CUDA injection validation. Product restrictions depend on the
       output options described below.
   * - ROOT compatibility
     - Legacy ROOT-backed modules and cWB library
     - Legacy workflow contracts; scalable processors without an input-provider
       contract use supervised direct reads and receive no cache-reuse guarantee.
     - Follow the selected legacy processor's contracts. ROOT support does not
       imply interchangeability with native payloads.

CUDA workflow assembly
----------------------

Add this overlay to an otherwise valid search YAML::

    segment_processer: pycwb.workflow.subflow.process_job_segment_gpu.process_job_segment
    execution_profile:
      scalar_dpf: true
    gpu:
      selection_cuda: true
      dpf: true
      likelihood: true
      lag_workers: 1
      output_batch: 1

Start with one worker. More workers require measured CPU/RAM/device capacity;
worker counts are limits, not speed guarantees. Run ``pycwb validate`` first,
then a representative paired validation job (``gpu.validate_stages: true``).
Validation repeats native work and must be disabled for performance measurements.
The driver, available devices, wavelet sizes, frame data and segment injections
are checked at runtime; offline validation cannot establish their suitability.

The native orchestration accepts explicit, keyword-only backend callbacks.
``pycwb.types.stages`` defines the scientific stage interfaces; factories create
process-owned callbacks and workflow stage bundles assemble them. Modules may
reuse native payloads and domain algorithms without importing the workflow.
CUDA kernel handles are cached by source, architecture and context identity;
reset/unloaded handles are rebuilt on the next load. Recreate stage objects after
resetting a device: their existing buffers also belong to the old context.

Output combinations
-------------------

* With ``output_batch: 1`` and ``q_reconstruction: false``, the parent uses
  native reconstruction and save logic. The workflow does not categorically
  reject all plots, waveforms or injections.
* ``output_batch > 1`` rejects saved waveforms and segments with injections.
  Trigger records are committed before progress; uncommitted lags rerun on resume.
* ``q_reconstruction: true`` computes only the whitened REC/DAT products needed
  by Q-veto. It rejects injections, saved products and plots; use native
  reconstruction when those products are needed.
* ``dpf: true`` with ``execution_profile.scalar_dpf: false`` is rejected both
  offline and at runtime.
* ``worker_output: true`` is retired. Remove it to use parent output. A historical
  128-lag LF comparison found exact records but slower execution; maintaining a
  second output path provided no measured benefit. False remains readable in
  old configuration snapshots. If an existing catalog recorded true, use a new
  working directory; recorded options must not be rewritten to bypass resume checks.

These combinations are covered by ``tests/test_runtime_validation.py`` and the
GPU output, pipeline, resume and numerical tests. Numerical tests skip when CUDA
is unavailable; a CPU-only CI pass is not CUDA evidence. Historical measurements
and retained alternatives are described in :doc:`validation_status` and
``docs/dev/quality_cleanup.md`` in the source tree.

GPU option reference
--------------------

Generated from the same nested schema used for configuration validation. All
boolean defaults are false; CPU preparation and lag worker counts default to one.

.. exec::

    from pycwb.constants.gpu_options import GPU_SCHEMA
    from pycwb.utils.generate_params_table import generate_rst_table
    params = {}
    for key, value in GPU_SCHEMA['properties'].items():
        item = dict(value)
        if 'maximum' in item:
            item['description'] += ' Allowed range: %s–%s.' % (item['minimum'], item['maximum'])
        params['gpu.' + key] = item
    print(generate_rst_table(params))

Compatibility and naming
------------------------

New code imports the stage packages listed above. ``background_cuda`` is the
legacy import namespace; aliases remain where they preserve cache identity.
Its frame-read and conditioning adapters translate old GPU settings into the
explicit CPU scheduling arguments. Tests retained under ``background_cuda/tests``
cover both CUDA parity and these compatibility paths.

The retired worker-output import is removed with its experimental implementation.
The old chirp-plan import remains an alias to shared host helpers. Existing YAML
spellings ``segment_processer`` and ``parallel_injection_trail`` remain supported.
The corrected ``optimize_sky_loc_from_td`` also retains the historical misspelled
import alias. Avoid renaming persisted configuration keys without a migration.
