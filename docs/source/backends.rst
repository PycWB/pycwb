.. _backends:

Scientific backends and execution support
=========================================

Choose a scientific implementation separately from the workflow scheduler.
``execution.profile`` controls job admission and input reuse; ``execution_profile``
controls numerical conventions and processing options. Select the CUDA
processor explicitly with ``segment_processer``, as shown below.

.. list-table:: Backend options
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
       native configuration.
   * - JAX likelihood (experimental)
     - ``likelihoodWPGPU``; JAX device backend, mostly FP32 kernels
     - A separate likelihood implementation; it is not the CUDA workflow
       selected below. Scheduling and resume depend on the processor adapter.
     - A likelihood-stage implementation; use the native or CUDA workflow
       for the complete search and its output options.
   * - CUDA workflow (experimental)
     - ``coherence_gpu``, ``super_cluster_gpu``, ``likelihood_gpu``,
       ``pycwb.utils.gpu``; NVIDIA driver, bundled NVRTC and JAX CUDA wheels, x64 enabled
     - Simple or scalable job scheduling; spawned lag workers own device state.
       Parent commits catalog progress. Resume requires matching recorded options.
     - Injections use serial lag dispatch. Saved products and injection support
       depend on ``output_batch`` and ``q_reconstruction``; see below.
   * - ROOT compatibility
     - Legacy ROOT-backed modules and cWB library
     - Legacy processors without an input-provider adapter read data directly
       instead of using the scalable executor's input cache.
     - Uses ROOT-backed data structures; see :doc:`dev_cxx_core`.

.. raw:: html

   <span id="regression-target-and-witness"></span>

For regression and whitening, see :doc:`data_conditioning`.

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

Start with one worker and measure CPU, host-memory and device-memory use before
increasing concurrency. ``gpu.validate_stages: true`` compares CUDA stages with
the native calculation; disable it when measuring performance because it
repeats that work.


Output combinations
-------------------

* With ``output_batch: 1`` and ``q_reconstruction: false``, the parent uses
  native reconstruction and save logic, including waveforms, plots and injections.
* ``output_batch > 1`` rejects saved waveforms and segments with injections.
  Trigger records are committed before progress; uncommitted lags rerun on resume.
* ``q_reconstruction: true`` computes only the whitened REC/DAT products needed
  by Q-veto. It rejects injections, saved products and plots; use native
  reconstruction when those products are needed.
* ``dpf: true`` with ``execution_profile.scalar_dpf: false`` is rejected both
  offline and at runtime.
* Remove the retired ``worker_output: true`` option to use parent output.
  If an existing catalog recorded it as true, start a new working directory.
  Old snapshots with ``worker_output: false`` remain readable.


GPU option reference
--------------------

All boolean defaults are false; CPU preparation and lag worker counts default
to one.

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

The YAML spellings ``segment_processer`` and ``parallel_injection_trail``
remain supported. Use the processor path in the example above to select the
CUDA workflow.
