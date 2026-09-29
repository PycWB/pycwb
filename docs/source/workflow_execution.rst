.. _workflow_execution:
.. _experimental-workflow-execution:

Configure Resources and Resume Work
===================================

.. warning::

   The scalable execution layer is experimental and opt-in. Its interfaces and
   behavior may change. Validate representative workloads before using it for
   a campaign. The default remains ``execution.profile: simple``.

Use this guide to choose execution settings for your own workload and recover
unfinished work. For a small worked comparison, complete
:doc:`tutorial_resources`. The exact ``execution`` fields are also listed in
:doc:`schema`.

The ``pycwb.workflow.execution`` package handles resource-aware job planning,
raw-input reuse, worker supervision, and coordinated output writing. It provides
load balancing within an allocation by limiting concurrent segment workers
according to CPU availability and memory reservations. It also groups jobs that
share input frames, so decoded raw data can be reused.

Adaptive behavior currently covers bounded cache loading and memory-pressure
monitoring. It does not predict scientific processing cost or dynamically move
jobs between cluster nodes. Slurm/Condor batch membership is fixed when planned.

Configuration
-------------

The ``execution`` block controls job scheduling and resource management. It is
separate from ``execution_profile``, which controls native processing options
within each job (see :ref:`execution_profile`). Both blocks can appear in the
same analysis configuration.

Add a top-level ``execution`` block to an existing valid analysis YAML file:

.. code-block:: yaml

   execution:
     profile: scalable
     memory_limit: 32GiB
     worker_memory: 8GiB
     cache_limit: 2GiB
     headroom: 1GiB
     preload: auto
     batch_size: 8
     cores: 8

These values illustrate the syntax; they are not recommended limits for every
search. Measure representative jobs when choosing memory reservations.

The normal ``pycwb run``, ``pycwb batch-setup``, and ``pycwb batch-runner``
commands use this configuration. The YAML schema accepts the block, and its
settings are validated both when loading YAML and when restoring configuration
from a catalog. Unknown execution keys and invalid values are rejected.

Omit the block, or select ``profile: simple``, to retain the existing execution
path. Setting only memory or cache limits does not enable scalable execution.
Explicit ``planner`` or ``executor`` factory paths also enable dispatch through
this package and are intended for advanced extensions.

The scientific implementation is selected separately through
``segment_processer``. Scalable execution preserves scientific job windows and
does not adjust numerical selection settings to fit memory. Processors that
support ``input_provider`` can reuse cached raw inputs; unsupported processors
use direct reads with a warning.

.. list-table:: Execution settings
   :header-rows: 1
   :widths: 22 18 60

   * - Setting
     - Default
     - Meaning
   * - ``profile``
     - ``simple``
     - Select ``scalable`` to enable the experimental supervisor and planner.
   * - ``memory_limit``
     - Detected
     - Host RAM ceiling, clipped to available resources.
   * - ``worker_memory``
     - ``6GiB``
     - Reservation for an entire segment process tree, including lag workers.
   * - ``cache_limit``
     - ``1GiB``
     - Maximum cached raw sample payload; the actual allowance may be smaller.
   * - ``headroom``
     - ``512MiB``
     - Unallocated memory safety margin.
   * - ``message_limit``
     - ``64MiB``
     - Maximum serialized worker output message.
   * - ``worker_shutdown_timeout``
     - ``60`` seconds
     - Deadline to exit after reporting completion. A timeout fails the allocation
       and cleans up workers; this does not limit scientific processing time.
   * - ``preload``
     - ``auto``
     - Input loading policy: ``"off"``, ``auto``, or ``batch``.
   * - ``batch_size``
     - ``8``
     - Maximum jobs per planned group, not the concurrent worker count.
   * - ``cache_entries``
     - ``256``
     - Maximum live mapped cache entries.
   * - ``cores``
     - Allocation affinity
     - Optional total logical CPU cap.
   * - ``planner``, ``executor``
     - Unset
     - Optional dotted paths to zero-argument factories.

Memory sizes accept integer bytes or explicit SI/IEC units such as ``500MB``
or ``2GiB``. CPU and memory budgets, together with the runner's worker limit,
constrain actual concurrency.

Input reuse and execution
-------------------------

The planner groups jobs with shared frame/channel sources. The supervisor owns
a bounded cache of raw arrays and shares read-only memory-mapping descriptors
with workers. Each job receives an owned copy before modifying input data.

- ``preload: "off"`` uses direct reads. Quote ``"off"`` in YAML to avoid
  parsers interpreting it as a boolean.
- ``preload: auto`` loads reusable planned input intervals on demand.
- ``preload: batch`` preloads reusable inputs for a group if they fit, otherwise
  falls back to bounded demand loading.

Preloading is synchronous. This is local raw-input caching; it does not cache
conditioned data or provide a distributed cache.

Cluster setup creates a self-contained catalog fragment for each planned batch,
for both shared-filesystem and file-transfer runs. ``--batch-id`` selects that
fragment directly. Missing fragments require running ``batch-setup``; setup
rejects changes to an existing batch's job definitions or membership. Executed
fragments write diagnostic plans and metrics under ``RUN/execution/``.

The supervisor coordinates output writes and records progress after corresponding
trigger products are flushed. Resume skips committed work. Keep batch membership
and scientific configuration stable when resuming: progress is not reconciled
across regrouped catalog fragments.

.. _execution_resume:

Resume an unchanged run
-----------------------

For a local run using ``execution.profile: scalable``, keep the original YAML
and work directory and allow reuse of the existing output directory:

.. code-block:: bash

   pycwb progress --work-dir RUN_DIRECTORY --verbose
   pycwb run user_parameters.yaml --work-dir RUN_DIRECTORY --force-overwrite
   pycwb progress --work-dir RUN_DIRECTORY --verbose

The scalable executor skips committed jobs/lags and retains their products.
``--force-overwrite`` passes the existing-output-directory check; it does not
bypass YAML or execution-profile consistency checks and is not a general
recovery command for every processor. Use a new directory for changed inputs
or scientific settings. For batch runs, retain the original job membership
and select unfinished work within those prepared fragments.

After completion, follow :ref:`cluster_collect_results` for merging and
preserving the result set.

.. raw:: html

   <span id="limits-and-validation"></span>

Memory and performance
----------------------

Memory reservations and sampled monitoring are not a hard OS memory limit.
Scientific peaks can exceed estimates between samples. The runtime can stop
workers under memory pressure; use scheduler/cgroup enforcement for a hard
ceiling. This layer budgets host RAM, not GPU VRAM.

For GPU or processor comparisons, use :doc:`backends`. Match the scientific
workload and supported injection/output settings. Paired stage validation
repeats work, so disable it when measuring speed; keep the numerical comparison
and the performance measurement as separate results.

Worker startup and decoding overhead can make small jobs slower. Measure
end-to-end runtime on a representative workload when choosing worker counts.

Implementation reference
------------------------

See :doc:`pycwb.workflow.execution` for the Python API. The package is divided
into planning (``planner.py``), supervision (``executor.py``), resource accounting
(``resources.py``), raw-input handling (``cache.py`` and ``decoder.py``), output
transport (``writer.py``), batch membership (``scheduling.py``), configuration
(``settings.py``), and extension interfaces (``contracts.py``).

The repository also contains :download:`detailed execution design and
verification notes <../dev/scalable_execution.md>`.
