"""Compatibility imports for the modular GPU background workflow.

New configurations should select
``pycwb.workflow.subflow.process_job_segment_gpu.process_job_segment``.
Scientific implementations live in ``coherence_gpu``, ``super_cluster_gpu``,
``likelihood_gpu`` and ``gpu_utils``; see this package's README for migration,
YAML options and validation instructions. Importing this namespace creates no GPU context.
"""
