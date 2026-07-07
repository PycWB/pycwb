"""
pycwb.modules.slurm — Slurm batch job submission.

Generates and submits Slurm job arrays for distributed PycWB analysis.
Creates job scripts with configurable partitions, constraints, and
resource allocations for HPC clusters.
"""

from .slurm import Slurm

__all__ = ["Slurm"]
