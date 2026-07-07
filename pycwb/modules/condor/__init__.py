"""
pycwb.modules.condor — HTCondor batch job submission.

Generates and submits HTCondor DAG (directed acyclic graph) batch jobs
for distributed PycWB analysis. Creates job scripts, merge scripts, and
simulation summary scripts with configurable resource requests.
"""

from .condor import HTCondor

__all__ = ["HTCondor"]
