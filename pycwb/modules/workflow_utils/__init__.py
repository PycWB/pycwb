"""
pycwb.modules.workflow_utils — Trigger persistence utilities.

Creates organized folder structures by job segment, trial, GPS time, and
hash ID. Saves event, cluster, and skymap data as JSON and registers
triggers in the Parquet catalog.
"""

from .trigger_utils import create_single_trigger_folder, save_trigger, add_trigger_to_catalog
