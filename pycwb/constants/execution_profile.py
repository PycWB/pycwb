"""Compatibility exports for the released processing-profile import path.

New code imports ``pycwb.config.processing``. Keep these aliases for existing
callers and pickles; all validation, defaults and models live in that module.
"""

from pycwb.config.processing import (
    ExecutionProfile as ExecutionProfile,
    PROFILE_SCHEMA as PROFILE_SCHEMA,
    DEFAULT_EXECUTION_PROFILE as DEFAULT_EXECUTION_PROFILE,
    resolve_execution_profile as resolve_execution_profile,
    execution_profile as execution_profile,
    wdm_options as wdm_options,
    recorded_execution_profile as recorded_execution_profile,
    check_recorded_execution_profile as check_recorded_execution_profile,
)
