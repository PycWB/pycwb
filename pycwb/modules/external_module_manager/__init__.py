"""
pycwb.modules.external_module_manager — External module management.

Manages installation and versioning of external PycWB modules from Git
repositories. Loads module config from YAML, checks existence, and
pulls/clones external modules into the PycWB modules directory.
"""

from .manager import load_config, check_module_existence, pull_external_module

__all__ = [
    "load_config",
    "check_module_existence",
    "pull_external_module",
]
