"""
pycwb.modules.catalog — Arrow/Parquet trigger and event catalog.

Primary I/O layer for PycWB. Stores triggers and events in self-contained
Parquet files with schema metadata, supports atomic writes via
``SoftFileLock``, deduplication on merge, and SQL/DuckDB query support.
Also provides JSON catalog format and trigger-to-simulation matching.
"""

from typing import Any

from importlib import import_module

from pycwb.types.base_catalog import BaseCatalog
from .catalog import (
    Catalog,
    create_catalog,
    add_triggers_to_catalog,
    add_events_to_catalog,
    read_catalog_metadata,
    read_catalog_triggers,
)
from .catalog_json import JSONCatalog
from .matching import match_triggers_to_simulations, match_simulations_parquet

__all__ = [
    "BaseCatalog",
    "Catalog",
    "create_catalog",
    "add_triggers_to_catalog",
    "add_events_to_catalog",
    "read_catalog_metadata",
    "read_catalog_triggers",
    "JSONCatalog",
    "convert_root_to_catalog",
    "read_root_triggers",
    "match_triggers_to_simulations",
    "match_simulations_parquet",
]

# Keep ROOT conversion helpers available as:
#     from pycwb.modules.catalog import convert_root_to_catalog
# without importing convert_root during package import. convert_root imports the
# Catalog implementation, so eager import here would recreate a package cycle.
_ROOT_EXPORTS = {"convert_root_to_catalog", "read_root_triggers"}


def __getattr__(name: str) -> Any:
    """Lazily resolve optional ROOT conversion exports on first access."""
    if name in _ROOT_EXPORTS:
        module = import_module("pycwb.modules.catalog.convert_root")
        value = getattr(module, name)
        # Cache the resolved helper so repeated access behaves like a normal import.
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
