import os
import numpy as np
import pyarrow as pa

from pycwb.types import BaseCatalog
from pycwb.modules.catalog import Catalog


def read_catalog(catalog_file: str) -> dict:
    """Open a catalog and return a dict with keys
    ``"version"``, ``"config"``, ``"jobs"``, and ``"triggers"``
    (a de-duplicated :class:`pyarrow.Table`).
    """
    cat: BaseCatalog = Catalog.open(catalog_file)
    return {
        "version":  cat.version,
        "config":   cat.config,
        "jobs":     cat.jobs,
        "triggers": cat.triggers(deduplicate=True),
    }


def read_triggers(work_dir, run_dir, filters=None,
                  file=f'catalog/{Catalog.DEFAULT_FILENAME}', **kwargs) -> pa.Table:
    """Return the trigger table from a catalog, with optional filtering.

    *filters* is a list of Python boolean expression strings referencing column
    names, e.g. ``["rho > 5", "net_cc > 0.5"]``.  For struct sub-field queries
    (injection parameters) use :meth:`~pycwb.modules.catalog.Catalog.query`
    with DuckDB SQL directly.

    Example::

        cat = Catalog.open("catalog/catalog.parquet")
        table = cat.query(
            "SELECT * FROM triggers WHERE injection.mchirp > 10"
        )
    """
    path = os.path.join(work_dir, run_dir, file)
    print(f"Reading triggers from {path}")
    cat: BaseCatalog = Catalog.open(path)
    table = cat.triggers()
    print(f"Read {table.num_rows} triggers")
    if filters:
        table = cat.filter(*filters)
        print(f"{table.num_rows} triggers after filtering")
    return table


def read_live_time(work_dir, run_dir, filters=None,
                   file=f'catalog/{Catalog.DEFAULT_FILENAME}', **kwargs) -> list:
    """Return per-lag livetime dicts from the catalog."""
    path = os.path.join(work_dir, run_dir, file)
    print(f"Reading live time from {path}")
    cat: BaseCatalog = Catalog.open(path)
    livetimes = cat.live_time(filters=filters)
    total     = sum(lt["livetime"] for lt in livetimes)
    print(f"Total live time: {total:.1f} s "
          f"({total/86400:.2f} days, {total/86400/365:.2f} years)")
    return livetimes