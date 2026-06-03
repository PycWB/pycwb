"""Random subsampling filter for parquet catalogs — workflow-compatible.

Provides a reusable step that reads a parquet catalog, optionally filters
out zero-lag background rows, randomly subsamples, and writes the result
to a new parquet file.  Designed for use in ``simple_module.yaml``-style
YAML workflows driven by :func:`pycwb.post_production.workflow.run_workflow`.

Workflow action
---------------
``postprocess.random_filter.random_filter_parquet``

Parameters (via YAML ``args``)
------------------------------
work_dir : str
    Base directory; relative paths are resolved against this.
input_file : str
    Path to the input parquet file (relative to *work_dir* unless absolute).
output_file : str
    Path for the filtered output parquet.
fraction : float, default 1.0
    Fraction of events to retain after filtering (0 < fraction ≤ 1).
zero_lag_filter : bool, default False
    If True, remove rows where ``lag_idx == 0`` (the nominal zero-lag
    coincident background).  Works with both pycWB Catalog parquet
    (which uses ``lag_idx``) and flat parquet (which may use ``lag`` +
    ``shift[0]`` / ``shift[1]`` columns).
seed : int, default 150914
    Random seed for reproducible subsampling.

Returns
-------
dict
    ``{"filtered_file": str, "n_before": int, "n_after": int}``
    Stored in the workflow global context under ``output_alias`` (if set).

Notes
-----
- Input is read with ``pandas.read_parquet()`` — works with both pycWB
  Catalog parquet files and plain pandas parquet files.
- Output is written as a plain pandas parquet (NOT a Catalog).
- Zero-lag detection tries ``lag_idx`` first (pycWB Catalog), then falls
  back to ``lag`` + ``shiftN`` columns (flat parquet).
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import pandas as pd

from pycwb.post_production.action_spec import action_spec

logger = logging.getLogger(__name__)

# Columns checked for zero-lag filtering (in priority order)
# pycWB Catalog uses 'lag_idx'; flat parquets may use 'lag' + 'shiftN'
_ZERO_LAG_COLS_CATALOG = ["lag_idx"]
_ZERO_LAG_COLS_FLAT = ["lag", "shift"]


@action_spec(
    outputs=['output_file'],
    inputs=['input_file'],
    description='Random subsample + optional zero-lag filter on parquet catalog',
)
def random_filter_parquet(
    work_dir: str,
    input_file: str,
    output_file: str,
    fraction: float = 1.0,
    zero_lag_filter: bool = False,
    seed: int = 150914,
    **kwargs,
) -> dict:
    """Read, filter, subsample, and write a parquet catalog.

    Parameters
    ----------
    work_dir : str
        Base working directory.  Relative *input_file* / *output_file* paths
        are resolved against this directory.
    input_file : str
        Path to the input parquet file.
    output_file : str
        Path for the filtered output parquet file.
    fraction : float
        Fraction of remaining events to keep (0 < fraction ≤ 1).
    zero_lag_filter : bool
        If True, exclude zero-lag coincident rows.
    seed : int
        Random seed for ``DataFrame.sample()``.

    Returns
    -------
    dict
        Keys: ``filtered_file`` (str), ``n_before`` (int), ``n_after`` (int).
    """
    # Resolve paths
    if not os.path.isabs(input_file):
        input_path = os.path.join(work_dir, input_file)
    else:
        input_path = input_file
    if not os.path.isabs(output_file):
        output_path = os.path.join(work_dir, output_file)
    else:
        output_path = output_file

    logger.info("Reading %s", input_path)
    df = _read_catalog_or_parquet(input_path)
    n_before = len(df)
    logger.info("  rows = %d", n_before)

    # ── optional zero-lag filter ──────────────────────────────────────────
    if zero_lag_filter:
        df, n_removed = _filter_zero_lag(df)
        if n_removed > 0:
            logger.info("  after zero-lag filter: %d  (removed %d)", len(df), n_removed)
        elif n_removed == 0:
            logger.info("  zero-lag filter: no rows matched (all non-zero-lag)")
        else:
            logger.warning(
                "zero_lag_filter=True but no recognized zero-lag columns found; skipped"
            )

    # ── random subsample ──────────────────────────────────────────────────
    if fraction < 1.0:
        n_sample = max(1, int(len(df) * fraction))
        df = df.sample(n=n_sample, random_state=seed).reset_index(drop=True)
        logger.info("  after random sample (frac=%.3f): %d", fraction, len(df))

    # ── write output ──────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    n_after = len(df)
    logger.info("Wrote %d rows → %s", n_after, output_path)

    return {
        "filtered_file": output_file,
        "n_before": n_before,
        "n_after": n_after,
    }


def _filter_zero_lag(df: pd.DataFrame):
    """Remove zero-lag rows.  Returns (filtered_df, n_removed).

    Returns (-1) for n_removed if no recognised zero-lag column exists.
    """
    # Try Catalog-style: lag_idx == 0
    if "lag_idx" in df.columns:
        n_before = len(df)
        df = df[df["lag_idx"] != 0].reset_index(drop=True)
        return df, n_before - len(df)

    # Try flat-parquet style: lag == 0 & shift[0] == 0 & shift[1] == 0 ...
    if "lag" in df.columns:
        has_shift = any(c.startswith("shift") for c in df.columns)
        if has_shift:
            mask = df["lag"] == 0
            for c in df.columns:
                if c.startswith("shift"):
                    mask = mask & (df[c] == 0)
            n_before = len(df)
            df = df[~mask].reset_index(drop=True)
            return df, n_before - len(df)
        else:
            # Only lag column, no shift — filter by lag == 0
            n_before = len(df)
            df = df[df["lag"] != 0].reset_index(drop=True)
            return df, n_before - len(df)

    return df, -1


def _read_catalog_or_parquet(path: str) -> pd.DataFrame:
    """Read a parquet file, preferring Catalog.open() for column-name fidelity.

    pycWB Catalog parquet files have a specific schema and metadata that
    :class:`~pycwb.modules.catalog.Catalog` uses to rename columns (e.g.
    ``coherent_energy`` → ``ecor``).  If the file is a valid Catalog, we
    read through Catalog to get the expected column names; otherwise we
    fall back to plain ``pd.read_parquet()``.
    """
    try:
        from pycwb.modules.catalog import Catalog

        cat = Catalog.open(path)
        table = cat.triggers(deduplicate=True)
        df = table.to_pandas()
        logger.info("  read via Catalog: %d rows, %d columns", len(df), len(df.columns))
        return df
    except Exception:
        logger.debug("  Catalog.open() failed, falling back to pd.read_parquet()")
        df = pd.read_parquet(path)
        logger.info("  read via pandas: %d rows, %d columns", len(df), len(df.columns))
        return df
