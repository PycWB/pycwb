"""Helpers for zero-lag / nonzero-lag filtering in post-production."""

from __future__ import annotations

from collections.abc import Iterable
import os

import pandas as pd


def unshifted_job_ids_from_catalog(catalog_file: str) -> set[int]:
    """Return job IDs whose catalog metadata has no segment/superlag shift."""
    from pycwb.modules.catalog.catalog import Catalog

    catalog = Catalog.open(catalog_file)
    job_ids: set[int] = set()
    for job in catalog.jobs:
        shift = job.get("shift")
        if shift is None or _sequence_is_zero(shift):
            job_ids.add(int(job["index"]))
    return job_ids


def try_unshifted_job_ids_from_catalog(catalog_file: str) -> set[int] | None:
    """Return unshifted job IDs, or ``None`` if the catalog has no job metadata."""
    from pycwb.modules.catalog.catalog import Catalog

    catalog = Catalog.open(catalog_file)
    if not catalog.jobs:
        return None
    return unshifted_job_ids_from_catalog(catalog_file)


def unshifted_job_ids_from_progress(progress_file: str) -> set[int]:
    """Infer the sibling catalog path for a progress parquet and read unshifted jobs."""
    dirname = os.path.dirname(progress_file)
    basename = os.path.basename(progress_file).replace("progress", "catalog", 1)
    return unshifted_job_ids_from_catalog(os.path.join(dirname, basename))


def zero_lag_mask(
    df: pd.DataFrame,
    unshifted_job_ids: set[int] | None = None,
) -> pd.Series:
    """Return rows with no regular lag and no segment/superlag shift.

    A physical zero-lag row must be unshifted in both senses:

    * regular time-slide lag is zero (``lag_idx == 0`` or ``lag == 0``), and
    * segment/superlag shift is zero for every detector.

    For progress tables, pass ``unshifted_job_ids`` from the parent
    :class:`~pycwb.modules.catalog.catalog.Catalog` metadata.  Progress rows do
    not carry superlag shifts themselves.
    """
    mask = pd.Series(True, index=df.index)

    if unshifted_job_ids is not None and "job_id" in df.columns:
        mask &= pd.to_numeric(df["job_id"], errors="coerce").isin(unshifted_job_ids)

    if "lag_idx" in df.columns:
        mask &= pd.to_numeric(df["lag_idx"], errors="coerce").eq(0)
    elif "lag" in df.columns:
        mask &= pd.to_numeric(df["lag"], errors="coerce").eq(0)

    time_lag_cols = _matching_columns(df, ("time_lag_", "time_lag"))
    if time_lag_cols:
        mask &= _all_zero(df, time_lag_cols)

    if unshifted_job_ids is None:
        segment_lag_cols = _matching_columns(
            df,
            ("segment_lag_", "segment_lag", "segment_shift_", "segment_shift", "shift_", "shift"),
        )
        if segment_lag_cols:
            mask &= _all_zero(df, segment_lag_cols)

    return mask.fillna(False)


def nonzero_lag_mask(
    df: pd.DataFrame,
    unshifted_job_ids: set[int] | None = None,
) -> pd.Series:
    """Return rows that are not physical zero-lag rows."""
    return ~zero_lag_mask(df, unshifted_job_ids=unshifted_job_ids)


def _matching_columns(df: pd.DataFrame, names: tuple[str, ...]) -> list[str]:
    columns: list[str] = []
    for col in df.columns:
        if col in names or any(col.startswith(f"{name}_") for name in names):
            columns.append(col)
    return columns


def _all_zero(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for col in columns:
        values = df[col]
        if _looks_like_sequence(values):
            mask &= values.apply(_sequence_is_zero)
        else:
            mask &= pd.to_numeric(values, errors="coerce").fillna(0.0).abs().le(1e-12)
    return mask


def _looks_like_sequence(values: pd.Series) -> bool:
    sample = values.dropna().head(1)
    if sample.empty:
        return False
    value = sample.iloc[0]
    return not isinstance(value, (str, bytes)) and isinstance(value, Iterable)


def _sequence_is_zero(value) -> bool:
    if value is None:
        return True
    try:
        return all(abs(float(item)) <= 1e-12 for item in value)
    except TypeError:
        return abs(float(value)) <= 1e-12
