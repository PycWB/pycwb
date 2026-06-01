"""
Catalog and cluster-artifact loading for Phase 1/2 studies.

Testing deferred — functions here require representative Parquet catalogs
that are too large for the unit-test repository.

Public API
----------
load_trigger_table(path) -> pandas.DataFrame
build_feature_matrix(rows, feature_names) -> np.ndarray
"""

from __future__ import annotations

import numpy as np


def load_trigger_table(path: str):
    """Load a Parquet catalog into a :class:`pandas.DataFrame`.

    Parameters
    ----------
    path : str
        Path to a Parquet file or directory of Parquet files (glob patterns
        accepted by :func:`pandas.read_parquet`).

    Returns
    -------
    pandas.DataFrame
        One row per trigger with all catalog columns.

    Raises
    ------
    ImportError
        If ``pandas`` or ``pyarrow`` are unavailable.
    FileNotFoundError
        If *path* does not exist.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required for load_trigger_table")

    return pd.read_parquet(path)


def build_feature_matrix(
    rows: list[dict[str, float]],
    feature_names: list[str] | None = None,
) -> np.ndarray:
    """Convert a list of feature dicts to a 2-D NumPy matrix.

    Parameters
    ----------
    rows : list[dict[str, float]]
        Each dict should contain the same set of scalar keys.
    feature_names : list[str] | None
        Ordered list of feature keys to include.  If *None*, uses the sorted
        key set of the first row.

    Returns
    -------
    np.ndarray, shape (n_samples, n_features)
        Rows correspond to elements in *rows*; columns to *feature_names*.
        Missing values are filled with ``NaN``.
    """
    if not rows:
        return np.zeros((0, 0), dtype=np.float64)

    if feature_names is None:
        feature_names = sorted(rows[0].keys())

    matrix = np.full((len(rows), len(feature_names)), fill_value=float("nan"), dtype=np.float64)
    for ri, row in enumerate(rows):
        for ci, name in enumerate(feature_names):
            v = row.get(name, float("nan"))
            try:
                matrix[ri, ci] = float(v)
            except (TypeError, ValueError):
                pass  # leave NaN

    return matrix
