"""
Report generation for Phase 1/2 clustering validation (deferred).

Functions here produce summary tables and plots once large-data catalogs are
available.  They are intentionally stubs — importing this module does not pull
in any heavy plotting dependencies at collection time.

Public API
----------
summary_table(feature_matrix, labels, feature_names) -> str
plot_cluster_sizes(labels, output_path) -> None  [requires matplotlib]
"""

from __future__ import annotations


def summary_table(
    feature_matrix,
    labels,
    feature_names: list[str] | None = None,
) -> str:
    """Return a plain-text summary table of per-cluster feature statistics.

    Parameters
    ----------
    feature_matrix : array-like, shape (n_samples, n_features)
        Feature matrix built by :func:`~validation.datasets.build_feature_matrix`.
    labels : array-like of int, shape (n_samples,)
        Cluster labels from a clustering run.
    feature_names : list[str] | None
        Column names for *feature_matrix*.  Defaults to ``"f0"``, ``"f1"``, …

    Returns
    -------
    str
        Formatted table suitable for logging.
    """
    import numpy as np

    X = np.asarray(feature_matrix, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)

    if feature_names is None:
        feature_names = [f"f{i}" for i in range(X.shape[1])]

    unique_labels = sorted(set(y.tolist()))
    header = f"{'cluster':>8}  {'n':>6}  " + "  ".join(f"{fn:>10}" for fn in feature_names)
    lines = [header, "-" * len(header)]

    for cl in unique_labels:
        mask = y == cl
        n = int(mask.sum())
        means = X[mask].mean(axis=0) if n > 0 else [float("nan")] * len(feature_names)
        row = f"{cl:>8}  {n:>6}  " + "  ".join(f"{m:>10.3g}" for m in means)
        lines.append(row)

    return "\n".join(lines)


def plot_cluster_sizes(labels, output_path: str | None = None) -> None:
    """Plot a histogram of cluster sizes.

    Parameters
    ----------
    labels : array-like of int
        Cluster labels.  ``-1`` is treated as noise and excluded.
    output_path : str | None
        Save the figure to this path.  Shows interactively if *None*.

    Raises
    ------
    ImportError
        If ``matplotlib`` is unavailable.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plot_cluster_sizes")

    import numpy as np

    y = np.asarray(labels, dtype=np.int64)
    valid = y[y >= 0]
    _, counts = np.unique(valid, return_counts=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(counts, bins=30, edgecolor="black")
    ax.set_xlabel("Cluster size (pixels)")
    ax.set_ylabel("Count")
    ax.set_title("Cluster size distribution")
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()
