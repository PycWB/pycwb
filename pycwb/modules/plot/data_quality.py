"""Data-quality segment plotting helpers."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_data_quality(dq_files, figsize=(24, 6)):
    """Plot data-quality segment lists and return ``(fig, ax)``."""
    seg_list = []
    labels = []
    for dq_file in dq_files:
        seg_list.append(dq_file.get_periods())
        labels.append(f"{dq_file.ifo} ({dq_file.dq_cat})")

    fig, ax = plt.subplots(figsize=figsize)

    min_start = min([np.nanmin(seg[0][seg[0] != 0]) for seg in seg_list])
    max_stop = max([np.nanmax(seg[1][seg[1] != np.inf]) for seg in seg_list])

    for i, seg in enumerate(seg_list):
        data = zip(seg[0], seg[1])
        for segment in data:
            ax.plot(segment, [i + 1, i + 1], color="red", linewidth=3)

    ax.set_yticks(np.arange(len(seg_list)) + 1, labels)
    ax.set_xlim(min_start, max_stop)
    return fig, ax

