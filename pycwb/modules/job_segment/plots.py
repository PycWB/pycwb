import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def plot_dq_list(dq_file_list, show_merge=False, dq_cat='CWB_CAT1', periods=None, figsize=(24, 6)):
    """Plot the data quality file list with IFO and dq_cat labels.

    :param dq_file_list: List of DQFile objects to plot.
    :type dq_file_list: list[DQFile]
    :param show_merge: If True, also plot the merged segment list from read_seg_list.
    :type show_merge: bool
    :param dq_cat: Data quality category used for merging when show_merge is True.
    :type dq_cat: str
    :param periods: Optional (starts, stops) constraint passed to read_seg_list.
    :type periods: tuple[list, list] or None
    :param figsize: Figure size, defaults to (24, 6).
    :type figsize: tuple[int, int]
    :return: The matplotlib pyplot object.
    """
    from .dq_segment import read_seg_list

    fig, ax = plt.subplots(figsize=figsize)

    y_ticks = []
    y_labels = []

    legend_handles = []
    for i, dq_file in enumerate(dq_file_list):
        starts, stops = dq_file.get_periods()
        y = i + 1
        label = f"{dq_file.ifo} {dq_file.dq_cat}"
        xranges = [(s, e - s) for s, e in zip(starts, stops)]
        ax.broken_barh(xranges, (y - 0.4, 0.8),
                       facecolor=(1, 0, 0, 0.2), edgecolor='red', linewidth=1.5)
        legend_handles.append(mpatches.Patch(facecolor=(1, 0, 0, 0.2), edgecolor='red',
                                             linewidth=1.5, label=label))
        y_ticks.append(y)
        y_labels.append(label)

    if show_merge:
        merged = read_seg_list(dq_file_list, dq_cat=dq_cat, periods=periods)
        y_merge = len(dq_file_list) + 1
        merge_label = f"merged ({dq_cat})"
        xranges = [(s, e - s) for s, e in zip(merged[0], merged[1])]
        ax.broken_barh(xranges, (y_merge - 0.4, 0.8),
                       facecolor=(0, 0, 1, 0.2), edgecolor='blue', linewidth=1.5)
        legend_handles.append(mpatches.Patch(facecolor=(0, 0, 1, 0.2), edgecolor='blue',
                                             linewidth=1.5, label=merge_label))
        y_ticks.append(y_merge)
        y_labels.append(merge_label)

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("GPS time")
    ax.set_title("Data Quality Segments")
    if legend_handles:
        ax.legend(handles=legend_handles)
    plt.tight_layout()

    return plt


def plot_seg_list(seg_list, merged_seg_list=None, figsize=(24, 6)):
    """Plot the segment list.

    :param seg_list: list of segments
    :type seg_list: tuple[np.ndarray | list, np.ndarray | list]
    :param merged_seg_list: list of merged segments which will be analyzed
    :type merged_seg_list: tuple[np.ndarray | list, np.ndarray | list], optional
    :param figsize: figure size, defaults to (24, 6)
    :type figsize: tuple[int, int], optional
    """

    plt.figure(figsize=figsize)

    min_start = min([np.nanmin(seg[0][seg[0] != 0]) for seg in seg_list])
    max_stop = max([np.nanmax(seg[1][seg[1] != np.inf]) for seg in seg_list])

    for i, seg in enumerate(seg_list):
        data = zip(seg[0], seg[1])
        for d in data:
            plt.plot(d, [i + 1, i + 1], color='red', linewidth=3)

    if merged_seg_list is not None:
        data = zip(merged_seg_list[0], merged_seg_list[1])
        for d in data:
            plt.plot(d, [len(seg_list) + 1, len(seg_list) + 1], color='blue', linewidth=10)

    plt.xlim(min_start, max_stop)

    return plt
