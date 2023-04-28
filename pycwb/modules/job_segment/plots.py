import matplotlib.pyplot as plt
import numpy as np


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
