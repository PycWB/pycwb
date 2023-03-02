import logging
import numpy as np
import csv
logger = logging.getLogger(__name__)


class WaveSegment:
    def __init__(self, index, start_time, end_time):
        self.index = index
        self.start_time = start_time
        self.end_time = end_time


class DQFile:
    def __init__(self, ifo, file, dq_cat, shift, invert, c4):
        self.ifo = ifo
        self.file = file
        self.dq_cat = dq_cat
        self.shift = shift
        self.invert = invert
        self.c4 = c4


def select_job_segment():
    """Select a job segment from the database.

    Returns:
        A list of job segments.
    """
    job_segment = []

    return job_segment


def read_seg_list(dq_file_list: list[DQFile], dq_cat):
    """Read the segment list from the database.

    Args:
        dq_file_list (str): The data quality files.
        dq_cat (str): The data quality category.

    Returns:
        A list of segments.
    """
    seg_list = []
    for dq_file in dq_file_list:
        if dq_file.dq_cat <= dq_cat:
            logger.info(f"Loading data quality file {dq_file.file}")
            seg_list.append(load_dq_file(dq_file))

    if len(seg_list) == 0:
        logger.error("No CWB_CAT=%s files in the list", dq_cat)
        raise Exception("No CWB_CAT=%s files in the list", dq_cat)

    merged_seg_list = seg_list[0]
    for seg_list in seg_list[1:]:
        merged_seg_list = merge_seg_list(merged_seg_list, seg_list)

    return merged_seg_list


def load_dq_file(dq_file: DQFile):
    start = []
    stop = []
    # read the file in dq_file
    with open(dq_file.file, 'r') as f:
        lines = csv.reader(f, delimiter=" ", skipinitialspace=True)
        for line in lines:
            if line[0] == '#':
                continue
            if dq_file.c4:
                _, _start, _stop, _ = line
            else:
                _start, _stop = line

            _start = float(_start)
            _stop = float(_stop)

            if _stop <= _start:
                raise Exception("Error Ranges : %s %s", _start, _stop)

            start.append(_start + dq_file.shift)
            stop.append(_stop + dq_file.shift)

    start = np.array(start)
    stop = np.array(stop)
    order = start.argsort()
    start = start[order]
    stop = stop[order]

    # check if all start > 0
    if np.any(start < 0):
        error_indexes = np.where(start < 0)
        for i in error_indexes:
            logger.error(f"Error Ranges : {float(start[i])} {float(stop[i])}")
        raise Exception("Error Ranges")

    # check if all start < stop
    if np.any(start > stop):
        error_indexes = np.where(start > stop)
        for i in error_indexes:
            logger.error(f"Error Ranges : {float(start[i])} {float(stop[i])}")
        raise Exception("Error Ranges")

    # check if all stop < next start
    if np.any(stop[:-1] > start[1:]):
        error_indexes = np.where(stop[:-1] > start[1:])
        for i in error_indexes:
            logger.error(f"Error Ranges : {float(start[i])} {float(stop[i])}")
        raise Exception("Error Ranges")

    if dq_file.invert:
        start, stop = np.append(0, stop), np.append(start, np.inf)

    return start, stop


def merge_seg_list(seg_list_1: tuple[np.ndarray | list, np.ndarray | list], seg_list_2:  tuple[np.ndarray | list, np.ndarray | list]):
    """Merge the segment list.

    Args:
        seg_list_1 (list): The first segment list.
        seg_list_2 (list): The second segment list.

    Returns:
        A list of merged segments.
    """
    merged_start = []
    merged_stop = []
    i = 0
    j = 0
    n1 = len(seg_list_1[0])
    n2 = len(seg_list_2[0])

    while i < n1 and j < n2:
        # if stop2 <= start1, precede list2
        if seg_list_2[1][j] <= seg_list_1[0][i]:
            j += 1
        # if stop1 <= start2, precede list1
        elif seg_list_1[1][i] <= seg_list_2[0][j]:
            i += 1

        else:
            if seg_list_2[0][j] < seg_list_1[0][i]:
                start = seg_list_1[0][i]
            else:
                start = seg_list_2[0][j]
            if seg_list_2[1][j] > seg_list_1[1][i]:
                stop = seg_list_1[1][i]
            else:
                stop = seg_list_2[1][j]
            if seg_list_2[1][j] >= seg_list_1[1][i]:
                i += 1
            else:
                j += 1

            merged_start.append(start)
            merged_stop.append(stop)

    return merged_start, merged_stop


def plot_seg_list(seg_list: tuple[np.ndarray | list, np.ndarray | list], merged_seg_list=None, figsize=(24, 6)):
    """Plot the segment list.

    Args:
        seg_list (list): The segment list.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=figsize)

    min_start = min([np.nanmin(seg[0][seg[0] != 0]) for seg in seg_list])
    max_stop = max([np.nanmax(seg[1][seg[1] != np.inf]) for seg in seg_list])

    for i, seg in enumerate(seg_list):
        data = zip(seg[0], seg[1])
        for d in data:
            plt.plot(d, [i+1, i+1], color='red', linewidth=3)

    if merged_seg_list is not None:
        data = zip(merged_seg_list[0], merged_seg_list[1])
        for d in data:
            plt.plot(d, [len(seg_list)+1,len(seg_list)+1], color='blue', linewidth=10)

    plt.xlim(min_start, max_stop)

    return plt