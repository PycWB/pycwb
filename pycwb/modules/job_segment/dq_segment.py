import logging
import numpy as np
import csv, math
from .types import DQFile, WaveSegment

logger = logging.getLogger(__name__)


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


def merge_seg_list(seg_list_1: tuple[np.ndarray | list, np.ndarray | list],
                   seg_list_2: tuple[np.ndarray | list, np.ndarray | list]):
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


def get_seg_list(dq_list, seg_len, seg_mls, seg_edge):
    job_list = get_job_list(dq_list, seg_len, seg_mls, seg_edge)
    return []


def get_job_list(dq_list, seg_len, seg_mls, seg_edge):
    """
    Build the job segment list.

    The job segments are builded starting from the input ilist
    each segment must have a minimum length of segMLS+2segEdge and a maximum length of segLen+2*segEdge
    in order to maximize the input live time each segment with lenght<2*(segLen+segEdge) is
    divided in 2 segments with length<segLen+2*segEdge
    segEdge     : xxx
    segMLS      : -------
    segLen      : ---------------
    input seg   : ----------------------------
    output segA : xxx---------xxx
    output segB :             xxx----------xxx

    :param dq_list:  number of detectors
    :param seg_len:  Segment length [sec]
    :param seg_mls:  Minimum Segment Length after DQ_CAT1 [sec]
    :param seg_edge:  wavelet boundary offset [sec]
    :return:  Return the job segment list
    """
    if seg_mls > seg_len:
        logger.error('seg_mls must be <= seg_len')
        raise ValueError('seg_mls must be <= seg_len')

    lostlivetime = 0
    job_list = []

    seg_index = 0

    for i in range(len(dq_list[0])):
        start = math.ceil(dq_list[0][i]) + seg_edge
        stop = math.floor(dq_list[1][i]) - seg_edge
        length = stop - start
        if length <= 0:
            continue

        seg_index += 1
        n = int(length / seg_len)
        if n == 0:
            if length < seg_mls:
                lostlivetime += length
                continue
            job_list.append(WaveSegment(seg_index, start, stop))
            continue
        if n == 1:
            if length > seg_len:
                remainder = length
                half = int(remainder / 2)
                if half >= seg_mls:
                    job_list.append(WaveSegment(seg_index, start, start + half))

                    seg_index += 1
                    job_list.append(WaveSegment(seg_index, start + half, stop))
                else:
                    job_list.append(WaveSegment(seg_index, start, start + seg_len))
            else:
                job_list.append(WaveSegment(i, start, stop))
            continue

        for j in range(n - 1):
            job_list.append(WaveSegment(seg_index, seg_len * j + start, seg_len * j + start + seg_len))

        remainder = stop - job_list[1][-1]
        half = int(remainder / 2)
        if half >= seg_mls:
            job_list.append(WaveSegment(i, job_list[1][-1], job_list[1][-1] + half))
            job_list.append(WaveSegment(i, job_list[1][-1], stop))
        else:
            job_list.append(WaveSegment(i, job_list[1][-1], job_list[1][-1] + seg_len))

    logger.info('lost livetime after building of the standard job list = %d sec' % lostlivetime)

    return job_list
