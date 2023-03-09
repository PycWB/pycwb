import logging
import numpy as np
import csv, math
from .types import WaveSegment

logger = logging.getLogger(__name__)


def read_seg_list(dq_file_list, dq_cat):
    """Load the files contains segment list from the data quality files below the data quality category (load_dq_file).
    Then merge the segment list (merge_seg_list) and return the merged segment list.

    :param dq_file_list: The data quality files.
    :type dq_file_list: list[DQFile]
    :param dq_cat: The data quality category.
    :type dq_cat: str
    :return: A list of segments.
    :rtype: list[WaveSegment]
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


def load_dq_file(dq_file):
    """
    Load and process the data quality file.

    :param dq_file: The data quality file.
    :type dq_file: DQFile
    :return: a list of start and end times (start, end)
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    start = []
    stop = []
    # read the file in dq_file
    with open(dq_file.file, 'r') as f:
        lines = csv.reader(f, delimiter=" ", skipinitialspace=True)
        for line in lines:
            try:
                if line[0] == '#':
                    continue
                if dq_file.c4:
                    _, _start, _stop, _ = line
                else:
                    _start, _stop = line
            except ValueError:
                logger.error(f"Error to parse : {line}")
                raise Exception("Wrong format")


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


def merge_seg_list(seg_list_1, seg_list_2):
    """Merge the segment list.

    :param seg_list_1: The first segment list.
    :type seg_list_1: tuple[np.ndarray | list, np.ndarray | list]
    :param seg_list_2: The second segment list.
    :type seg_list_2: tuple[np.ndarray | list, np.ndarray | list]
    :return: A list of merged segments (start, end)
    :rtype: tuple[list, list]
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
    """
    Not implemented yet.

    :param dq_list:
    :param seg_len:
    :param seg_mls:
    :param seg_edge:
    :return:
    """
    job_list = get_job_list(dq_list, seg_len, seg_mls, seg_edge)
    return []


def get_job_list(dq_list, seg_len, seg_mls, seg_edge):
    """
    Build the job segment list.

    The job segments are builded starting from the input ilist, each segment must have a minimum length
    of segMLS+2segEdge and a maximum length of segLen+2*segEdge in order to maximize the input live time
    each segment with lenght<2*(segLen+segEdge) is divided in 2 segments with length<segLen+2*segEdge

    segEdge     : xxx \n
    segMLS      : ------- \n
    segLen      : --------------- \n
    input seg   : ---------------------------- \n
    output segA : xxx---------xxx \n
    output segB :             xxx----------xxx \n

    :param dq_list:  number of detectors
    :type dq_list:  list[DQFile]
    :param seg_len:  Segment length [sec]
    :type seg_len:  int
    :param seg_mls:  Minimum Segment Length after DQ_CAT1 [sec]
    :type seg_mls:  int
    :param seg_edge:  wavelet boundary offset [sec]
    :type seg_edge:  int
    :return:  Return the job segment list
    :rtype:  list[WaveSegment]
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

        n = int(length / seg_len)
        if n == 0:
            if length < seg_mls:
                lostlivetime += length
                continue
            seg_index += 1
            job_list.append(WaveSegment(seg_index, start, stop))
            continue
        if n == 1:
            if length > seg_len:
                remainder = length
                half = int(remainder / 2)
                if half >= seg_mls:
                    seg_index += 1
                    job_list.append(WaveSegment(seg_index, start, start + half))

                    seg_index += 1
                    job_list.append(WaveSegment(seg_index, start + half, stop))
                else:
                    seg_index += 1
                    job_list.append(WaveSegment(seg_index, start, start + seg_len))
            else:
                seg_index += 1
                job_list.append(WaveSegment(i, start, stop))
            continue

        for j in range(n - 1):
            seg_index += 1
            job_list.append(WaveSegment(seg_index, seg_len * j + start, seg_len * j + start + seg_len))

        remainder = stop - job_list[-1].end_time
        half = int(remainder / 2)
        if half >= seg_mls:
            seg_index += 1
            job_list.append(WaveSegment(seg_index, job_list[-1].end_time, job_list[-1].end_time + half))
            seg_index += 1
            job_list.append(WaveSegment(seg_index, job_list[-1].end_time, stop))
        else:
            seg_index += 1
            job_list.append(WaveSegment(seg_index, job_list[-1].end_time, job_list[-1].end_time + seg_len))

    logger.info('lost livetime after building of the standard job list = %d sec' % lostlivetime)

    return job_list
