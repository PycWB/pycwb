"""
Module for Super Lag, not yet implemented
"""
import numpy as np
import logging
from pycwb.types.job import SLag

logger = logging.getLogger(__name__)


def get_slag_job_list(seg: tuple[np.ndarray | list, np.ndarray | list], seg_len):
    """
    Get the job list for SLag

    :param seg:
    :param seg_len:
    :return:
    """
    if len(seg[0]) == 0:
        logger.error("Error ilist size=0")
        raise Exception("Error ilist size=0")
    if seg_len <= 0:
        logger.error("Error seglen<=0")
        raise Exception("Error seglen<=0")

    start = seg[0][0]
    stop = seg[1][-1]

    start = seg_len * np.round(start / seg_len)
    stop = seg_len * np.round(stop / seg_len)

    njob = (stop - start) / seg_len
    job_list = []
    for n in range(int(njob)):
        job_list.append((start + n * seg_len, start + n * seg_len + seg_len))

    return job_list


def get_slag_list(slag_list, ifos, seg_len, seg_min, seg_edge, nDQF, iDQF, dqcat):
    """
    Get the list of slags \n
    if dqcat=CWB_CAT1 -> return the list of slags with dq len > segMin+2*segEdge \n
    if dqcat=CWB_CAT2 -> return the list of slags with dq len > segMin

    :param slag_list:  vector list of slag structures
    :param ifos:  vector list of ifo names
    :param seg_len:  Segment length [sec]
    :param seg_min:  Minimum Segment Length after dqcat [sec]
    :param seg_edge:  wavelet boundary offset [sec]
    :param nDQF:  size of iDQF array
    :param iDQF:  DQ structure array
    :param dqcat:  dq cat
    :return:
    """
    if dqcat not in ['CWB_CAT1', 'CWB_CAT2']:
        logger.error("dqcat must be CWB_CAT1 or CWB_CAT2 !!!")
        raise Exception("dqcat must be CWB_CAT1 or CWB_CAT2 !!!")

    # TODO: finish this function


def add_slag_shifts(slag, ifos, seg_len, dq_files):
    """
    Add the shift to the dq files

    :param slag:
    :param ifos:
    :param seg_len:
    :param dq_files:
    :return:
    """
    for dq_file in dq_files:
        ifo_id = ifos.indexof(dq_file.ifo)
        if ifo_id:
            dq_file.shift += -seg_len * (slag.seg_id[ifo_id] - slag.seg_id[0])
