from pathlib import Path
from pycwb.types.job import FrameFile
import logging

logger = logging.getLogger(__name__)


def get_frame_meta(frame_list_file, ifo, label=".gwf"):
    """
    Get the frame metadata (start time, duration) from a frame list file.
    The metadata will be extracted from the filename and stored in a FrameFile object.

    :param frame_list_file: file containing the frame list
    :type frame_list_file: str
    :param ifo: name of the interferometer
    :type ifo: str
    :param label: label of the frame file for filtering, default is ".gwf" which will select all frame files
    :type label: str
    :return: list of frame metadata
    :rtype: list[FrameFile]
    """
    # read the frame list file
    frame_list = []

    with open(frame_list_file, 'r') as f:
        for frame_path in f.readlines():
            if frame_path.startswith("#"):
                continue

            # remove header created with gw_data_find
            frame_path = frame_path.replace("framefile=", "")
            frame_path = frame_path.replace("file://localhost", "")
            frame_path = frame_path.replace("gsiftp://ldr.aei.uni-hannover.de:15000", "")
            frame_path = frame_path.strip()

            # if frame_path contains label
            if not frame_path.find(label):
                continue

            # test if file exists
            if not Path(frame_path).is_file():
                logger.error("Frame file not found: %s", frame_path)
                raise FileNotFoundError("Frame file not found: {}".format(frame_path))

            # get the file name without the extension with pathlib
            frame_name = Path(frame_path).stem
            # get the gps time and duration with int type
            gps_start, duration = [int(i) for i in frame_name.split("-")[-2:]]

            # if gps start smaller than the gps time 2015-01-01 or duration is smaller than 1,
            # throw an error of bad format
            if gps_start < 1104105616 or duration < 1:
                raise ValueError("Frame file name format is not correct: {}".format(frame_path))

            # append the frame file to the list
            frame_list.append(FrameFile(ifo, frame_path, gps_start, duration))

    return frame_list


def select_frame_list(frame_list, start, stop, seg_edge):
    """
    Select the frame files that are within the segment (start, stop) with a buffer of seg_edge seconds.

    :param frame_list: list of frame metadata
    :type frame_list: list[FrameFile]
    :param start: start time of the segment
    :type start: int or float
    :param stop: stop time of the segment
    :type stop: int or float
    :param seg_edge: buffer of the segment
    :type seg_edge: int or float
    :return: list of frame
    :rtype: list[FrameFile]
    """
    seg_start = start - seg_edge
    seg_stop = stop + seg_edge

    frame_list = [frame for frame in frame_list
              if frame.start_time < seg_stop and frame.start_time + frame.duration > seg_start]
    return frame_list


def get_frame_files_from_gwdatafind(ifo, site, frametype, start, stop, seg_edge, host=None):
    """
    Use gwdatafind to get the frame files for the job segments

    :param ifo: name of the interferometer
    :type ifo: str
    :param site: site name of the interferometer
    :type site: str
    :param frametype: frame type
    :type frametype: str
    :param start: start time of the segment
    :type start: int or float
    :param stop: stop time of the segment
    :type stop: int or float
    :param seg_edge: buffer of the segment
    :type seg_edge: int or float
    :param host: host name of the frame server
    :type host: str

    :return: list of frame files
    :rtype: list[FrameFile]
    """
    from gwdatafind import find_urls

    seg_start = start - seg_edge
    seg_stop = stop + seg_edge

    frame_list = []

    url = find_urls(site, frametype, seg_start, seg_stop, host=host)
    frame_paths = [fp.replace("file://localhost", "") for fp in url]

    for frame_path in frame_paths:
        # get the file name without the extension with pathlib
        frame_name = Path(frame_path).stem
        # get the gps time and duration with int type
        gps_start, duration = [int(i) for i in frame_name.split("-")[-2:]]
        # append the frame file to the list
        frame_list.append(FrameFile(ifo, frame_path, gps_start, duration))

    # TODO: check if the framefiles match with the seg_start and seg_stop

    return frame_list


