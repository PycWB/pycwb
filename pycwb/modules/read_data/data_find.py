from gwdatafind import find_urls
from pycwb.modules.job_segment import WaveSegment
from pycwb.types.job import FrameFile
from pathlib import Path


def get_from_data_find(ifo_list, t_start=None, t_end=None, t_center=None, duration=1200, host='datafind.ldas.cit:80'):
    """
    Get data from data find
    """
    if t_center is not None:
        t_start = t_center - duration / 2
        t_end = t_center + duration / 2
    else:
        if t_start is None or t_end is None:
            raise ValueError("Either t_center or both t_start and t_end must be specified")

    frames = []

    for ifo in ifo_list:
        url = find_urls(ifo[0], f'{ifo}_HOFT_C00', t_start, t_end, host=host)
        frame_paths = [fp.replace("file://localhost", "") for fp in url]
        for frame_path in frame_paths:
            frame_name = Path(frame_path).stem
            gps_start, duration = [int(i) for i in frame_name.split("-")[-2:]]
            frames.append(FrameFile(ifo, frame_path, gps_start, duration))

    job_seg = WaveSegment(1, ifo_list, t_start, t_end, frames=frames)

    return job_seg