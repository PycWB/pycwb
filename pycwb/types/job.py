from typing import List, Optional, Dict
from dataclasses import dataclass, asdict


@dataclass
class FrameFile:
    """
    Class to store the metadata of a frame file, which contains the ifo, the path, the start time, and the duration.

    Parameters
    ----------
    ifo: str
        name of the interferometer
    path: str
        path of the frame file
    start_time: float
        start time of the frame file
    duration: float
        duration of the frame file
    """
    ifo: str
    path: str
    start_time: float
    duration: float

    @property
    def end_time(self) -> float:
        """
        Get the end time of the frame file.

        Returns
        -------
        end_time: float
            end time of the frame file
        """
        return self.start_time + self.duration


@dataclass
class WaveSegment:
    """
    Class to store the metadata of a wave segment for analysis, which contains the index of the segment,
    the start and end time of the segment, and the list of frame files that are within the segment.

    Parameters
    ----------
    index: int
        index of the segment
    trail_idx: int
        trail index of the segment for injections, leave it 0 for no injections
    ifos: list of str
        list of interferometers
    start_time: float
        start time of the segment
    end_time: float
        end time of the segment
    sample_rate: float
        sample rate of the segment
    shift: list, optional
        list of shifts for each interferometer, used for superlags
    seg_edge: float, optional
        the edge of the segment
    channels: list, optional
        list of data  channels for each interferometer
    frames: list, optional
        list of frame files that are within the segment
    noise: dict, optional
        The noise configurations that are within the segment
    injections: list, optional
        list of injections that are within the segment
    """
    index: int
    ifos: List[str]
    start_time: float
    end_time: float
    sample_rate: float
    seg_edge: float
    shift: Optional[List[float]] = None
    channels: Optional[List[str]] = None
    frames: Optional[List[FrameFile]] = None
    noise: Optional[Dict] = None
    injections: Optional[List[Dict]] = None
    trail_idx: int = 0

    @property
    def duration(self) -> float:
        """
        Duration of the segment.

        Returns
        -------
        duration: float
            duration of the segment
        """
        return self.end_time - self.start_time

    @property
    def physical_start_times(self) -> Dict[str, float]:
        """
        Get the physical start times of the segment.

        Returns
        -------
        physical_start_times: dict
            physical start times of the segment
        """
        if self.shift is None:
            return {ifo: self.start_time for ifo in self.ifos}
        return {ifo: self.start_time - self.shift[i] for i, ifo in enumerate(self.ifos)}

    @property
    def physical_end_times(self) -> Dict[str, float]:
        """
        Get the physical end times of the segment.

        Returns
        -------
        physical_end_times: dict
            physical end times of the segment
        """
        if self.shift is None:
            return {ifo: self.end_time for ifo in self.ifos}
        return {ifo: self.end_time - self.shift[i] for i, ifo in enumerate(self.ifos)}

    to_dict = asdict


@dataclass
class SLag:
    """
    Class to store the metadata of a SLag, which contains the job id, the slag id, and the segment id.

    Parameters
    ----------
    job_id: int
        job id
    slag_id: list[int]
        slag id vector, [0]=jobId - [1]=1/0 1=header slag - [2,..,nIFO+1] ifo slag
    seg_id: list[int]
        seg id vector, [0,..,nIFO-1] ifo segment number
    """
    job_id: int
    slag_id: List[int]
    seg_id: List[int]