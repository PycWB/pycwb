class WaveSegment:
    """
    Class to store the metadata of a wave segment for analysis, which contains the index of the segment,
    the start and end time of the segment, and the list of frame files that are within the segment.

    Parameters
    ----------
    index: int
        index of the segment
    ifos: list of str
        list of interferometers
    start_time: float
        start time of the segment
    end_time: float
        end time of the segment
    frames: list, optional
        list of frame files that are within the segment
    noise: list, optional
        list of noise configuration that are within the segment
    injections: list, optional
        list of injections that are within the segment
    """
    __slots__ = ('index', 'ifos', 'start_time', 'end_time', 'frames',  'noise', 'injections')

    def __init__(self, index, ifos, start_time, end_time, frames=None, noise=None, injections=None):
        #: index of the segment
        self.index = index
        #: list of interferometers
        self.ifos = ifos
        #: start time of the segment
        self.start_time = float(start_time)
        #: end time of the segment
        self.end_time = float(end_time)
        #: list of frame files that are within the segment
        self.frames = frames or []
        #: list of noise configuration that are within the segment
        self.noise = noise or []
        #: list of injections that are within the segment
        self.injections = injections or []

    def __repr__(self):
        return f"WaveSegment(index={self.index}, start_time={self.start_time}, " \
               f"end_time={self.end_time}, frames={len(self.frames)}, injections={self.injections})"

    def to_dict(self):
        """
        Convert the WaveSegment object to a dictionary.

        :return: dictionary of the WaveSegment object
        :rtype: dict
        """
        return {
            'index': self.index,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'frames': [{
                'ifo': frame.ifo,
                'path': frame.path,
                'start_time': frame.start_time,
                'duration': frame.duration
            } for frame in self.frames]
        }

    @property
    def duration(self):
        """
        Duration of the segment.

        Returns
        -------
        duration: float
            duration of the segment
        """
        return self.end_time - self.start_time


class SLag:
    """
    Class to store the metadata of a SLag, which contains the job id, the slag id, and the segment id.

    :param job_id: job id
    :type job_id: int
    :param slag_id: slag id vector, [0]=jobId - [1]=1/0 1=header slag - [2,..,nIFO+1] ifo slag
    :type slag_id: list[int]
    :param seg_id: segment id vector, [0,..,nIFO-1] ifo segment number
    :type seg_id: list[int]
    """
    __slots__ = ('job_id', 'slag_id', 'seg_id')

    def __init__(self, job_id, slag_id, seg_id):
        #: job id
        self.job_id = job_id
        #: slag id vector : [0]=jobId - [1]=1/0 1=header slag - [2,..,nIFO+1] ifo slag
        self.slag_id = slag_id
        #: seg id vector : [0,..,nIFO-1] ifo segment number
        self.seg_id = seg_id

    def __repr__(self):
        return f"SLag(job_id={self.job_id}, slag_id={self.slag_id}, seg_id={self.seg_id})"


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
    __slots__ = ('ifo', 'path', 'start_time', 'duration')

    def __init__(self, ifo, path, start_time, duration):
        #: name of the interferometer
        self.ifo = ifo
        #: path of the frame file
        self.path = path
        #: start time of the frame file
        self.start_time = start_time
        #: duration of the frame file
        self.duration = duration

    def __repr__(self):
        return f"FrameFile(ifo={self.ifo}, path={self.path}, " \
               f"start_time={self.start_time}, duration={self.duration})"

    @property
    def end_time(self):
        """
        Get the end time of the frame file.

        Returns
        -------
        end_time: float
            end time of the frame file
        """
        return self.start_time + self.duration
