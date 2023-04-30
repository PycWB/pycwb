import csv
import logging
import numpy as np

logger = logging.getLogger(__name__)


class DQFile:
    __slots__ = ['ifo', 'file', 'dq_cat', 'shift', 'invert', 'c4']
    """
    Class to store data quality file information

    :param ifo: ifo name
    :type ifo: str
    :param file: data quality file path
    :type file: str
    :param dq_cat: data quality category
    :type dq_cat: str
    :param shift: shift in seconds
    :type shift: float
    :param invert: flag for inversion
    :type invert: bool
    :param c4: flag for 4 column data
    :type c4: bool
    """

    def __init__(self, ifo, file, dq_cat, shift, invert: bool, c4):
        self.ifo = ifo
        self.file = file
        self.dq_cat = dq_cat
        self.shift = shift
        self.invert = invert
        self.c4 = c4

    def __repr__(self):
        return f"DQFile(ifo={self.ifo}, file={self.file}, dq_cat={self.dq_cat}, " \
               f"shift={self.shift}, invert={self.invert}, c4={self.c4})"

    @property
    def __dict__(self):
        return {
            "ifo": self.ifo,
            "file": self.file,
            "dq_cat": self.dq_cat,
            "shift": self.shift,
            "invert": self.invert,
            "c4": self.c4
        }

    def get_periods(self):
        """
        Load and process the data quality file.

        :param self: The data quality file.
        :type self: DQFile
        :return: a list of start and end times (start, end)
        :rtype: tuple[np.ndarray, np.ndarray]
        """
        start = []
        stop = []
        # read the file in dq_file
        with open(self.file, 'r') as f:
            lines = csv.reader(f, delimiter=" ", skipinitialspace=True)
            for line in lines:
                try:
                    if line[0] == '#':
                        continue
                    if self.c4:
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

                start.append(_start + self.shift)
                stop.append(_stop + self.shift)

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

        if self.invert:
            start, stop = np.append(0, stop), np.append(start, np.inf)

        return start, stop