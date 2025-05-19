import csv
import logging
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DQFile:
    """
    Class to store data quality file information

    Parameters
    ----------
    ifo : str
        ifo name
    file : str
        data quality file path
    dq_cat : str
        data quality category
    shift : float
        shift in seconds
    invert : bool
        flag for inversion
    c4 : bool
        flag for 4 column data
    """

    ifo: str
    file: str
    dq_cat: str
    shift: float
    invert: bool
    c4: bool

    # @property
    # def __dict__(self):
    #     return {
    #         "ifo": self.ifo,
    #         "file": self.file,
    #         "dq_cat": self.dq_cat,
    #         "shift": self.shift,
    #         "invert": self.invert,
    #         "c4": self.c4
    #     }

    def get_periods(self):
        """
        Get the periods from the data quality file.

        Returns
        -------
        start : np.ndarray
            start times
        stop : np.ndarray
            stop times
        """
        start = []
        stop = []
        # read the file in dq_file
        with open(self.file, 'r') as f:
            # remove the spaces at the end of each line
            lines = (line.rstrip().replace("\t", " ") for line in f)
            # read the file as csv
            reader = csv.reader(lines, delimiter=" ", skipinitialspace=True)
            for line in reader:
                try:
                    if line[0] == '#':
                        continue
                    if self.c4:
                        _, _start, _stop, _ = line
                    else:
                        _start, _stop = line
                except ValueError:
                    logger.error(f"Error to parse {self.file}: {line}")
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