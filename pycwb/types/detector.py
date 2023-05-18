from pycbc.detector import Detector as PyCBCDetector


class Detector(PyCBCDetector):
    """
    Class for storing detector information.

    Parameters
    ----------
    name : str
        detector name
    """
    def __init__(self, name):
        super().__init__(name)
