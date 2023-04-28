from pycbc.detector import Detector as PyCBCDetector


class Detector(PyCBCDetector):
    """
    Class for storing detector information.

    :param name: detector name
    :type name: str
    """
    def __init__(self, name):
        super().__init__(name)
