class Network:
    """
    Class to hold information about a network of detectors.

    :param detectors: list of detectors
    :type detectors: list[Detector]
    :param ref_ifo: reference detector
    :type ref_ifo: str
    """
    __slots__ = ['detectors', 'ref_ifo']
    
    def __init__(self, detectors, ref_ifo):
        self.detectors = detectors
        self.ref_ifo = ref_ifo
