class WDM:
    """
    Wrapper for ROOT.WDM object

    :param wavelet: ROOT.WDM object
    :type wavelet: ROOT.WDM
    """
    def __init__(self, wavelet):
        #: ROOT.WDM object
        self.wavelet = wavelet
