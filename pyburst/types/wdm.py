class WDM:
    """
    Wrapper for ROOT.WDM object

    :param wavelet: ROOT.WDM object
    :type wavelet: ROOT.WDM
    """
    def __init__(self, wavelet):
        #: ROOT.WDM object
        self.wavelet = wavelet

    def set_td_filter(self, coeff_factor, upsample_factor):
        """
        initialization of the time delay filters

        :param coeff_factor: define the number of the filter coefficients
        :type coeff_factor: int
        :param upsample_factor: upsample factor, defines the fundamental time delay step dt = tau/L , where tau is the sampling interval of the original time series
        :type upsample_factor: int
        """
        self.wavelet.setTDFilter(coeff_factor, upsample_factor)
