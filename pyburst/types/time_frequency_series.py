class TimeFrequencySeries:
    """
    Class for storing a time-frequency series.

    :param data: data
    :type data: pycbc.types.timeseries.TimeSeries
    :param wavelet: wavelet method
    :type wavelet: WDM
    :param whiten_mode: whiten mode
    :type whiten_mode: str
    :param bpp: black pixel probability
    :type bpp: float
    :param w_rate: wavelet zero layer rate
    :type w_rate: float
    :param f_low: low frequency cutoff
    :type f_low: float
    :param f_high: high frequency cutoff
    :type f_high: float
    """
    __slots__ = ['data', 'wavelet', 'whiten_mode', 'bpp', 'w_rate', 'f_low', 'f_high']

    def __init__(self, data, wavelet, whiten_mode=None, bpp=None, w_rate=None, f_low=None, f_high=None):
        #: Time series data
        self.data = data
        #: Wavelet method
        self.wavelet = wavelet
        #: Whiten mode
        self.whiten_mode = whiten_mode
        #: black pixel probability
        self.bpp = bpp
        #: wavelet zero layer rate
        self.w_rate = w_rate
        #: low frequency cutoff
        self.f_low = f_low
        #: high frequency cutoff
        self.f_high = f_high

    # def forward(self, k=-1):
    #     """
    #     Performs forward wavelet transform on data
    #     """
    #     if self.wavelet.allocate():
    #         self.wavelet.nSTS = self.wavelet.nWWS
    #         self.wavelet.t2w(k)
    #         if self.wavelet.pWWS != self.data or self.wavelet.nWWS != len(self.data):
    #             self.data = self.wavelet.pWWS
    #             self.Size = self.wavelet.nWWS
    #             self.Slice = slice(0, self.wavelet.nWWS, 1)
    #         self.wrate = self.Slice.size() / (self.stop() - self.start())


class SparseTable(TimeFrequencySeries):
    def __init__(self, data, wavelet, whiten_mode=None, bpp=None, w_rate=None, f_low=None, f_high=None):
        super().__init__(data, wavelet, whiten_mode, bpp, w_rate, f_low, f_high)
        #: List of significant pixels
        self.pixels = []