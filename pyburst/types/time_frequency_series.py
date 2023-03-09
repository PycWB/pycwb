class TimeFrequencySeries:
    """
    Class for storing a time-frequency series.

    :param data: data
    :type data: pycbc.types.timeseries.TimeSeries
    :param wavelet: wavelet method
    :type wavelet: WDM
    """
    __slots__ = ['data', 'wavelet', 'whiten_mode']

    def __init__(self, data, wavelet, whiten_mode=None):
        #: Time series data
        self.data = data
        #: Wavelet method
        self.wavelet = wavelet
        #: Whiten mode
        self.whiten_mode = whiten_mode