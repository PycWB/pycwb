class TimeFrequencySeries:
    """
    Class for storing a time-frequency series.

    :param data: data
    :type data: pycbc.types.timeseries.TimeSeries
    :param wavelet: wavelet method
    :type wavelet: WDM
    :param whiten_mode: whiten mode
    :type whiten_mode: int
    :param bpp: black pixel probability
    :type bpp: float
    :param w_rate: wavelet zero layer rate
    :type w_rate: float
    :param f_low: low frequency cutoff
    :type f_low: float
    :param f_high: high frequency cutoff
    :type f_high: float
    """
    __slots__ = ['_wavelet', 'data', 'whiten_mode', 'bpp', 'w_rate', '_f_low', '_f_high', '_wseries', '_ptr']

    def __init__(self, data=None, wavelet=None, whiten_mode=None, bpp=None, w_rate=None, f_low=None, f_high=None, 
                 wseries=None):
        self._wavelet = None
        #: Time series data
        self.data = data
        #: Wavelet method, a new wavelet will be copied and data will be allocated automatically
        self.wavelet = wavelet
        #: Whiten mode
        self.whiten_mode = 0 if whiten_mode is None else whiten_mode
        #: black pixel probability
        self.bpp = 1. if bpp is None else bpp
        #: wavelet zero layer rate
        self.w_rate = (data.sample_rate if data else 0.) if w_rate is None else w_rate
        #: low frequency cutoff
        self._f_low = f_low
        #: high frequency cutoff
        self._f_high = f_high
        #: WSeries object, used for data cleaning
        self._wseries = wseries

    def __dict__(self):
        return {key: getattr(self, key) for key in self.__slots__}

    def copy(self):
        new = TimeFrequencySeries(
            data=self.data.copy(),
            wavelet=self.wavelet,
            whiten_mode=self.whiten_mode,
            bpp=self.bpp,
            w_rate=self.w_rate,
            f_low=self.f_low,
            f_high=self.f_high,
        )

        return new

    def __del__(self):
        if self._wseries:
            self._wseries.resize(0)
            del self._wseries
        if self._wavelet:
            self._wavelet.release()
            del self._wavelet

    __copy__ = copy

    def forward(self, k=-1):
        """
        Performs forward wavelet transform on data (Not working yet)
        """
        if self.wavelet.allocate():
            self.wavelet.nSTS = self.wavelet.nWWS
            # was failed every second time to copy and forward a new time series because of
            # the pointer pWWS is not malloced, so it can not be reallocated, fixed by converting
            # the list to malloced memory
            self.wavelet.t2w(k)
            # if self.wavelet.pWWS != self.data or self.wavelet.nWWS != len(self.data):
            #     # TODO: implement and convert (or not)
            #     print(' ====== Implementation missing')
            #     #     self.data = self.wavelet.pWWS
            #     #     self.Size = self.wavelet.nWWS
            #     #     self.Slice = slice(0, self.wavelet.nWWS, 1)
            self.w_rate = float(self.wavelet.get_slice_size(0) / (self.stop - self.start))
        else:
            raise ValueError('Wavelet transform failed')

    @property
    def wavelet(self):
        """
        ROOT.WDM object
        """
        return self._wavelet

    @wavelet.setter
    def wavelet(self, value):
        if not value:
            self._wavelet = None

        if self._wavelet:
            self._wavelet.release()
            del self._wavelet

        self._wavelet = value.clone()
        self._wavelet.allocate(self.data)

    @property
    def start(self):
        """
        start time
        """
        return self.data.start_time

    @property
    def stop(self):
        """
        stop time
        """
        return self.data.end_time

    @property
    def edge(self):
        """
        TODO: dummy edge
        """
        return 0.0

    @property
    def sample_rate(self):
        """
        sample rate
        """
        return self.data.sample_rate

    @property
    def f_high(self):
        """
        high frequency cutoff
        """
        return self.sample_rate / 2 if self._f_high is None else self._f_high

    @f_high.setter
    def f_high(self, value):
        self._f_high = value

    @property
    def f_low(self):
        """
        low frequency cutoff
        """
        return 0.0 if self._f_low is None else self._f_low

    @f_low.setter
    def f_low(self, value):
        self._f_low = value
