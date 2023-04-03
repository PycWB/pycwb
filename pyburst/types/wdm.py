import copy

import ROOT
import numpy as np


class WDM:
    """
    Wrapper for ROOT.WDM object

    after forward transformation the structure of the WDM sliced array is the following:

        f    phase 0       phase 90   \n
        M  *  * ...  *   *   * ...  * \n
        ... ... ... ... ...  ... ...\n
        1  *  * .... *   *   * ...  * \n
        0  *  * ...  *   *   * ...  * \n
        t  1  2  ... n  n+1 n+2... 2n \n

    where t/f is the time/frequency index, the global TF index in the linear array is i = t*(M+1)+f

    :param m: number of bands
    :type m: int
    :param k: k=n*m, where n is integer. K defines the width of the 'edge' of the basis function
     in Fourier domain (see the paper). larger n, longer the WDM filter, lower the spectral leakage between the bands.
    :type k: int
    :param beta_order: defines the sharpness of the 'edge' of the basis function in Fourier domain (see paper)
    :type beta_order: int
    :param precision: defines filter length by truncation error quantified by P = -log10(1 - norm_of_filter) (see the paper)
    :type precision: int
    :param wavelet: ROOT.WDM object, if provided, other parameters are ignored and extracted from the object
    :type wavelet: ROOT.WDM, optional
    """

    def __init__(self, m=None, k=None, beta_order=None, precision=None, wavelet=None):
        if not wavelet:
            #: ROOT.WDM object
            self.wavelet = ROOT.WDM(np.double)(m, k, beta_order, precision)
        else:
            self.wavelet = wavelet

        #: number of bands
        self.m = m
        #: k=n*m, where n is integer. K defines the width of the 'edge' of the basis function
        # in Fourier domain (see the paper). larger n, longer the WDM filter,
        # lower the spectral leakage between the bands.
        self.k = k
        #: defines the sharpness of the 'edge' of the basis function in Fourier domain (see paper)
        self.beta_order = beta_order
        #: defines filter length by truncation error quantified by P = -log10(1 - norm_of_filter) (see the paper)
        self.precision = precision

    def set_td_filter(self, coeff_factor, upsample_factor):
        """
        initialization of the time delay filters

        :param coeff_factor: define the number of the filter coefficients
        :type coeff_factor: int
        :param upsample_factor: upsample factor, defines the fundamental time delay step dt = tau/L , where tau is the sampling interval of the original time series
        :type upsample_factor: int
        """
        self.wavelet.setTDFilter(coeff_factor, upsample_factor)

    def allocate(self, data=None, n=None):
        """
        allocate memory for the WDM sliced array

        :param data: data to be stored in the WDM sliced array
        :type data: pycbc.types.timeseries.TimeSeries
        :param n: size of samples
        :type n: int
        """
        if data is None:
            return self.wavelet.allocate()

        if not n:
            return self.wavelet.allocate(len(data), data.data)
        else:
            return self.wavelet.allocate(n, data)

    def release(self):
        """
        release memory of the WDM sliced array
        """
        self.wavelet.release()

    def clone(self):
        """
        clone the WDM object
        """
        new_wavelet = self.wavelet.Clone()
        return WDM(wavelet=new_wavelet, m=self.m, k=self.k, beta_order=self.beta_order, precision=self.precision)

    def lightweight_dump(self):
        """
        lightweight duplication of the WDM object
        """
        new_wavelet = self.wavelet.Init()
        return WDM(wavelet=new_wavelet, m=self.m, k=self.k, beta_order=self.beta_order, precision=self.precision)

    def get_slice_size(self, level):
        """
        get slice of the WDM sliced array

        :param level: level of the wavelet transform
        :type level: int
        :return: slice of the WDM sliced array
        :rtype: numpy.ndarray
        """
        return self.wavelet.getSlice(level).size()

    @property
    def time_delay_filter_size(self):
        """
        half size of time delay filter
        """
        return self.wavelet.getTDFsize()

    @property
    def max_level(self):
        """
        maximum level of the wavelet transform
        """
        return self.wavelet.getMaxLevel()

    @property
    def size_at_zero_layer(self):
        """
        number of samples at zero level
        """
        return self.wavelet.getSlice(0).size()

    @property
    def max_index(self):
        """
        maximum index of the WDM sliced array
        """
        return self.size_at_zero_layer * (self.max_level + 1) - 1

    @property
    def m_H(self):
        """
        number of highpass wavelet filter coefficients
        """
        return self.wavelet.m_H

    @property
    def m_L(self):
        """
        number of lowpass wavelet filter coefficients
        """
        return self.wavelet.m_L

    @property
    def nWWS(self):
        """
        size of the wavelet work space
        """
        return self.wavelet.nWWS

    @property
    def nSTS(self):
        """
        size of the original time series
        """
        return self.wavelet.nSTS

    @nSTS.setter
    def nSTS(self, value):
        self.wavelet.nSTS = value

    @property
    def pWWS(self):
        """
        pointer to wavelet work space
        """
        return self.wavelet.pWWS

    def get_map_00(self, index):
        """
        get map00/90 value from index

        :param index:
        :return:
        """
        return self.wavelet.pWWS[index]

    def get_map_90(self, index):
        """
        get map00/90 value from index

        :param index:
        :return:
        """
        return self.wavelet.pWWS[index + self.max_index + 1]

    def t2w(self, k):
        """
        direct transform.

        :param k: param: -1 - orthonormal, 0 - power map, >0 - upsampled map. k = 0 requests power map of combined quadratures (not amplitudes for both)
        :type k: int
        """
        self.wavelet.t2w(k)
