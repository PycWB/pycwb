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
