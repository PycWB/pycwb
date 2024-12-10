"""
This module is inspired by ROOT::TMath::Poisson and ROOT::TF1::GetQuantiles
"""
import numpy as np
from scipy.special import gammaln
from scipy.integrate import quad
from scipy.interpolate import interp1d
from numba import njit, prange


@njit(fastmath=True,error_model='numpy',parallel=True)
def gammaln_nr(z):
    """Numerical Recipes 6.1"""
    #Don't use global variables.. (They only can be changed if you recompile the function)
    coefs = np.array([
        57.1562356658629235, -59.5979603554754912,
        14.1360979747417471, -0.491913816097620199,
        .339946499848118887e-4, .465236289270485756e-4,
        -.983744753048795646e-4, .158088703224912494e-3,
        -.210264441724104883e-3, .217439618115212643e-3,
        -.164318106536763890e-3, .844182239838527433e-4,
        -.261908384015814087e-4, .368991826595316234e-5])

    y = z
    tmp = z + 5.24218750000000000
    tmp = (z + 0.5) * np.log(tmp) - tmp
    ser = 0.999999999999997092

    n = coefs.shape[0]
    for j in range(n):
        y = y + 1.
        ser = ser + coefs[j] / y

    return tmp + np.log(2.5066282746310005 * ser / z)


@njit(fastmath=True,error_model='numpy',parallel=True)
def gammaln_nr_p(z):
    """Numerical Recipes 6.1"""
    #Don't use global variables.. (They only can be changed if you recompile the function)
    coefs = np.array([
        57.1562356658629235, -59.5979603554754912,
        14.1360979747417471, -0.491913816097620199,
        .339946499848118887e-4, .465236289270485756e-4,
        -.983744753048795646e-4, .158088703224912494e-3,
        -.210264441724104883e-3, .217439618115212643e-3,
        -.164318106536763890e-3, .844182239838527433e-4,
        -.261908384015814087e-4, .368991826595316234e-5])

    out=np.empty(z.shape[0])


    for i in prange(z.shape[0]):
        y = z[i]
        tmp = z[i] + 5.24218750000000000
        tmp = (z[i] + 0.5) * np.log(tmp) - tmp
        ser = 0.999999999999997092

        n = coefs.shape[0]
        for j in range(n):
            y = y + 1.
            ser = ser + coefs[j] / y

        out[i] = tmp + np.log(2.5066282746310005 * ser / z[i])
    return out


@njit
def continues_poisson(x, mu):
    """
    compute the Poisson distribution function for (x,mu)
 The Poisson PDF is implemented by means of Euler's Gamma-function
 (for the factorial), so for any x integer argument it is correct.
 BUT for non-integer x values, it IS NOT equal to the Poisson distribution.

    Parameters
    ----------
    x : float or array_like
        The number of events
    mu : float
        The mean number of events

    Returns
    -------
    float or array_like
        The Poisson distribution function for (x, mu)
    """
    return np.exp(x * np.log(mu) - mu - gammaln_nr(x + 1))


def get_percentiles(mu, percentiles):
    """
    Get the quantiles for a Poisson distribution with mean mu and given percentiles

    Parameters
    ----------
    mu : float
        The mean number of events
    percentiles : float or array_like
        The percentiles to compute the quantiles for

    Returns
    -------
    float or array_like
        The quantiles for the Poisson distribution with mean mu and given percentiles
    """
    max_val = 10 * mu if mu > 10 else 100  # Maximum range for interpolation
    x = np.linspace(0, max_val, 1000)     # Fine grid of x-values for interpolation

    # integrate the pmf to get the cdf
    # FIXME: the integration can't reach 1 for mu <= 2
    cdf = [quad(continues_poisson, 0, x_val, args=(mu))[0] for x_val in x]

    # interpolate the cdf to get the quantiles
    interpolate = interp1d(cdf, x, bounds_error=False, fill_value="extrapolate")
    return interpolate(percentiles)


def get_percentiles_ROOT(mu, prob_sum):
    import ROOT

    q = np.array([0., 0., 0., 0., 0., 0.])
    max_val = 10 * mu if mu > 10 else 100
    f = ROOT.TF1("poisson", "TMath::Poisson(x,[1])", 0, max_val)
    f.SetParameter(1, mu)
    f.SetNpx(1000)

    f.GetQuantiles(6, q, prob_sum)
    return q
