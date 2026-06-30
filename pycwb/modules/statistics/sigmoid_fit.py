import numpy as np
import warnings
from scipy.optimize import curve_fit, brentq
from scipy.special import erfc
from iminuit import Minuit

def logNfit(x, par0, par1, par2, par3, par4):
    """
    Vectorized sigmoid (log-normal) fit function for efficiency curves.

    Computes the detection efficiency at a given hrss value x, parameterized
    by a piecewise sigmoid in log-space. Fully vectorized over x.

    Parameters
    ----------
    x : array-like
        hrss values at which to evaluate the efficiency.
    par0 : float
        log10(hrss50), the hrss at 50% detection efficiency.
    par1 : float
        Sigma (width) parameter of the sigmoid.
    par2 : float
        Beta-minus slope (below hrss50).
    par3 : float
        Beta-plus slope (above hrss50).
    par4 : int (0 or 1)
        Flag controlling sigmoid orientation.

    Returns
    -------
    ndarray
        Detection efficiency values in [0, 1] for each input x.
    """
    # Vectorized computation of y based on par0 and par4
    y = x - par0
    y = np.where(par4, -y, y)

    # Vectorized computation of s based on y and parameters par1, par2, par3
    s = np.where(y < 0, par1 * np.exp(y * par2), par1 * np.exp(y * par3))

    # Adjust s when y > 0 and par3 > 1. / y
    mask = (y > 0) & (par3 * y > 1.)
    s = np.where(mask, par1 * par3 * np.exp(1.), s)
    y = np.where(mask, 1, y)

    # Compute output based on s and y
    ny = np.where(y != 0, np.where(s > 0, np.abs(y / s), 100), y)

    result = np.where(y > 0, 1 - erfc(ny) / 2, erfc(ny) / 2)
    result = np.where(y == 0, 0.5, result)  # Handle the case y == 0

    return result

def fit(xdata, ydata, debug=False):
    """
    Fit a sigmoid function to the given data using Minuit for optimization.

    Parameters
    ----------
    xdata : array-like
        The x data points.
    ydata : array-like
        The y data points corresponding to xdata.
    debug : bool, optional
        If True, print debug information and plot the fit. Default is False.
    
    Returns
    -------
    list
        A list containing the chi-squared value, hrss50, hrss error, sigma, betam, betap, and flag.
    """ 

    # Find starting point at 0.5 efficiency
    xdata = np.array(xdata)
    ydata = np.array(ydata)
    idx = np.argmin(np.abs(ydata - 0.5))

    # Define chi2 function
    def chi2(par0, par1, par2, par3, par4):
        yfit = logNfit(xdata, par0, par1, par2, par3, par4)

        std = np.std(ydata)
        if std == 0:
            std = 1.0   # fallback: treat as unweighted residuals
        return np.sum(((yfit - ydata) / std) ** 2)

    # Track best fit across par4 attempts
    best_fit = None
    best_chi2 = np.inf
    # best_chi2 = 1
    for par4 in [0, 1]:
        # create minuit instance
        m = Minuit(chi2, par0 = xdata[idx], par1 = 0.3, par2 = 1.0, par3 = 1.0, par4 = par4)

        # set bounds
        m.limits['par0'] = (-25, -19)
        m.limits['par1'] = (0.08, 10.0)
        m.limits['par2'] = (0.00, 3.0)
        m.limits['par3'] = (0.00, 2.50)
        # m.limits['par4'] = (0, 1)
        m.fixed['par4'] = True

        # run minimization
        m.migrad()
        
        # Track best fit (use lowest chi2)
        if m.fval < best_chi2:
            best_chi2 = m.fval
            best_fit = m

    if best_fit is None:
        raise RuntimeError("Fit failed")

    if best_chi2 > 0.01:
        warnings.warn(f"Best fit error is high: {best_chi2:.2E} > 0.01")
        if best_chi2 > 1e20:
            warnings.warn(f"Fit failed")
    
    par0, sigma, betam, betap, flag = best_fit.values
    hrss50 = 10 ** par0
    hrssEr = (10 ** (par0 + best_fit.errors[0]) - 10 ** par0)
    chi2_val = best_fit.fval

    return [chi2_val, hrss50, hrssEr, sigma, betam, betap, flag]

def estimate_hrss(params, xlim, target_dp):
    """
    Estimate hrss for a given target detection probability (dp) using the fitted parameters.

    Parameters
    ----------
    params : list
        The fitted parameters [chi2, hrss50, sigma, betam, betap, flag].
    xlim : tuple
        The x-axis limits for the estimation.
    target_dp : float
        The target detection probability for which to estimate hrss.
    
    Returns
    -------
    float
        The estimated hrss for the given target detection probability.
    """
    hrss50, sigma, betam, betap, flag = params

    def minimizer(x):
        return logNfit(x, np.log10(hrss50), sigma, betam, betap, flag) - target_dp
    
    try:
        par0 = brentq(minimizer, xlim[0], xlim[1])
        return 10**par0
    except ValueError:
        return np.nan

# def fit(xdata, ydata, debug=False):
#     # find the ydata that close to 0.5 and the corresponding xdata
#     xdata = np.array(xdata)
#     ydata = np.array(ydata)
#     idx = np.argmin(np.abs(ydata - 0.5))

#     # Initial parameters
#     initial_guesses = [
#         [xdata[idx], 0.3, 0.5, 1, 0],
#         [xdata[idx], 0.5, 0.5, 1, 0],
#         [xdata[idx], 0.7, 0.5, 1, 0],
#         [xdata[idx], 0.3, 1, 1, 0],
#         [xdata[idx], 0.5, 1, 1, 0],
#         [xdata[idx], 0.7, 1, 1, 0],
#         [xdata[idx], 0.3, 2, 0.5, 0],
#         [xdata[idx], 0.5, 2, 0.5, 0],
#         [xdata[idx], 0.7, 2, 0.5, 0],
#         [xdata[idx], 0.3, 3, 1, 0],
#         [xdata[idx], 0.5, 3, 1, 0],
#         [xdata[idx], 0.7, 3, 1, 0],
#     ]

#     best_fit = None
#     best_chi2 = 1
#     for initial_params in initial_guesses:
#         if debug:
#             print(initial_params)

#         params_bounds = ([-25, 0, 0, 0, 0], [-19, np.inf, np.inf, 2.5, np.inf])

#         try:
#             params, covariance = curve_fit(logNfit, xdata, ydata, method='dogbox',
#                                            p0=initial_params, bounds=params_bounds,
#                                            )
#             chi2 = np.sum(((logNfit(xdata, *params) - ydata) / ydata.std()) ** 2)
#         except:
#             chi2 = 1e23
#             print("error in fit")
#         if debug:
#             print(chi2)
#         if chi2 < best_chi2:
#             best_fit = (params, covariance)
#             best_chi2 = chi2

#     if best_chi2 > 0.01:
#         warnings.warn(f"Best fit error is high: {best_chi2:.2E} > 0.01")
#         if best_chi2 > 1e20:
#             warnings.warn(f"Fit failed")

#     params, covariance = best_fit
#     # Calculate derived quantities and errors
#     hrss50 = 10 ** params[0]
#     hrssEr = (10 ** (params[0] + np.sqrt(np.diag(covariance)[0])) - 10 ** params[0])
#     sigma, betam, betap = params[1], params[2], params[3]
#     chi2 = np.sum(((logNfit(xdata, *params) - ydata) / ydata.std()) ** 2)

#     # Output results
#     if debug:
#         print(f"{chi2:.2E} {hrss50:.2E} +- {hrssEr:.2E} {sigma:.2E} {betam:.2E} {betap:.2E}")
#         import matplotlib.pyplot as plt
#         plt.plot(xdata, ydata, 'o')
#         new_xdata = np.linspace(xdata[0], xdata[-1], 1000)
#         plt.plot(new_xdata, logNfit(new_xdata, np.log10(hrss50), sigma, betam, betap, 0))

#     return [chi2, hrss50, hrssEr, sigma, betam, betap]
