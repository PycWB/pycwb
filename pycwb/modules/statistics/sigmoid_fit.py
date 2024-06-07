import numpy as np
import warnings
from scipy.optimize import curve_fit
from scipy.special import erfc


def logNfit(x, par0, par1, par2, par3, par4):
    # Vectorized computation of y based on par0 and par4
    y = x - par0
    y = np.where(par4, -y, y)

    # Vectorized computation of s based on y and parameters par1, par2, par3
    s = np.where(y < 0, par1 * np.exp(y * par2), par1 * np.exp(y * par3))

    # Adjust s when y > 0 and par3 > 1. / y
    mask = (y > 0) & (par3 > 1. / y)
    s = np.where(mask, par1 * par3 * np.exp(1.), s)
    y = np.where(mask, 1, y)

    # Compute output based on s and y
    ny = np.where(y != 0, np.where(s > 0, np.abs(y / s), 100), y)

    result = np.where(y > 0, 1 - erfc(ny) / 2, erfc(ny) / 2)
    result = np.where(y == 0, 0.5, result)  # Handle the case y == 0

    return result


def fit(xdata, ydata, debug=False):
    # find the ydata that close to 0.5 and the corresponding xdata
    xdata = np.array(xdata)
    ydata = np.array(ydata)
    idx = np.argmin(np.abs(ydata - 0.5))

    # Initial parameters
    initial_guesses = [
        [xdata[idx], 0.3, 0.5, 1, 0],
        [xdata[idx], 0.5, 0.5, 1, 0],
        [xdata[idx], 0.7, 0.5, 1, 0],
        [xdata[idx], 0.3, 1, 1, 0],
        [xdata[idx], 0.5, 1, 1, 0],
        [xdata[idx], 0.7, 1, 1, 0],
        [xdata[idx], 0.3, 2, 0.5, 0],
        [xdata[idx], 0.5, 2, 0.5, 0],
        [xdata[idx], 0.7, 2, 0.5, 0],
        [xdata[idx], 0.3, 3, 1, 0],
        [xdata[idx], 0.5, 3, 1, 0],
        [xdata[idx], 0.7, 3, 1, 0],
    ]

    best_fit = None
    best_chi2 = 1
    for initial_params in initial_guesses:
        if debug:
            print(initial_params)

        params_bounds = ([-25, 0, 0, 0, 0], [-19, np.inf, np.inf, 2.5, np.inf])

        try:
            params, covariance = curve_fit(logNfit, xdata, ydata, method='dogbox',
                                           p0=initial_params, bounds=params_bounds,
                                           )
            chi2 = np.sum(((logNfit(xdata, *params) - ydata) / ydata.std()) ** 2)
        except:
            chi2 = 1e23
            print("error in fit")
        if debug:
            print(chi2)
        if chi2 < best_chi2:
            best_fit = (params, covariance)
            best_chi2 = chi2

    if best_chi2 > 0.01:
        warnings.warn(f"Best fit error is high: {best_chi2:.2E} > 0.01")
        if best_chi2 > 1e20:
            warnings.warn(f"Fit failed")

    params, covariance = best_fit
    # Calculate derived quantities and errors
    hrss50 = 10 ** params[0]
    hrssEr = (10 ** (params[0] + np.sqrt(np.diag(covariance)[0])) - 10 ** params[0])
    sigma, betam, betap = params[1], params[2], params[3]
    chi2 = np.sum(((logNfit(xdata, *params) - ydata) / ydata.std()) ** 2)

    # Output results
    if debug:
        print(f"{chi2:.2E} {hrss50:.2E} +- {hrssEr:.2E} {sigma:.2E} {betam:.2E} {betap:.2E}")
        import matplotlib.pyplot as plt
        plt.plot(xdata, ydata, 'o')
        new_xdata = np.linspace(xdata[0], xdata[-1], 1000)
        plt.plot(new_xdata, logNfit(new_xdata, np.log10(hrss50), sigma, betam, betap, 0))

    return [chi2, hrss50, hrssEr, sigma, betam, betap]
