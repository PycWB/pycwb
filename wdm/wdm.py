import math
import numpy as np
from numpy.fft import fft
from numba import njit
from .fourier_coeff import load_fourier_coeff


class WDM:
    """
    WDM class
    """

    def __init__(self, M, K, beta_order, precision):
        self.M = M
        self.K = K
        self.beta_order = beta_order
        self.precision = precision

        Cos, Cos2, SinCos, CosSize, Cos2Size, SinCosSize = load_fourier_coeff()

        n_max = int(3e5)
        filter = get_filter(M, K, beta_order, n_max, Cos, CosSize)

        residual = 1 - filter[0] ** 2
        prec = 10 ** (-precision)

        N = 1
        M2 = M * 2
        while residual > prec or (N - 1) % M2 or N // M2 < 3:
            residual -= 2 * filter[N] ** 2
            N += 1

        self.filter = filter[:N]
        self.m_H = N


@njit(cache=True)
def get_filter(M, K, BetaOrder, n, Cos, CosSize):
    # Extract necessary attributes from the wdm
    B = math.pi / K
    A = (K - M) * math.pi / (2.0 * K * M)
    K2 = K ** 2
    gNorm = math.sqrt(2 * M) / math.pi

    # Initialize the filter array
    filter = np.zeros(n)

    # Get the Fourier array
    Fourier = Cos[BetaOrder]
    nFourier = CosSize[BetaOrder]
    fourier = np.copy(Fourier)

    # Calculate filter coefficients
    filter[0] = (A + fourier[0] * B) * gNorm
    fNorm = filter[0] ** 2
    fourier[0] /= math.sqrt(2)

    for i in range(1, n):
        di = float(i)
        i2 = di * di
        sumEven = 0
        sumOdd = 0

        # Calculating the sums for even and odd
        for j in range(nFourier):
            if i % K:
                pass
            elif j == i // K:
                continue
            coeff = di / (i2 - j ** 2 * K2)
            if j & 1:
                sumOdd += coeff * fourier[j]
            else:
                sumEven += coeff * fourier[j]

        # Calculating intAB
        intAB = 0
        if i % K == 0 and i // K < nFourier:
            intAB = fourier[i // K] * B / 2 * math.cos(i * A)

        intAB += 2 * (sumEven * math.sin(i * B / 2) * math.cos(i * math.pi / (2 * M)) -
                      sumOdd * math.sin(i * math.pi / (2 * M)) * math.cos(i * B / 2))

        filter[i] = gNorm * (math.sin(i * A) / di + math.sqrt(2) * intAB)
        fNorm += 2 * filter[i] ** 2

    return filter


@njit(cache=True)
def t2w(M, m_H, WWS, filter, MM):
    """
    Transform time series to WDM

    :param M: max layers
    :param m_H:
    :param WWS: time series
    :param nWWS: length of time series
    :param filter: filter coefficients
    :param MM:  MM = 0 requests power map of combined quadratures (not amplitudes for both)
    :return:
    """

    M1 = M + 1
    M2 = M * 2
    nWDM = m_H
    nTS = len(WWS)
    KK = MM

    if MM <= 0:
        MM = M

    # adjust nWWS to be a multiple of MM
    nWWS = len(WWS)
    # this->nWWS += this->nWWS%MM ? MM-this->nWWS%MM : 0;
    nWWS += MM - nWWS % MM if nWWS % MM else 0

    # initialize time series with boundary conditions (mirror)
    m = nWWS + 2 * nWDM
    ts = np.zeros(m)

    for n in range(nWDM):
        ts[nWDM - n] = WWS[n]
    for n in range(nTS):
        ts[nWDM + n] = WWS[n]
    for n in range(int(m - nWDM - nTS)):
        ts[n + nWDM + nTS] = WWS[nTS - n - 1]


    # create symmetric arrays
    wdm = filter[:nWDM]
    # INV = np.array(filter[nWDM - 1::-1])

    # WDM = INV[::-1]

    # reallocate TF array
    N = int(nWWS / MM)
    L = 2 * N * M1 if KK < 0 else N * M1
    m_L = m_H if KK < 0 else 0

    pWDM = np.zeros(L)  # Assuming pWWS is a numpy array

    odd = 0
    sqrt2 = np.sqrt(2)

    for n in range(N):
        # create references
        map00 = pWDM[n * M1:]
        map90 = pWDM[(N + n) * M1:]
        pTS = ts[nWDM + n * MM:]

        re = np.zeros(M2)
        im = np.zeros(M2)

        J = M2
        for j in range(0, nWDM - 1, M2):
            J = M2 + j
            pTS_inv = ts[nWDM + n * MM - J:]
            for m in range(M2):
                re[m] += pTS[j + m] * wdm[j + m] + pTS_inv[m] * wdm[J - m]

        re[0] += wdm[J] * pTS[J]

        # Perform FFT
        fft_result = fft(re)
        re, im = fft_result.real, fft_result.imag

        re[0] = im[0] = re[0] / sqrt2
        re[M] = im[M] = re[M] / sqrt2

        if KK < 0:
            for m in range(M + 1):
                if (m + odd) & 1:
                    map00[m] = sqrt2 * im[m]
                    map90[m] = sqrt2 * re[m]
                else:
                    map00[m] = sqrt2 * re[m]
                    map90[m] = -sqrt2 * im[m]
        else:  # power map
            map00[:M + 1] = re[:M + 1] ** 2 + im[:M + 1] ** 2

        odd = 1 - odd

    return m_L, nWWS, pWDM.reshape(2, N, M1)
