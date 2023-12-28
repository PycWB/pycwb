from math import sqrt

import numpy as np
from numba import njit, prange


@njit(parallel=True, cache=True)
def calculate_dpf(FP, FX, rms, n_sky, gamma_regulator, network_energy_threshold):
    MM = np.zeros(n_sky)
    aa = np.zeros(n_sky)
    for i in prange(n_sky):
        MM[i] = 1
        aa[i], fp, fx, si, co, ni = dpf_np_loops_local(FP[i], FX[i], rms)

    FF = MM.sum()
    ff = (aa > gamma_regulator).sum()

    return (FF ** 2 / (ff ** 2 + 1.e-9) - 1) * network_energy_threshold


@njit(cache=True)
def avx_dpf_ps(Fp0, Fx0, rms):
    # sign = np.sign(np.dot(Fp0, Fx0))

    n_pix = len(rms)
    # n_ifo = len(FP)

    NI = 0
    NN = 0

    fp_array = np.zeros(n_pix)
    fx_array = np.zeros(n_pix)
    si_array = np.zeros(n_pix)
    co_array = np.zeros(n_pix)
    ni_array = np.zeros(n_pix)

    for pid in range(n_pix):
        f = rms[pid] * Fp0
        F = rms[pid] * Fx0
        ff = np.dot(f, f)
        FF = np.dot(F, F)
        fF = np.dot(F, f)

        si = 2 * fF  # rotation 2*sin*cos*norm
        co = ff - FF  # rotation (cos^2-sin^2)*norm
        AP = ff + FF  # total antenna norm
        cc = co ** 2
        ss = si ** 2
        nn = sqrt(cc + ss)  # co/si norm
        fp = (AP + nn) / 2  # |f+|^2
        cc = co / (nn + 0.0001)  # cos(2p)
        ss = 1 if si > 0 else -1  # 1 if sin(2p)>0. or-1 if sin(2p)<0.
        si = sqrt((1 - cc) / 2)  # |sin(p)|
        co = sqrt((1 + cc) / 2)  # |cos(p)|
        co = co * ss  # cos(p)

        f, F = f * co + F * si, F * co - f * si
        _cc = f * f
        nn = np.dot(_cc, _cc)
        fF = np.dot(f, F)

        fF = fF / (fp + 0.0001)

        F -= f * fF
        fx = np.dot(F, F)

        ni = nn / (fp ** 2 + 0.0001)  # network index
        ff = ni + 0.0001  # network index
        NI += fx / ff  # sum of |fx|^2/2/ni
        NN += 1 if fp > 0 else 0  # pixel count

        fp_array[pid] = fp
        fx_array[pid] = fx
        si_array[pid] = si
        co_array[pid] = co
        ni_array[pid] = ni

    return sqrt(NI / (NN + 0.01)), fp_array, fx_array, si_array, co_array, ni_array

@njit(cache=True)
def dpf_np_loops_local(Fp0, Fx0, rms):
    NPIX, NIFO = rms.shape

    # variables for return
    f = np.empty((NPIX, NIFO), dtype=np.float32)
    F = np.empty((NPIX, NIFO), dtype=np.float32)
    si = np.empty(NPIX, dtype=np.float32)
    co = np.empty(NPIX, dtype=np.float32)
    fp = np.empty(NPIX, dtype=np.float32)
    fx = np.zeros(NPIX, dtype=np.float32)
    ni = np.zeros(NPIX, dtype=np.float32)

    # Prepare constants
    _o = np.float32(0.0001)
    _0 = np.float32(0)
    _2 = np.float32(2)
    _1 = np.float32(1)
    NI = np.float32(0.0)
    NN = np.uint32(0)

    # local variables
    _si = np.float32(0.0)
    _co = np.float32(0.0)
    _AP = np.float32(0.0)
    _nn = np.float32(0.0)
    _cc = np.float32(0.0)
    _ff = np.float32(0.0)
    _FF = np.float32(0.0)
    _fF = np.float32(0.0)

    # Compute f and F
    for i in range(NPIX):
        for j in range(NIFO):
            f[i, j] = rms[i, j] * Fp0[j]
            F[i, j] = rms[i, j] * Fx0[j]

    # Compute ff, FF, and fF
    for i in range(NPIX):
        _ff = _0
        _FF = _0
        _fF = _0
        for j in range(NIFO):
            _ff += f[i, j] * f[i, j]
            _FF += F[i, j] * F[i, j]
            _fF += F[i, j] * f[i, j]

    # Compute si, co, AP, nn, fp, and cc
    for i in range(NPIX):
        _si = _2 * _fF
        _co = _ff - _FF
        _AP = _ff + _FF
        _nn = sqrt(_co * _co + _si * _si)
        _cc = _co / (_nn + _o)
        fp[i] = (_AP + _nn) / _2
        si[i], co[i] = sqrt((_1 - _cc) / _2), (sqrt((_1 + _cc) / _2) if _si > _0 else - sqrt((_1 + _cc) / _2))

    # Compute f_new, F_new, fF_new, F_new, fx, ni
    for i in range(NPIX):
        fF_new = _0
        for j in range(NIFO):
            f[i, j], F[i, j] = f[i, j] * co[i] + F[i, j] * si[i], F[i, j] * co[i] - f[i, j] * si[i]
            fF_new += f[i, j] * F[i, j]
        fF_new /= (fp[i] + _o)

        for j in range(NIFO):
            F[i, j] -= f[i, j] * fF_new
            fx[i] += F[i, j] * F[i, j]
            ni[i] += f[i, j] ** 4

    # Compute NI and NN
    for i in range(NPIX):
        ni[i] /= (fp[i] * fp[i] + _o)
        NI += fx[i] / (ni[i] + _o)
        NN += 1 if fp[i] > _0 else 0

    return sqrt(NI / (NN + np.float32(0.01))), fp, fx, si, co, ni


@njit(cache=True)
def dpf_np_loops(Fp0, Fx0, rms):
    NPIX, NIFO = rms.shape

    # Initialize arrays
    f = np.zeros_like(rms)
    F = np.zeros_like(rms)
    ff = np.zeros(NPIX)
    FF = np.zeros(NPIX)
    fF = np.zeros(NPIX)
    si = np.zeros(NPIX)
    co = np.zeros(NPIX)
    AP = np.zeros(NPIX)
    nn = np.zeros(NPIX)
    fp = np.zeros(NPIX)
    fF_new = np.zeros(NPIX)
    fx = np.zeros(NPIX)
    ni = np.zeros(NPIX)
    cc = np.zeros(NPIX)

    # Compute f and F
    for i in range(NPIX):
        for j in range(NIFO):
            f[i, j] = rms[i, j] * Fp0[j]
            F[i, j] = rms[i, j] * Fx0[j]

    # Compute ff, FF, and fF
    for i in range(NPIX):
        for j in range(NIFO):
            ff[i] += f[i, j] * f[i, j]
            FF[i] += F[i, j] * F[i, j]
            fF[i] += F[i, j] * f[i, j]

    # Compute si, co, AP, nn, fp, and cc
    for i in range(NPIX):
        si[i] = 2 * fF[i]
        co[i] = ff[i] - FF[i]
        AP[i] = ff[i] + FF[i]
        nn[i] = sqrt(co[i] * co[i] + si[i] * si[i])
        fp[i] = (AP[i] + nn[i]) / 2
        cc[i] = co[i] / (nn[i] + 0.0001)
        si[i], co[i] = sqrt((1 - cc[i]) / 2), sqrt((1 + cc[i]) / 2) * (1 if si[i] > 0 else -1)

    # Compute f_new, F_new, fF_new, F_new, fx, ni
    for i in range(NPIX):
        for j in range(NIFO):
            f[i, j], F[i, j] = f[i, j] * co[i] + F[i, j] * si[i], F[i, j] * co[i] - f[i, j] * si[i]
            fF_new[i] += f[i, j] * F[i, j]
        fF_new[i] /= (fp[i] + 0.0001)
        for j in range(NIFO):
            F[i, j] -= f[i, j] * fF_new[i]
            fx[i] += F[i, j] * F[i, j]
            ni[i] += f[i, j] ** 4

    # Compute NI and NN
    NI = 0
    NN = 0
    for i in range(NPIX):
        ni[i] /= (fp[i] * fp[i] + 0.0001)
        NI += fx[i] / (ni[i] + 0.0001)
        NN += 1 if fp[i] > 0 else 0

    return sqrt(NI / (NN + 0.01)), fp, fx, si, co, ni


@njit(cache=True)
def dpf_np(Fp0, Fx0, rms):
    """
    :param Fp0: Fp0 is a NIFO x 1 array
    :param Fx0: Fx0 is a NIFO x 1 array
    :param rms: rms is a NIFO x NPIX array
    :return:
    """
    # sign = np.sign(np.dot(Fp0, Fx0))

    f = rms * Fp0
    F = rms * Fx0

    ff = (f * f).sum(axis=1)
    FF = (F * F).sum(axis=1)
    fF = (F * f).sum(axis=1)

    si = 2 * fF  # rotation 2*sin*cos*norm
    co = ff - FF  # rotation (cos^2-sin^2)*norm
    AP = ff + FF  # total antenna norm

    nn = np.sqrt(co * co + si * si)  # co/si norm
    fp = (AP + nn) / 2  # |f+|^2
    cc = co / (nn + 0.0001)  # cos(2p)
    si, co = np.sqrt((1 - cc) / 2), np.sqrt((1 + cc) / 2) * np.where(si > 0, 1, -1)

    f, F = f * co[:, np.newaxis] + F * si[:, np.newaxis], F * co[:, np.newaxis] - f * si[:, np.newaxis]

    fF = (f * F).sum(axis=1) / (fp + 0.0001)

    F -= f * fF[:, None]

    fx = (F * F).sum(axis=1)

    ni = (f ** 4).sum(axis=1) / (fp * fp + 0.0001)  # network index

    NI = (fx / (ni + 0.0001)).sum()  # sum of |fx|^2/2/ni

    NN = (fp > 0).sum()  # pixel count

    return sqrt(NI / (NN + 0.01)), fp, fx, si, co, ni
