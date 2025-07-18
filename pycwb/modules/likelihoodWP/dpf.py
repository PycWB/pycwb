from math import sqrt

import numpy as np
from numba import njit, prange, guvectorize, vectorize, float32, uint32


@njit(parallel=True, cache=True)
def calculate_dpf(FP, FX, rms, n_sky, n_ifo, gamma_regulator, network_energy_threshold):
    FP = FP.astype(np.float32)
    FX = FX.astype(np.float32)
    rms = rms.astype(np.float32)
    MM = np.zeros(n_sky)
    aa = np.zeros(n_sky)

    # check shape of FP, FX, and rms
    if FP.shape != (n_sky, n_ifo) and FX.shape != (n_sky, n_ifo) and rms.shape[1] != n_ifo:
        raise ValueError('FP and FX must have the shape of (n_sky, n_ifo) and rms must have the shape of (n_pix, n_ifo)')
    for i in prange(n_sky):
        MM[i] = 1
        aa[i], _, _, _, _, _, _, _ = dpf_np_loops_vec(FP[i], FX[i], rms)

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
    # _0 = np.float32(0)
    # _2 = np.float32(2)
    # _1 = np.float32(1)
    NI = np.float32(0.0)
    NN = np.uint32(0)

    # Compute f and F
    for i in range(NPIX):
        for j in range(NIFO):
            f[i, j] = rms[i, j] * Fp0[j]
            F[i, j] = rms[i, j] * Fx0[j]

    for i in range(NPIX):
        # Compute ff, FF, and fF
        _ff = float32(0.)
        _FF = float32(0.)
        _fF = float32(0.)
        for j in range(NIFO):
            _ff += f[i, j] * f[i, j]
            _FF += F[i, j] * F[i, j]
            _fF += F[i, j] * f[i, j]

        # Compute si, co, AP, nn, fp, and cc
        _si = float32(2.0) * _fF
        _co = _ff - _FF
        _AP = _ff + _FF
        _nn = sqrt(_co * _co + _si * _si)
        _cc = _co / (_nn + _o)
        fp[i] = (_AP + _nn) / float32(2.0)
        si[i], co[i] = sqrt((float32(1.) - _cc) / float32(2.0)), (sqrt((float32(1.) + _cc) / float32(2.0)) if _si > float32(0.) else - sqrt((float32(1.) + _cc) / float32(2.0)))

    # Compute f_new, F_new, fF_new, F_new, fx, ni
    for i in range(NPIX):
        fF_new = np.float32(0.)
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
        NN += 1 if fp[i] > float32(0.) else 0

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


@vectorize([float32(float32, float32)])
def mul_vec(a, b):
    return a * b


@vectorize([float32(float32, float32)])
def div_vec(a, b):
    _o = float32(0.0001)
    return a / (b + _o)


@vectorize([float32(float32, float32)])
def add_vec(a, b):
    return a + b


@vectorize([float32(float32, float32)])
def sub_vec(a, b):
    return a - b


@vectorize([float32(float32, float32)])
def norm_vec(a, b):
    return sqrt(a * a + b * b)


@vectorize([float32(float32, float32)])
def avg_vec(a, b):
    return (a + b) / float32(2.)


@vectorize([float32(float32)])
def sin_from_cc(a):
    return sqrt((float32(1.) - a) / float32(2.))


@vectorize([float32(float32, float32)])
def cos_from_cc(a, si):
    return sqrt((float32(1.) + a) / float32(2.)) if si > float32(0.) else - sqrt((float32(1.) + a) / float32(2.))


@vectorize([uint32(float32)])
def pos_sign_vec(a):
    return uint32(1) if a > float32(0.) else uint32(0)


@vectorize([float32(float32, float32, float32, float32)])
def rotate_fp_vec(fp, fx, si, co):
    return fp * co + fx * si


@vectorize([float32(float32, float32, float32, float32)])
def rotate_fx_vec(fp, fx, si, co):
    return fx * co - fp * si


@vectorize([float32(float32)])
def quad_vec(a):
    return a * a * a * a


@guvectorize([(float32[:], float32[:], float32[:])], '(n),(n)->()')
def sum_vec(a, b, res):
    s = float32(0.)
    for i in range(a.shape[0]):
        s += a[i] * b[i]
    res[0] = s


@njit(cache=True)
def dpf_np_loops_vec(Fp0, Fx0, rms):
    """
    Compute the dominant polarization frame (DPF)

    Parameters
    ----------
    Fp0 : np.ndarray
        The Fp0 vector for the current sky location.
    Fx0 : np.ndarray
        The Fx0 vector for the current sky location.
    rms : np.ndarray
        The rms values for the pixels, shape (NPIX, NIFO).

    Returns
    -------
    tuple
        - NI : float
            The normalized index. (?)
        - f: np.ndarray
            The plus polarization component in the DPF.
        - F: np.ndarray
            The cross polarization component in the DPF.
        - fp: np.ndarray
            |f+|^2 
        - fx: np.ndarray
            |fx|^2
        - si: np.ndarray
            The sine component of the DPF.
        - co: np.ndarray
            The cosine component of the DPF.
        - ni: np.ndarray
            The network index for each pixel.
    """
    NPIX, NIFO = rms.shape
    NPIX = uint32(NPIX)
    NIFO = uint32(NIFO)

    # variables for return
    f = np.empty((NPIX, NIFO), dtype=np.float32)
    F = np.empty((NPIX, NIFO), dtype=np.float32)
    si = np.empty(NPIX, dtype=np.float32)
    co = np.empty(NPIX, dtype=np.float32)
    fp = np.empty(NPIX, dtype=np.float32)
    fx = np.zeros(NPIX, dtype=np.float32)
    ni = np.zeros(NPIX, dtype=np.float32)

    _o = float32(0.0001)

    # Prepare constants
    # NI = np.float32(0.0)
    # NN = np.uint32(0)

    # Compute f and F
    for j in range(NIFO):
        for i in range(NPIX):
            f[i, j] = mul_vec(rms[i, j], Fp0[j])
            F[i, j] = mul_vec(rms[i, j], Fx0[j])

    # Compute ff, FF, and fF
    for i in range(NPIX):
        _ff = float32(0.)
        _FF = float32(0.)
        _fF = float32(0.)

        for j in range(NIFO):
            _ff += f[i, j] * f[i, j]
            _FF += F[i, j] * F[i, j]
            _fF += F[i, j] * f[i, j]

        # Compute si, co, AP, nn, fp, and cc
        _si = mul_vec(float32(2.), _fF)  # rotation 2*sin*cos*norm
        _co = sub_vec(_ff, _FF)          # rotation (cos^2-sin^2)*norm
        _AP = add_vec(_ff, _FF)          # total antenna norm
        _nn = norm_vec(_co, _si)         # co/si norm    np.sqrt(_co * _co + _si * _si)
        _cc = div_vec(_co, _nn)          # cos(2p)       _co / (_nn + 0.0001)
        fp[i] = avg_vec(_AP, _nn)        # |f+|^2        (_AP + _nn) / 2.
        si[i] = sin_from_cc(_cc)         # |sin(p)|      sqrt((1. - _cc) / 2.)
        co[i] = cos_from_cc(_cc, _si)    # cos(p)        (sqrt((1. + _cc) / 2.) if _si > 0.0 else - sqrt((1. + _cc) / 2.))

    # Compute f_new, F_new, fF_new, F_new, fx, ni
    for i in range(NPIX):
        for j in range(NIFO):
            f[i, j], F[i, j] = f[i, j] * co[i] + F[i, j] * si[i], F[i, j] * co[i] - f[i, j] * si[i]
            # f[i, j] = rotate_fp_vec(f[i, j], F[i, j], si[i], co[i])
            # F[i, j] = rotate_fx_vec(f[i, j], F[i, j], si[i], co[i])

        fF_new = float32(0.)
        for j in range(NIFO):
            fF_new += f[i, j] * F[i, j]
        # fF_new /= (fp[i] + _o)
        fF_new = div_vec(fF_new, fp[i])

        for j in range(NIFO):
            F[i, j] -= f[i, j] * fF_new
            fx[i] += F[i, j] * F[i, j]
            ni[i] += f[i, j] ** 4
            # ni[i] += quad_vec(f[i, j])

    NI, NN = float32(0.0), uint32(0)

    # Compute NI and NN
    for i in range(NPIX):
        # ni[i] /= (fp[i] * fp[i] + _o)
        ni[i] = div_vec(ni[i], mul_vec(fp[i], fp[i]))
        # NI += fx[i] / (ni[i] + _o)
        NI += div_vec(fx[i], ni[i])     # sum of |fx|^2/2/ni
        # if fp[i] > float32(0.0):
        NN += pos_sign_vec(fp[i])       # pixel count
        # NN += 1 if fp[i] > 0.0 else 0

    return sqrt(NI / (NN + 0.01)), f, F, fp, fx, si, co, ni
