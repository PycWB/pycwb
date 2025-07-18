# cython: boundscheck=False, wraparound=False, nonecheck=False, language_level=3
from libc.math cimport sqrt
from cython.view cimport array as cvarray
import cython

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef dpf_np_loops_local(float[::1] Fp0, float[::1] Fx0, float[:,::1] rms):
    cdef int NPIX = rms.shape[1]  # Exchanged
    cdef int NIFO = rms.shape[0]  # Exchanged

    cdef float[:,::1] f = cvarray(shape=(NIFO, NPIX), itemsize=sizeof(float), format="f")
    cdef float[:,::1] F = cvarray(shape=(NIFO, NPIX), itemsize=sizeof(float), format="f")
    cdef float[::1] si = cvarray(shape=(NPIX,), itemsize=sizeof(float), format="f")
    cdef float[::1] co = cvarray(shape=(NPIX,), itemsize=sizeof(float), format="f")
    cdef float[::1] fp = cvarray(shape=(NPIX,), itemsize=sizeof(float), format="f")
    cdef float[::1] fx = cvarray(shape=(NPIX,), itemsize=sizeof(float), format="f")
    cdef float[::1] ni = cvarray(shape=(NPIX,), itemsize=sizeof(float), format="f")

    cdef int i, j
    cdef float _ff, _FF, _fF, _si, _co, _AP, _nn, _cc, fF_new
    cdef float _o = 1e-7

    # Compute f and F
    for j in range(NIFO):
        for i in range(NPIX):
            f[j, i] = rms[j, i] * Fp0[j]
            F[j, i] = rms[j, i] * Fx0[j]

    # Compute ff, FF, and fF
    for i in range(NPIX):
        _ff = 0.
        _FF = 0.
        _fF = 0.

        for j in range(NIFO):
            _ff += f[j, i] * f[j, i]
            _FF += F[j, i] * F[j, i]
            _fF += f[j, i] * F[j, i]

        # Compute si, co, AP, nn, fp, and cc
        _si = 2. * _fF
        _co = _ff - _FF
        _AP = _ff + _FF
        _nn = sqrt(_co * _co + _si * _si)
        _cc = _co / (_nn + _o)
        fp[i] = (_AP + _nn) / 2.
        si[i] = sqrt((1. - _cc) / 2.)
        co[i] = sqrt((1. + _cc) / 2.) if _si > 0. else -sqrt((1. + _cc) / 2.)

    # Compute f_new, F_new, fF_new, F_new, fx, ni
    for i in range(NPIX):
        fF_new = 0.
        fx[i] = 0.
        ni[i] = 0.

        for j in range(NIFO):
            f[j,i], F[j,i] = f[j, i] * co[i] + F[j, i] * si[i], F[j, i] * co[i] - f[j, i] * si[i]
            fF_new += f[j,i] * F[j,i]
        fF_new /= (fp[i] + _o)

        for j in range(NIFO):
            F[j, i] -= f[j,i] * fF_new
            fx[i] += F[j, i] * F[j, i]
            ni[i] += f[j,i] ** 2

    cdef float NI = 0.0
    cdef int NN = 0

    # Compute NI and NN
    for i in range(NPIX):
        ni[i] /= (fp[i] * fp[i] + _o)
        NI += fx[i] / (ni[i] + _o)
        if fp[i] > 0.:
            NN += 1

    return sqrt(NI / (NN + 0.01)), fp, fx, si, co, ni


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef float calculate_dpf(float[:,::1] FP, float[:,::1] FX, float[:,::1] rms, int n_sky, float gamma_regulator, float network_energy_threshold):
    cdef int[::1] MM = cvarray(shape=(n_sky,), itemsize=sizeof(int), format="i")
    # cdef int[:] mm = cvarray(shape=(n_sky,), itemsize=sizeof(int), format="f", zeroed=True)

    cdef int i
    cdef float aa
    cdef float[::1] temp_fp, temp_fx, temp_si, temp_co, temp_ni

    cdef int FF = 0
    cdef int ff = 0
    for i in range(n_sky):
        MM[i] = 1
        FF += 1
        aa, temp_fp, temp_fx, temp_si, temp_co, temp_ni = dpf_np_loops_local(FP[i], FX[i], rms)

        ff += 1 if aa > gamma_regulator else 0


    return (FF ** 2 / (ff ** 2 + 1.e-9) - 1) * network_energy_threshold
