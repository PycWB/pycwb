import pickle
import numpy as np

from pycwb.modules.likelihoodWP.likelihood import load_data_from_pixels

# load FP, FX, rms, n_sky, gamma_regulator, network_energy_threshold from pickle
with open('test_data.pkl', 'rb') as f:
    test_data = pickle.load(f)

FP = test_data['FP']
FX = test_data['FX']
pixels = test_data['pixels']
n_ifo = test_data['n_ifo']
n_sky = test_data['n_sky']
gamma_regulator = test_data['gamma_regulator']
network_energy_threshold = test_data['network_energy_threshold']
rms, td00, td90, td_energy = load_data_from_pixels(pixels, n_ifo)

from numba import njit, prange, vectorize, guvectorize, float32, uint32

import numpy as np
from math import sqrt


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

    _ff = np.sum(f * f, axis=1)
    _FF = np.sum(F * F, axis=1)
    _fF = np.sum(f * F, axis=1)

    _si = 2.0 * _fF
    _co = _ff - _FF
    _AP = _ff + _FF
    _nn = np.sqrt(_co * _co + _si * _si)
    _cc = _co / (_nn + 0.0001)
    fp = (_AP + _nn) / 2.0

    si = np.sqrt((1.0 - _cc) / 2.0)
    co = np.where(_si > 0.0, np.sqrt((1.0 + _cc) / 2.0), -np.sqrt((1.0 + _cc) / 2.0))

    f, F = f * co[:, np.newaxis] + F * si[:, np.newaxis], F * co[:, np.newaxis] - f * si[:, np.newaxis]
    fF_new = np.sum(f * F, axis=1) / (fp + 0.0001)

    F -= f * fF_new[:, np.newaxis]
    fx = np.sum(F * F, axis=1)
    ni = np.sum(f ** 4, axis=1) / ((fp * fp) + 0.0001)

    NI = np.sum(fx / (ni + 0.0001))
    NN = np.sum(fp > 0.0)

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
        _si = mul_vec(float32(2.), _fF)  # 2. * _fF
        _co = sub_vec(_ff, _FF)  # _ff - _FF
        _AP = add_vec(_ff, _FF)  # _ff + _FF
        _nn = norm_vec(_co, _si)  # np.sqrt(_co * _co + _si * _si)
        _cc = div_vec(_co, _nn)  # _co / (_nn + 0.0001)
        fp[i] = avg_vec(_AP, _nn)  # (_AP + _nn) / 2.
        si[i] = sin_from_cc(_cc)   # sqrt((1. - _cc) / 2.)
        co[i] = cos_from_cc(_cc,_si) # (sqrt((1. + _cc) / 2.) if _si > 0.0 else - sqrt((1. + _cc) / 2.))

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
        NI += div_vec(fx[i], ni[i])
        # if fp[i] > float32(0.0):
        NN += pos_sign_vec(fp[i])
        # NN += 1 if fp[i] > 0.0 else 0

    return sqrt(NI / (NN + 0.01)), fp, fx, si, co, ni


@njit(cache=True)
def calculate_dpf(FP, FX, rms, n_sky: int, gamma_regulator: float, network_energy_threshold: float):
    FP = FP.astype(np.float32)
    FX = FX.astype(np.float32)
    rms = rms.astype(np.float32)

    MM = np.zeros(n_sky, dtype=np.uint8)
    mm = np.zeros(n_sky, dtype=np.uint8)

    for i in range(n_sky):
        # todo:          if(!mm[l]) continue;           // skip delay configurations
        # if(bBB && !BB[l]) continue;                    // skip delay configurations : big clusters
        MM[i] = 1
        # FF += 1
        aa, fp, fx, si, co, ni = dpf_np_loops_vec(FP[i], FX[i], rms)

        mm[i] = 1 if aa > gamma_regulator else 0
    FF = MM.sum()
    ff = mm.sum()

    return (FF ** 2 / (ff ** 2 + 1.e-9) - 1) * network_energy_threshold


calculate_dpf(FP.T, FX.T, rms.T,
              int(n_sky),
              gamma_regulator, network_energy_threshold)

print(dpf_np_loops_vec(FP.T[1000].astype(np.float32), FX.T[1000].astype(np.float32), rms.T.astype(np.float32))[0])

import time

total_time = 0
# # convert FP, FX, rms to float32
for i in range(10):
    start = time.time()
    calculate_dpf(FP.T, FX.T, rms.T,
                  int(n_sky),
                  gamma_regulator, network_energy_threshold)
    end = time.time()
    total_time += end - start
    print(end - start)
print(f"Average time: {total_time / 10:.4f} s")

from subprocess import Popen
from contextlib import contextmanager
from os import getpid
from time import sleep
from signal import SIGINT

# @contextmanager
# def perf_stat():
#     p = Popen(["perf", "stat", "-p", str(getpid())])
#     sleep(0.5)
#     yield
#     p.send_signal(SIGINT)
#
#
# with perf_stat():
#     calculate_dpf(FP.T, FX.T, rms.T,
#                   int(n_sky),
#                   gamma_regulator, network_energy_threshold)
