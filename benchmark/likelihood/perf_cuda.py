import cupy as cp

def dpf_cupy(FP, FX, rms):
    """
    Vectorized version of dpf_np using CuPy for GPU acceleration.
    
    :param FP: (n_sky, NIFO, 1) array
    :param FX: (n_sky, NIFO, 1) array
    :param rms: (NIFO, NPIX) array
    :return: aa, fp, fx, si, co, ni (all as cupy arrays)
    """
    # Broadcast multiply
    f = FP * rms[cp.newaxis, :, :]  # Shape: (n_sky, NIFO, NPIX)
    F = FX * rms[cp.newaxis, :, :]

    # Sum over NIFO (axis=1)
    _ff = (f ** 2).sum(axis=1)
    _FF = (F ** 2).sum(axis=1)
    _fF = (f * F).sum(axis=1)

    # Intermediate terms
    _si = 2.0 * _fF
    _co = _ff - _FF
    _AP = _ff + _FF
    _nn = cp.sqrt(_co**2 + _si**2)
    _cc = _co / (_nn + 0.0001)
    fp = (_AP + _nn) / 2.0

    # Compute co and si with sign correction
    si = cp.sqrt((1.0 - _cc) / 2.0)
    co = cp.where(_si > 0.0, 
                  cp.sqrt((1.0 + _cc) / 2.0), 
                  -cp.sqrt((1.0 + _cc) / 2.0))

    # Expand dimensions for broadcasting
    co_exp = co[:, cp.newaxis, :]
    si_exp = si[:, cp.newaxis, :]

    # Update basis vectors
    f_new = f * co_exp + F * si_exp
    F_new = F * co_exp - f * si_exp

    # Compute new correlation
    fF_new = (f_new * F_new).sum(axis=1) / (fp + 0.0001)

    # Adjust F
    F_adj = F_new - f_new * fF_new[:, cp.newaxis, :]

    # Final energy terms
    fx = (F_adj ** 2).sum(axis=1)
    ni = (f_new ** 4).sum(axis=1) / (fp**2 + 0.0001)

    # Network statistics
    NI = (fx / (ni + 0.0001)).sum(axis=1)
    NN = (fp > 0.0).sum(axis=1)
    aa = cp.sqrt(NI / (NN + 0.01))

    return aa, fp, fx, si, co, ni


def dpf_cupy(Fp0, Fx0, rms):
    """
    :param Fp0: Fp0 is a NIFO x 1 array (CuPy array)
    :param Fx0: Fx0 is a NIFO x 1 array (CuPy array)
    :param rms: rms is a NIFO x NPIX array (CuPy array)
    :return: Tuple of results as CuPy arrays
    """
    f = rms * Fp0
    F = rms * Fx0

    _ff = cp.sum(f * f, axis=1)
    _FF = cp.sum(F * F, axis=1)
    _fF = cp.sum(f * F, axis=1)

    _si = 2.0 * _fF
    _co = _ff - _FF
    _AP = _ff + _FF
    _nn = cp.sqrt(_co**2 + _si**2)
    _cc = _co / (_nn + 0.0001)
    fp = (_AP + _nn) / 2.0

    si = cp.sqrt((1.0 - _cc) / 2.0)
    co = cp.where(_si > 0.0, 
                 cp.sqrt((1.0 + _cc) / 2.0), 
                 -cp.sqrt((1.0 + _cc) / 2.0))

    # Update f and F with broadcasting
    f = f * co[:, cp.newaxis] + F * si[:, cp.newaxis]
    F = F * co[:, cp.newaxis] - f * si[:, cp.newaxis]

    fF_new = cp.sum(f * F, axis=1) / (fp + 0.0001)
    F -= f * fF_new[:, cp.newaxis]

    fx = cp.sum(F * F, axis=1)
    ni = cp.sum(f**4, axis=1) / (fp**2 + 0.0001)

    NI = cp.sum(fx / (ni + 0.0001))
    NN = cp.sum(fp > 0.0)

    return cp.sqrt(NI / (NN + 0.01)), fp, fx, si, co, ni
            

def calculate_dpf_gpu(FP, FX, rms, n_sky: int, gamma_regulator: float, 
                     network_energy_threshold: float):
    """
    GPU-accelerated version of calculate_dpf using CuPy.
    
    :param FP: (n_sky, NIFO, 1) array
    :param FX: (n_sky, NIFO, 1) array
    :param rms: (NIFO, NPIX) array
    :return: network energy statistic
    """
    # Convert to float32 and ensure on GPU
    FP = cp.asarray(FP, dtype=cp.float32)
    FX = cp.asarray(FX, dtype=cp.float32)
    rms = cp.asarray(rms, dtype=cp.float32)

    # Compute all aa values in parallel
    aa, _, _, _, _, _ = dpf_cupy(FP, FX, rms)

    # Compute selection mask
    mm = (aa > gamma_regulator).astype(cp.uint8)
    ff = mm.sum()
    FF = n_sky  # Original code sets all MM[i] = 1

    # Final calculation
    return (FF**2 / (ff**2 + 1e-9) - 1) * network_energy_threshold



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


calculate_dpf_gpu(FP.T, FX.T, rms.T,
              int(n_sky),
              gamma_regulator, network_energy_threshold)

print(dpf_cupy(FP.T[1000].astype(np.float32), FX.T[1000].astype(np.float32), rms.T.astype(np.float32))[0])

import time

total_time = 0
# # convert FP, FX, rms to float32
for i in range(10):
    start = time.time()
    calculate_dpf_gpu(FP.T, FX.T, rms.T,
                  int(n_sky),
                  gamma_regulator, network_energy_threshold)
    end = time.time()
    total_time += end - start
    print(end - start)
print(f"Average time: {total_time / 10:.4f} s")