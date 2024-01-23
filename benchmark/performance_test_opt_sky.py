# add .. to path[0]
import sys, os
sys.path.insert(0, '..')

import pickle
from math import sqrt
import numpy as np
from pycwb.modules.likelihoodWP.likelihood import calculate_dpf, find_optimal_sky_localization, calculate_sky_statistics
from pycwb.modules.likelihoodWP.likelihood import load_data_from_pixels
from pycwb.modules.xtalk.monster import load_catalog, getXTalk_pixels
import time

#################################################################################
# load FP, FX, rms, n_sky, gamma_regulator, network_energy_threshold from pickle
#################################################################################
with open('test_data.pkl', 'rb') as f:
    test_data = pickle.load(f)

FP = test_data['FP']
FX = test_data['FX']
ml = test_data['ml']
n_sky = test_data['n_sky']
pixels = test_data['pixels']
gamma_regulator = test_data['gamma_regulator']
delta_regulator = test_data['delta_regulator']
network_energy_threshold = test_data['network_energy_threshold']
netCC = test_data['netCC']
n_ifo = test_data['n_ifo']
REG = np.array([delta_regulator * sqrt(2), 0., 0.])

rms, td00, td90, td_energy = load_data_from_pixels(pixels, n_ifo)

n_pix = rms.shape[1]
#####################
# Load xtalk catalog
#####################
if not os.environ.get('HOME_WAT_FILTERS'):
    print('Please set HOME_WAT_FILTERS to the directory of WAT filters')
    exit(1)

fn = f"{os.environ.get('HOME_WAT_FILTERS')}/wdmXTalk/OverlapCatalog16-1024.bin"

start_time = time.time()
xtalk_coeff, xtalk_lookup_table, layers, nRes = load_catalog(fn)
print(f"Time for load_catalog: {time.time() - start_time} s")

#######################
# Get xtalk for pixels
#######################
cluster_xtalk_lookup, cluster_xtalk = getXTalk_pixels(pixels, True, layers, xtalk_coeff, xtalk_lookup_table)

#############################################
# Convert FP, FX, rms, td00, td90 to float32
#############################################
td00 = np.transpose(td00.astype(np.float32), (2, 0, 1))  # (ndelay, nifo, npix)
td90 = np.transpose(td90.astype(np.float32), (2, 0, 1))  # (ndelay, nifo, npix)
FP = FP.T.astype(np.float32)
FX = FX.T.astype(np.float32)
rms = rms.T.astype(np.float32)

#############################################
# Calculate DPF and the optimal sky location
#############################################
REG[1] = calculate_dpf(FP, FX, rms, n_sky, n_ifo, gamma_regulator, network_energy_threshold)

l_max = find_optimal_sky_localization(n_ifo, n_pix, n_sky, FP, FX, rms, td00, td90, ml, REG, netCC, delta_regulator,
                                      network_energy_threshold)
print(f"l_max: {l_max}")

#############################################
# Calculate sky statistics
#############################################
calculate_sky_statistics(l_max, n_ifo, n_pix, FP, FX, rms, td00, td90, ml, REG, network_energy_threshold, cluster_xtalk,
                         cluster_xtalk_lookup)

#############################################
# Performance test
#############################################

total_time = 0
# # convert FP, FX, rms to float32
for i in range(10):
    start = time.time()
    l_max = find_optimal_sky_localization(n_ifo, n_pix, n_sky, FP, FX, rms, td00, td90, ml, REG, netCC, delta_regulator,
                                          network_energy_threshold)
    end = time.time()
    total_time += end - start
    # print(end - start)
print(f"Average time for find_optimal_sky_localization: {total_time / 10} s")

total_time = 0
for i in range(10):
    start = time.time()
    calculate_sky_statistics(l_max, n_ifo, n_pix, FP, FX, rms, td00, td90, ml, REG, network_energy_threshold,
                             cluster_xtalk, cluster_xtalk_lookup)
    end = time.time()
    total_time += end - start
    # print(end - start)
print(f"Average time for calculate_sky_statistics: {total_time / 10} s")

total_time = 0

for i in range(10):
    start = time.time()
    cluster_xtalk_lookup, cluster_xtalk = getXTalk_pixels(pixels, True, layers, xtalk_coeff, xtalk_lookup_table)
    end = time.time()
    # print(end - start)
    total_time += end - start

print(f"Average time for getXTalk_pixels: {total_time / 10} s")
