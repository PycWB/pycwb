# add .. to path[0]
import sys, os
sys.path.insert(0, '../..')

import pickle
from math import sqrt
import numpy as np
from pycwb.modules.likelihoodWP.dpf import compute_dpf_regulator
from pycwb.modules.likelihoodWP.sky_scan import scan_sky
from pycwb.modules.likelihoodWP.sky_statistics import compute_statistics_at_sky_position
from pycwb.modules.likelihoodWP.pixel_data import extract_pixel_time_delay_data
from pycwb.modules.likelihoodWP.detection_statistics import get_likelihood_rejection_reason, populate_detection_statistics
from pycwb.modules.xtalk.monster import load_catalog, getXTalk_pixels
from pycwb.modules.xtalk.type import XTalk
from pycwb.modules.likelihoodWP.results import SkyMapStatistics
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
cluster = test_data['cluster']
pixels = test_data['pixels']
gamma_regulator = test_data['gamma_regulator']
delta_regulator = test_data['delta_regulator']
network_energy_threshold = test_data['network_energy_threshold']
netCC = test_data['netCC']
n_ifo = test_data['n_ifo']
netEC_threshold = test_data['netEC_threshold']
REG = np.array([delta_regulator * sqrt(2), 0., 0.])

rms, td00, td90, td_energy = extract_pixel_time_delay_data(pixels, n_ifo)

n_pix = rms.shape[1]
#####################
# Load xtalk catalog
#####################
fn = f"./wdmXTalk/OverlapCatalog16-1024.bin"

start_time = time.time()
# xtalk_coeff, xtalk_lookup_table, layers, nRes = load_catalog(fn)
xtalk = XTalk.load(fn)
print(f"Time for load_catalog: {time.time() - start_time} s")

#######################
# Get xtalk for pixels
#######################
cluster_xtalk_lookup, cluster_xtalk = xtalk.get_xtalk_pixels(pixels, True)

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
sky_valid_indices = np.arange(n_sky, dtype=np.int64)
REG[1] = compute_dpf_regulator(FP, FX, rms, n_sky, n_ifo, gamma_regulator, network_energy_threshold, sky_valid_indices)

skymap_statistics = scan_sky((FP, FX, ml), (rms, td00, td90),
        (REG, netCC, delta_regulator, network_energy_threshold, sky_valid_indices))
skymap_statistics = SkyMapStatistics.from_tuple(skymap_statistics)

#############################################
# Calculate sky statistics
#############################################
sky_statistics = compute_statistics_at_sky_position(skymap_statistics.l_max, n_ifo, n_pix, FP, FX, rms, td00, td90, ml, REG, network_energy_threshold, cluster_xtalk,
                         cluster_xtalk_lookup, DEBUG=True)

rejected = get_likelihood_rejection_reason(sky_statistics, network_energy_threshold, netEC_threshold)
if rejected:
    print(f"Cluster rejected due to threshold cuts: {rejected}")


populate_detection_statistics(sky_statistics, skymap_statistics, cluster=cluster,
                            n_ifo=n_ifo, xtalk=xtalk,
                            network_energy_threshold=network_energy_threshold)
# Eo = 8255.51, Lo = 7805.59, Ep = 1552.52, Lp = 1467.77
# Gn = 83.9425, Ec = 7571.68, Dc = 0.441406, Rc = 0.999999, Eh = 9.45466
print(f"expected Eo: 8255.51, Lo: 7805.59, Ep: 1552.52, Lp: 1467.77")
print(f"expected Gn: 83.9425, Ec: 7571.68, Dc: 0.441406, Rc: 0.999999, Eh: 9.45466, Es: 0, NC: 369, NS: 368")
print(f"expected N = 83.0302, N2 = 83.9448")
print(f"expected Np = 70.9542, Em = 1676.79, Lm = 1521.89, norm = 4.91776, Ec = 1539.66, Dc = 0.0897575, ch = 0.932774")
print(f"expected cc = 1, rho = 27.7458, ecor = 1.4013e-45, penalty = 2.92971e-33, xrho = 4.2039e-45")

print("Expected Eo = 8469.68, Lo = 7973.51, Ep = 1620.1, Lp = 1524.92")
print("Expected Gn = 92.2829, Ec = 7705.1, Dc = 6.82031, Rc = 0.999999, Eh = 33.0292 , ES = 0 , NC = 422 , NS = 417")
print("Expected N = 94.2876, N2 = 92.2847")
print("Expected Np = 109.1, Em = 1741.43, Lm = 1540.04, norm = 4.84468, Ec = 1590.42, Dc = 1.40779, ch = 1.06792")
print("Expected cc = 1.06792, rho = 28.1995, ecor = 4.30479e-42, penalty = 1328.17, xrho = 0")
print("Expected Mw = 422, M = 417, Esub = 1967.29, Nmax = 107.095")
print("Expected sky size: 422, sub_net: 0.948373, sub_net2: 0.933051, like_sky: 1741.43, energy_sky: 8469.68, net_ecor: 1590.42, norm_cor: 1590.42, g_net: 0.717639, a_net: 0.21022, i_net: 0, ndof: 94.2876, sky_chi2: 1.06188, g_noise: 92.2829")
#############################################
# Performance test
#############################################
exit()
total_time = 0
# # convert FP, FX, rms to float32
for i in range(10):
    start = time.time()
    l_max = scan_sky((FP, FX, ml), (rms, td00, td90),
        (REG, netCC, delta_regulator, network_energy_threshold, sky_valid_indices))
    end = time.time()
    total_time += end - start
    # print(end - start)
print(f"Average time for find_optimal_sky_localization: {total_time / 10} s")

total_time = 0
for i in range(10):
    start = time.time()
    compute_statistics_at_sky_position(l_max, n_ifo, n_pix, FP, FX, rms, td00, td90, ml, REG, network_energy_threshold,
                             cluster_xtalk, cluster_xtalk_lookup)
    end = time.time()
    total_time += end - start
    # print(end - start)
print(f"Average time for compute_statistics_at_sky_position: {total_time / 10} s")

total_time = 0

for i in range(10):
    start = time.time()
    cluster_xtalk_lookup, cluster_xtalk = getXTalk_pixels(pixels, True, layers, xtalk_coeff, xtalk_lookup_table)
    end = time.time()
    # print(end - start)
    total_time += end - start

print(f"Average time for getXTalk_pixels: {total_time / 10} s")
