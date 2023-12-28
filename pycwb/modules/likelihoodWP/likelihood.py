from math import sqrt

import numpy as np
from numba import njit, prange

from pycwb.modules.cwb_conversions import convert_wavearray_to_nparray
from .dpf import calculate_dpf

def likelihood(network, nIFO, cluster):
    acor = network.net.acor
    network_energy_threshold = 2 * acor * acor * nIFO
    gamma_regulator = network.net.gamma * network.net.gamma * 2 / 3
    delta_regulator = abs(network.net.delta) if abs(network.net.delta) < 1 else 1
    REG = [delta_regulator * np.sqrt(2), 0, 0]
    netEC_threshold = network.net.netRHO * network.net.netRHO * 2

    n_sky = network.net.index.size()

    ml, FP, FX = load_data_from_ifo(network, nIFO)

    rms, td00, td90, td_energy = load_data_from_pixels(cluster.pixels, nIFO)

    REG[1] = calculate_dpf(n_sky, gamma_regulator, network_energy_threshold)

    l_max = find_optimal_sky_localization()

    calculate_sky_statistics(l_max)

    calculate_detection_statistic()

    threshold_cut()
    likelihood_by_pixel()
    subnetwork_statistic()
    detection_statistic()
    get_error_region()
    get_chip_mass()


def load_data_from_pixels(pixels, nifo):
    tsize = len(pixels[0].td_amp[0])

    rms = np.zeros((nifo, len(pixels)))
    td00 = np.zeros((nifo, len(pixels), int(tsize / 2)))
    td90 = np.zeros((nifo, len(pixels), int(tsize / 2)))
    td_energy = np.zeros((nifo, len(pixels), int(tsize / 2)))
    for pid, pix in enumerate(pixels):
        rms_pix = 0
        rms_array = np.zeros(nifo)

        for i in range(nifo):
            xx = 1. / pix.data[i].noise_rms
            rms_pix += xx * xx
            rms_array[i] = xx

        rms_pix = np.sqrt(1. / rms_pix)

        for i in range(nifo):
            rms[i, pid] = rms_array[i] * rms_pix
            td00[i, pid] = pix.td_amp[i][0:int(tsize / 2)]
            td90[i, pid] = pix.td_amp[i][int(tsize / 2):tsize]
            td_energy[i, pid] = td00[i, pid] ** 2 + td90[i, pid] ** 2
    return rms, td00, td90, td_energy


def load_data_from_ifo(network, nIFO):
    ml = []
    FP = []
    FX = []
    for i in range(nIFO):
        ml.append(convert_wavearray_to_nparray(network.get_ifo(i).index, short=True))
        FP.append(convert_wavearray_to_nparray(network.get_ifo(i).fp))
        FX.append(convert_wavearray_to_nparray(network.get_ifo(i).fx))

    ml = np.array(ml)
    FP = np.array(FP)
    FX = np.array(FX)

    return ml, FP, FX


def find_optimal_sky_localization(L):
    l_max = 0
    for l in range(L):
        calculate_sky_statistics(l)

    return l_max


def calculate_sky_statistics(l):
    pass


def calculate_detection_statistic():
    pass


def threshold_cut():
    pass


def likelihood_by_pixel():
    pass


def subnetwork_statistic():
    pass


def detection_statistic():
    pass


def get_error_region():
    # pwc->p_Ind[id - 1].push_back(Mo);
    # double T = To + pwc->start;                          // trigger time
    # std::vector<float> sArea;
    # pwc->sArea.push_back(sArea);
    # pwc->p_Map.push_back(sArea);
    #
    # double var = norm * Rc * sqrt(Mo) * (1 + fabs(1 - CH));
    #
    # // TODO: fix this
    # if (iID <= 0 || ID == id) {
    # network::getSkyArea(id, lag, T, var);       // calculate error regions
    # }
    pass


def get_chip_mass():
    # if (netRHO >= 0) {
    # ee = pwc->mchirp(id);        // original mchirp 2G
    # cc = Ec / (fabs(Ec) + ee);            // chirp cc
    # printf("mchirp_2g : %d %g %.2e %.3f %.3f %.3f %.3f \n\n",
    # int(id), cc, pwc->cData[id - 1].mchirp,
    # pwc->cData[id - 1].mchirperr, pwc->cData[id - 1].tmrgr,
    # pwc->cData[id - 1].tmrgrerr, pwc->cData[id - 1].chi2chirp);
    # } else {                // Enabled only for Search=CBC/BBH/IMBHB
    # if (m_chirp && (TString(Search) == "CBC" || TString(Search) == "BBH" || TString(Search) == "IMBHB")) {
    # ee = pwc->mchirp_upix(id, nRun);        // mchirp micropixel version
    # }
    # }
    pass
