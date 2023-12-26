from math import sqrt

import numpy as np
from numba import njit, prange

from pycwb.modules.cwb_conversions import convert_wavearray_to_nparray


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

    sky_statistic_by_pixel()
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


@njit(parallel=True)
def calculate_dpf(FP, FX, rms, n_sky, gamma_regulator, network_energy_threshold):
    MM = np.zeros(n_sky)
    FF = 0
    ff = 0
    for i in prange(n_sky):
        # todo:          if(!mm[l]) continue;           // skip delay configurations
        # if(bBB && !BB[l]) continue;                    // skip delay configurations : big clusters
        MM[i] = 1
        FF += 1
        aa, fp, fx, si, co, ni = avx_dpf_ps(FP, FX, rms, i)

        if aa > gamma_regulator:
            ff += 1
    return (FF ** 2 / (ff ** 2 + 1.e-9) - 1) * network_energy_threshold


@njit
def avx_dpf_ps(FP, FX, rms, sky_id):
    Fp0 = FP[:, sky_id]
    Fx0 = FX[:, sky_id]
    # sign = np.sign(np.dot(Fp0, Fx0))

    n_pix = len(rms[0])
    # n_ifo = len(FP)

    NI = 0
    NN = 0

    fp_array = np.zeros(n_pix)
    fx_array = np.zeros(n_pix)
    si_array = np.zeros(n_pix)
    co_array = np.zeros(n_pix)
    ni_array = np.zeros(n_pix)

    rms = rms.T
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


def sky_statistic_by_pixel():
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
