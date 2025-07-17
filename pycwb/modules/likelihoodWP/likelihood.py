from math import sqrt
import numpy as np
from numba import njit, prange, float32
from numba.typed import List
from pycwb.modules.cwb_conversions import convert_wavearray_to_nparray
from pycwb.types.network_cluster import Cluster
from .dpf import calculate_dpf, dpf_np_loops_vec
from .sky_stat import avx_GW_ps, avx_ort_ps, avx_stat_ps, load_data_from_td
from .utils import avx_packet_ps, packet_norm_numpy, gw_norm_numpy, avx_noise_ps, \
        avx_setAMP_ps, avx_pol_ps, avx_loadNULL_ps
from ..xtalk.monster import load_catalog, getXTalk_pixels, getXTalk
from .typing import SkyStatistics

def likelihood(network, nIFO, cluster, MRAcatalog):
    # load network parameters

    acor = network.net.acor
    network_energy_threshold = 2 * acor * acor * nIFO
    gamma_regulator = network.net.gamma * network.net.gamma * 2 / 3
    delta_regulator = abs(network.net.delta) if abs(network.net.delta) < 1 else 1
    REG = np.array([delta_regulator * np.sqrt(2), 0., 0.])
    netEC_threshold = network.net.netRHO * network.net.netRHO * 2
    netCC = network.net.netCC

    n_sky = network.net.index.size()
    n_pix = len(cluster.pixels)

    # Load xtalk catalog

    catalog, layers, nRes = load_catalog(MRAcatalog)
    sizeCC, wdm_xtalk = getXTalk_pixels(cluster.pixels, True, layers, catalog)

    # Extract data from python object to numpy arrays for numba
    ml, FP, FX = load_data_from_ifo(network, nIFO)
    rms, td00, td90, td_energy = load_data_from_pixels(cluster.pixels, nIFO)

    # Transpose array and convert to float32 for speedup
    td00 = np.transpose(td00.astype(np.float32), (2, 0, 1))  # (ndelay, nifo, npix)
    td90 = np.transpose(td90.astype(np.float32), (2, 0, 1))  # (ndelay, nifo, npix)
    FP = FP.T.astype(np.float32)
    FX = FX.T.astype(np.float32)
    rms = rms.T.astype(np.float32)
    wdm_xtalk = List(wdm_xtalk)


    REG[1] = calculate_dpf(FP, FX, rms, n_sky, nIFO, gamma_regulator, network_energy_threshold)

    l_max = find_optimal_sky_localization(nIFO, n_pix, n_sky, FP, FX, rms, td00, td90, ml, REG, netCC,
                                          delta_regulator, network_energy_threshold)

    sky_statistics: SkyStatistics = calculate_sky_statistics(l_max, nIFO, n_pix, FP, FX, rms, td00, td90, ml, REG, network_energy_threshold,
                             wdm_xtalk)

    rejected = threshold_cut(sky_statistics, network_energy_threshold, netEC_threshold)
    if rejected:
        print(f"Cluster rejected due to threshold cuts: {rejected}")
        return None


    fill_detection_statistic(sky_statistics, cluster=cluster, n_ifo=nIFO, wdm_xtalk=wdm_xtalk, layers=layers)


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


@njit(cache=True, parallel=True)
def find_optimal_sky_localization(n_ifo, n_pix, n_sky, FP, FX, rms, td00, td90, ml, REG, netCC, delta_regulator, network_energy_threshold):
    # td00 = np.transpose(td00.astype(np.float32), (2, 0, 1))  # (ndelay, nifo, npix)
    # td90 = np.transpose(td90.astype(np.float32), (2, 0, 1))  # (ndelay, nifo, npix)
    # FP = FP.T.astype(np.float32)
    # FX = FX.T.astype(np.float32)
    # rms = rms.T.astype(np.float32)
    REG = REG.astype(np.float32)


    # get unique list of ml.T for each nsky for numba
    # ml_set = set()
    # for i in range(n_sky):
    #     ml_set.add(tuple(ml.T[i]))

    nAlignment = np.zeros(n_sky, dtype=float32)
    nLikelihood = np.zeros(n_sky, dtype=float32)
    nNullEnergy = np.zeros(n_sky, dtype=float32)
    nCorrEnergy = np.zeros(n_sky, dtype=float32)
    nCorrelation = np.zeros(n_sky, dtype=float32)
    nSkyStat = np.zeros(n_sky, dtype=float32)
    nProbability = np.zeros(n_sky, dtype=float32)
    nDisbalance = np.zeros(n_sky, dtype=float32)
    nNetIndex = np.zeros(n_sky, dtype=float32)
    nEllipticity = np.zeros(n_sky, dtype=float32)
    nPolarisation = np.zeros(n_sky, dtype=float32)
    nAntenaPrior = np.zeros(n_sky, dtype=float32)

    Eh = float32(0.0)
    # sky = 0.0
    # l_max = 0
    # STAT=-1.e12
    offset = int(td00.shape[0] / 2)
    # TODO: sky sky mask
    AA_array = np.zeros(n_sky, dtype=float32)
    for l in prange(n_sky):
        # get time delayed data slice at sky location l, make sure it is numpy float32 array
        v00 = np.empty((n_ifo, n_pix), dtype=float32)
        v90 = np.empty((n_ifo, n_pix), dtype=float32)
        for i in range(n_ifo):
            v00[i] = td00[ml[i, l] + offset, i]
            v90[i] = td90[ml[i, l] + offset, i]

        # calculate data stats for time delayed data slice
        Eo, NN, energy_total, mask = load_data_from_td(v00, v90, network_energy_threshold)
        # print Eo at 0, 1000, 2000
        # if l == 0 or l == 1000 or l == 2000:
        #     print(f"Eo({l}): ", Eo)
        #     print(f"mask[{l}]: ", np.sum(mask))
        # print(f"v00[{l}]:" , np.max(v00[0]), np.max(v00[1]), f", v90[{l}]:", np.max(v90[0]), np.max(v90[1]))
        # print(f"ml[0, {l}]: {ml[0, l]}, ml[1, {l}]: {ml[1, l]}")

        # calculate DPF f+,fx and their norms
        _, f, F, fp, fx, si, co, ni = dpf_np_loops_vec(FP[l], FX[l], rms)

        # gw strain packet, return number of selected pixels
        Mo, ps, pS, mask, au, AU, av, AV = avx_GW_ps(v00, v90, f, F, fp, fx, ni, energy_total, mask, REG)
        # print Mo at 0, 1000, 2000
        # if l == 0 or l == 1000 or l == 2000:
        #     print(f"Mo({l}): ", Mo)
        #     print(f"ps[{l}]: ", np.max(ps[0]), np.max(ps[1])), print(f"pS[{l}]: ", np.max(pS[0]), np.max(pS[1]))
        #     print(f"mask[{l}]: ", np.sum(mask))
        #     print(f"fp[{l}]: ", np.max(fp), f" fx[{l}]: ", np.max(fx))


        # othogonalize signal amplitudes
        Lo, si, co, ee, EE = avx_ort_ps(ps, pS, mask)
        # print Lo at 0, 1000, 2000
        # if l == 0 or l == 1000 or l == 2000:
        #     print(f"Lo({l}): ", Lo)

        # coherent statistics
        Cr, Ec, Mp, No, coherent_energy, _, _ = avx_stat_ps(v00, v90, ps, pS, si, co, mask)

        CH = No / (n_ifo * Mo + sqrt(Mo))  # chi2 in TF domain
        cc = CH if CH > float(1.0) else 1.0  # noise correction factor in TF domain
        Co = Ec / (Ec + No * cc - Mo * (n_ifo - 1))  # network correlation coefficient in TF

        if Cr < netCC:
            continue

        aa = Eo - No if Eo > float32(0.) else float32(0.)  # likelihood skystat
        AA = aa * Co  # x-correlation skystat
        # if l == 0 or l == 1000 or l == 2000:
        #     print(f"Cr({l}): ", Cr, f" Ec[{l}]: ", Ec, f" Mp[{l}]: ", Mp, f" No[{l}]: ", No)
        #     print(f"CH({l}): ", CH, f" cc[{l}]: ", cc, f" Co[{l}]: ", Co)
        #     print(f"aa({l}): ", aa, f" AA[{l}]: ", AA)
        nProbability[l] = aa if delta_regulator < 0 else AA

        ff, FF, ee = float32(0.), float32(0.), float32(0.)

        for j in range(n_pix):
            if mask[j] <= 0:
                continue
            ee += energy_total[j]  # total energy
            ff += fp[j] * energy_total[j]  # |f+|^2
            FF += fx[j] * energy_total[j]  # |fx|^2
        ff = ff / ee if ee > float32(0.) else float32(0.)
        FF = FF / ee if ee > float32(0.) else float32(0.)

        nAntenaPrior[l] = sqrt(ff + FF)
        nAlignment[l] = sqrt(FF / ff) if ff > float32(0.) else float32(0.)
        nLikelihood[l] = Eo - No
        nNullEnergy[l] = No
        nCorrEnergy[l] = Ec
        nCorrelation[l] = Co
        nSkyStat[l] = AA
        nDisbalance[l] = CH
        nNetIndex[l] = cc
        nEllipticity[l] = Cr
        nPolarisation[l] = Mp

        AA_array[l] = AA
        # if AA >= STAT:
        #     STAT = AA
        #     l_max = l
        #     Em = Eo - Eh
        #
        # if nProbability[l] > sky:
        #     sky = nProbability[l]  # find max of skyloc stat
    STAT = np.max(AA_array)
    l_max = np.argmax(AA_array)
    sky = np.max(nProbability)

    return l_max


# @njit(cache=True)
def calculate_sky_statistics(l, n_ifo, n_pix, FP, FX, rms, td00, td90, ml, REG, 
                             network_energy_threshold, cluster_xtalk, cluster_xtalk_lookup_table,
                             DEBUG=False) -> SkyStatistics:
    """

    """
    # from numpy import float32
    # td00 = np.transpose(td00.astype(np.float32), (2, 0, 1))  # (ndelay, nifo, npix)
    # td90 = np.transpose(td90.astype(np.float32), (2, 0, 1))  # (ndelay, nifo, npix)
    v00 = np.empty((n_ifo, n_pix), dtype=np.float32)
    v90 = np.empty((n_ifo, n_pix), dtype=np.float32)
    td_energy = np.zeros((n_ifo, n_pix), dtype=np.float32)

    offset = int(td00.shape[0] / 2)

    # get time delayed data slice at sky location l
    for i in range(n_ifo):
        v00[i] = td00[ml[i, l] + offset, i]
        v90[i] = td90[ml[i, l] + offset, i]

    # compute the energy of the time delayed data slice
    for i in range(n_ifo):
        for j in range(n_pix):
            td_energy[i, j] = v00[i, j] * v00[i, j] + v90[i, j] * v90[i, j]

    # calculate the total energy, active pixels and mask
    Eo, NN, energy_total, mask = load_data_from_td(v00, v90, network_energy_threshold)

    # calculate DPF f+,fx and their norms for the sky location l
    _, f, F, fp, fx, si, co, ni = dpf_np_loops_vec(FP[l], FX[l], rms)

    # gw strain packet, return number of selected pixels
    Mo, ps, pS, mask, au, AU, av, AV = avx_GW_ps(v00, v90, f, F, fp, fx, ni, energy_total, mask, REG)

    # othogonalize signal amplitudes
    Lo, si, co, ee, EE = avx_ort_ps(ps, pS, mask)

    # coherent statistics
    _, _, _, _, coherent_energy, gn, rn = avx_stat_ps(v00, v90, ps, pS, si, co, mask)

    ##############################
    # Eo = _avx_packet_ps(pd, pD, _AVX, V4);            // get data packet
    # Lo = _avx_packet_ps(ps, pS, _AVX, V4);            // get signal packet
    # D_snr = _avx_norm_ps(wdmMRA, pd, pD, _AVX, V4);           // data packet energy snr
    # S_snr = _avx_norm_ps(pS, pD, p_ec, V4);           // set signal norms, return signal SNR
    # Ep = D_snr[0];
    # Lp = S_snr[0];
    ##############################
    Eo, pd, pD, pD_E, pD_si, pD_co, pD_a, pD_A = avx_packet_ps(v00, v90, mask)  # get data packet
    Lo, ps, pS, pS_E, pS_si, pS_co, pS_a, pS_A = avx_packet_ps(ps, pS, mask)  # get signal packet

    detector_snr, pD_E, rn, pD_norm = packet_norm_numpy(pd, pD, cluster_xtalk, cluster_xtalk_lookup_table, mask, pD_E)
    D_snr = np.sum(detector_snr)
    S_snr, signal_snr, pS_E, pS_norm = gw_norm_numpy(pD_norm, pD_E, pS_E, coherent_energy) # return signal norms and signal SNR
    # S_snr = np.sum(signal_norm)
    if DEBUG:
        print(S_snr, signal_snr)
        print("Eo = ", Eo, ", Lo = ", Lo, ", Ep = ", D_snr, ", Lp = ", S_snr)

    ############### G-noise correction ##################
    # _CC = _avx_noise_ps(pS, pD, _AVX, V4);            // get G-noise correction
    # _mm256_storeu_ps(vvv, _CC);                     // extract coherent statistics
    # Gn = vvv[0];                                   // gaussian noise correction
    # Ec = vvv[1];                                   // core coherent energy in TF domain
    # Dc = vvv[2];                                   // signal-core coherent energy in TF domain
    # Rc = vvv[3];                                   // EC normalization
    # Eh = vvv[4];                                   // satellite energy in TF domain
    #################################

    # TODO: one more pixel selected, need to be fixed
    # Gn: gaussian noise correction
    # Ec: core coherent energy in TF domain
    # Dc: signal-core coherent energy in TF domain
    # Rc: EC normalization
    # Eh: satellite energy in TF domain    
    Gn, Ec, Dc, Rc, Eh, Es, NC, NS = avx_noise_ps(pS_norm, pD_norm, energy_total, mask, coherent_energy, gn, rn)

    if DEBUG:
        print("Gn = ", Gn, ", Ec = ", Ec, ", Dc = ", Dc, ", Rc = ", Rc, ", Eh = ", Eh, ", Es = ", Es, ", NC = ", NC, ", NS = ", NS)
    # # DEBUG:
    # print(f"pD[0][393]: {pD[0][393]}, pS[0][393]: {pS[0][393]}")
    # print(f"pD[1][393]: {pD[1][393]}, pS[1][393]: {pS[1][393]}")

    ##################################
    # N = _avx_setAMP_ps(pd, pD, _AVX, V4) - 1;           // set data packet amplitudes
    # _avx_setAMP_ps(ps, pS, _AVX, V4);                 // set signal packet amplitudes
    # _avx_loadNULL_ps(pn, pN, pd, pD, ps, pS, V4);        // load noise TF domain amplitudes
    # D_snr = _avx_norm_ps(wdmMRA, pd, pD, _AVX, -V4);          // data packet energy snr
    # N_snr = _avx_norm_ps(wdmMRA, pn, pN, _AVX, -V4);          // noise packet energy snr
    # Np = N_snr.data[0];                            // time-domain NULL
    # Em = D_snr.data[0];                            // time domain energy
    # Lm = Em - Np - Gn;                                 // time domain signal energy
    # norm = Em > 0 ? (Eo - Eh) / Em : 1.e9;               // norm
    # if (norm < 1) norm = 1;                           // corrected norm
    # Ec /= norm;                                   // core coherent energy in time domain
    # Dc /= norm;                                   // signal-core coherent energy in time domain
    # ch = (Np + Gn) / (N * nIFO);                         // chi2
    ##################################

    # set data packet amplitudes
    N, pd, pD = avx_setAMP_ps(pd, pD, pD_norm, pD_si, pD_co, pD_a, pD_A, mask)  # set data packet amplitudes
    N = N - 1  # effective number of pixels
    # set signal packet amplitudes
    _, ps, pS = avx_setAMP_ps(ps, pS, pS_norm, pS_si, pS_co, pS_a, pS_A, mask)
    # load noise TF domain amplitudes
    pn, pN = avx_loadNULL_ps(pd, pD, ps, pS)
    # data packet energy snr
    _, pD_E, rn, _ = packet_norm_numpy(pd, pD, cluster_xtalk, cluster_xtalk_lookup_table, mask, pD_E)
    D_snr = np.sum(pD_E)  # data packet energy snr
    _, pN_E, rn, _ = packet_norm_numpy(pn, pN, cluster_xtalk, cluster_xtalk_lookup_table, mask, pS_E)
    N_snr = np.sum(pN_E)  # noise packet energy snr

    Np = N_snr  # time-domain NULL
    Em = D_snr  # time domain energy
    Lm = Em - Np - Gn  # time domain signal energy
    norm = (Eo - Eh) / Em if (Eo - Eh) > 0 else 1.e9  # norm
    if norm < 1:
        norm = 1
    Ec /= norm  # core coherent energy in time domain
    Dc /= norm  # signal-core coherent energy in time domain
    ch = (Np + Gn) / (N * n_ifo)  # chi
    if DEBUG:
        print("Np = ", Np, ", Em = ", Em, ", Lm = ", Lm, ", norm = ", norm, ", Ec = ", Ec, ", Dc = ", Dc, ", ch = ", ch)

    #################################
    # if (netRHO >= 0) {    // original 2G
    #     cc = ch > 1 ? ch : 1;                          // rho correction factor
    #     rho = Ec > 0 ? sqrt(Ec * Rc / 2.) : 0.;           // cWB detection stat
    # } else {        // (XGB.rho0)
    #     penalty = ch;
    #     ecor = Ec;
    #     rho = sqrt(ecor / (1 + penalty * (max((float) 1., penalty) - 1)));
    #     // original 2G rho statistic: only for test
    #     cc = ch > 1 ? ch : 1;                          // rho correction factor
    #     xrho = Ec > 0 ? sqrt(Ec * Rc / 2.) : 0.;          // cWB detection stat
    # }
    #################################
    xrho = 0.
    penalty = 0.
    ecor = 0.
    if network_energy_threshold >= 0: # original 2G
        cc = ch if ch > 1 else 1  # rho correction factor
        rho = np.sqrt(Ec * Rc / 2.) if Ec > 0 else 0  # cWB detection stat
        if DEBUG:
            print("cc = ", cc, ", rho = ", rho)
    else:  # (XGB.rho0)
        penalty = ch
        ecor = Ec
        rho = np.sqrt(ecor / (1 + penalty * (max(float(1), penalty) - 1)))
        # original 2G rho statistic: only for test
        cc = ch if ch > 1 else 1  # rho correction factor
        xrho = np.sqrt(Ec * Rc / 2.) if Ec > 0 else 0  # cWB detection stat
        if DEBUG:
            print("cc = ", cc, ", rho = ", rho, ", ecor = ", ecor, ", penalty = ", penalty, ", xrho = ", xrho)
    #################################
    # // save projection on network plane in polar coordinates
    # // The Dual Stream Transform (DSP) is applied to v00,v90
    # _avx_pol_ps(v00, v90, p00_POL, p90_POL, _APN, _AVX, V4);
    # // save DSP components in polar coordinates
    # _avx_pol_ps(v00, v90, r00_POL, r90_POL, _APN, _AVX, V4);
    #################################
    # save projection on network plane in polar coordinates
    v00, v90, p00_POL, p90_POL = avx_pol_ps(v00, v90, mask, fp, fx, f, F)
    # save DSP components in polar coordinates
    v00, v90, r00_POL, r90_POL = avx_pol_ps(v00, v90, mask, fp, fx, f, F)

    return SkyStatistics(
        Gn=np.float32(Gn),
        Ec=np.float32(Ec),
        Dc=np.float32(Dc),
        Rc=np.float32(Rc),
        Eh=np.float32(Eh),
        Es=np.float32(Es),
        Np=np.float32(Np),
        Lm=np.float32(Lm),
        norm=np.float32(norm),
        cc=np.float32(cc),
        rho=np.float32(rho),
        xrho=np.float32(xrho),
        Lo=np.float32(Lo),
        Eo=np.float32(Eo),
        energy_array_plus=ee,
        energy_array_cross=EE,
        pixel_mask=mask,
        v00=v00,
        v90=v90,
        gaussian_noise_correction=gn,
        noise_amplitude_00=pn,
        noise_amplitude_90=pN,
        pd=pd,
        pD=pD,
        ps=ps,
        pS=pS,
        p00_POL=p00_POL,
        p90_POL=p90_POL,
        r00_POL=r00_POL,
        r90_POL=r90_POL
    )

def fill_detection_statistic(sky_statistics: SkyStatistics, cluster: Cluster, n_ifo: int, wdm_xtalk: List, layers: List):
    pixel_mask = sky_statistics.pixel_mask
    energy_array_plus = sky_statistics.energy_array_plus
    energy_array_cross = sky_statistics.energy_array_cross
    pd = sky_statistics.pd
    pD = sky_statistics.pD
    ps = sky_statistics.ps
    pS = sky_statistics.pS
    gaussian_noise_correction = sky_statistics.gaussian_noise_correction
    noise_amplitude_00 = sky_statistics.noise_amplitude_00
    noise_amplitude_90 = sky_statistics.noise_amplitude_90
    n_pix = len(cluster.pixels)
    event_size = 0 # defined as Mw in cwb
    cluster_xtalk, cluster_xtalk_lookup_table = wdm_xtalk

    for i, pixel in enumerate(cluster.pixels):
        pixel.core = False
        pixel.likelihood = 0.0
        pixel.null = 0.0

        if pixel_mask[i] > 0:
            pixel.core = True
            pixel.likelihood = - (energy_array_plus[i] + energy_array_cross[i]) / 2.0

        for j in range(n_ifo):
            pixel.data[j].wave = pd[j][i]
            pixel.data[j].w_90 = pD[j][i]
            pixel.data[j].asnr = ps[j][i]
            pixel.data[j].a_90 = pS[j][i]

        if not pixel.core:
            continue
        if gaussian_noise_correction[i] <= 0:
            continue    # skip satellites
            
        event_size += 1

        for k, xpix in enumerate(cluster.pixels):
            if not xpix.core or not gaussian_noise_correction[k] <= 0:
                continue
            xt = getXTalk(pixel.layers, pixel.time, xpix.layers, xpix.time, layers, cluster_xtalk, cluster_xtalk_lookup_table)
            if xt[0] > 2:
                continue

            for j in range(n_ifo):
                pixel.null += xt[0] * noise_amplitude_00[j][i] * noise_amplitude_00[j][k] 
                pixel.null += xt[1] * noise_amplitude_00[j][i] * noise_amplitude_90[j][k]
                pixel.null += xt[2] * noise_amplitude_90[j][i] * noise_amplitude_00[j][k]
                pixel.null += xt[3] * noise_amplitude_90[j][i] * noise_amplitude_90[j][k]



    


def threshold_cut(sky_statistics: SkyStatistics, network_energy_threshold: float, netEC_threshold: float) -> str:
    """
    Apply threshold cuts based on the sky statistics and network energy threshold.
    
    Parameters:
    sky_statistics (SkyStatistics): The statistics calculated for the sky location.
    network_energy_threshold (float): The threshold for network energy.
    netEC_threshold (float): The threshold for net EC.
    
    Returns:
    str: A rejection reason if any condition is not met, otherwise None.
    """
    # if (this->netRHO >= 0)
    #   { // original 2G
    #     if (Lm <= 0. || (Eo - Eh) <= 0. || Ec * Rc / cc < netEC || N < 1)
    #     {
    #       pwc->sCuts[id - 1] = 1;
    #       count = 0; // reject cluster
    #       pwc->clean(id);
    #       continue;
    #     }
    #   }
    #   else
    #   { // (XGB.rho0)
    #     if (Lm <= 0. || (Eo - Eh) <= 0. || rho < fabs(this->netRHO) || N < 1)
    #     {
    #       pwc->sCuts[id - 1] = 1;
    #       count = 0; // reject cluster
    #       pwc->clean(id);
    #       continue;
    #     }
    #   }
    Lm = sky_statistics.Lm
    Eo = sky_statistics.Eo
    Np = sky_statistics.Np
    Eh = sky_statistics.Eh
    Ec = sky_statistics.Ec
    Rc = sky_statistics.Rc
    cc = sky_statistics.cc
    rho = sky_statistics.rho
    if network_energy_threshold > 0:
        condition_1 = Lm <= 0.
        condition_2 = (Eo - Eh) <= 0.
        condition_3 = Ec * Rc / cc < netEC_threshold
        condition_4 = Np < 1
        if condition_1 or condition_2 or condition_3 or condition_4:
            rejection_reason = ""
            if condition_1:
                rejection_reason += f"Lm > 0 but Lm = {Lm};"
            if condition_2:
                rejection_reason += f" (Eo - Eh) > 0 but (Eo - Eh) = {Eo - Eh};"
            if condition_3:
                rejection_reason += f" Ec * Rc / cc >= netEC_threshold but Ec * Rc / cc = {Ec * Rc / cc};"
            if condition_4:
                rejection_reason += f" Np > 1 but Np = {Np};"
            return rejection_reason
    else:
        # For XGB.rho0 case
        condition_1 = Lm <= 0.
        condition_2 = (Eo - Eh) <= 0.
        condition_3 = rho < abs(network_energy_threshold)
        condition_4 = Np < 1
        if condition_1 or condition_2 or condition_3 or condition_4:
            rejection_reason = ""
            if condition_1:
                rejection_reason += f"Lm > 0 but Lm = {Lm};"
            if condition_2:
                rejection_reason += f" (Eo - Eh) > 0 but (Eo - Eh) = {Eo - Eh};"
            if condition_3:
                rejection_reason += f" rho >= abs(network_energy_threshold) but rho = {rho} < {abs(network_energy_threshold)};"
            if condition_4:
                rejection_reason += f" Np > 1 but Np = {Np};"
            return rejection_reason
        
    return None  # No rejection, all conditions passed


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
