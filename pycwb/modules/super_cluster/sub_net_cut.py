import numpy as np
# from numba import float32
from numba import njit
from numpy import float32, float64
from pycwb.modules.likelihoodWP.dpf import dpf_np_loops_vec
from pycwb.modules.xtalk.monster import getXTalk_pixels
from pycwb.modules.likelihoodWP.likelihood import load_data_from_pixels


def sub_net_cut(pixels, ml, FP, FX, acor, e2or, n_ifo, n_sky, subnet, subcut, subnorm, subrho,
                xtalk_coeff, xtalk_lookup_table, layers):
    """
    This function is used to cut the subnet from the lag and return the new lag
    """
    network_energy_threshold = np.float32(2 * acor * acor * n_ifo)
    n_pix = len(pixels)

    rms, td00, td90, td_energy = load_data_from_pixels(pixels, n_ifo)

    cluster_xtalk_lookup, cluster_xtalk = getXTalk_pixels(pixels, True, layers, xtalk_coeff, xtalk_lookup_table)

    td00 = np.transpose(td00.astype(np.float32), (2, 0, 1))  # (ndelay, nifo, npix)
    td90 = np.transpose(td90.astype(np.float32), (2, 0, 1))  # (ndelay, nifo, npix)
    td_energy = np.transpose(td_energy.astype(np.float32), (2, 0, 1))  # (ndelay, nifo, npix)
    FP = FP.T.astype(np.float32)
    FX = FX.T.astype(np.float32)
    rms = rms.T.astype(np.float32)

    l_max, stat, Em, Am, lm, Vm, suball, EE = optimze_sky_loc(n_ifo, n_pix, n_sky, FP, FX, rms, td00, td90, td_energy,
                                                              ml, network_energy_threshold, e2or, subcut)

    submra, rHo, Eo, Lo, Ls, m = mra_statistics(n_ifo, n_pix, FP, FX, rms, td00, td90, td_energy, ml,
                                                      network_energy_threshold, e2or, subcut,
                                                      cluster_xtalk, cluster_xtalk_lookup, l_max)
    subnet_pass = min(suball, submra) > subnet
    subrho_pass = rHo > subrho
    subthr_pass = Em > subnorm * network_energy_threshold

    return {
        'subnet_passed': subnet_pass,
        'subrho_passed': subrho_pass,
        'subthr_passed': subthr_pass,
        'subnet_condition': f"min(suball = {suball:.4f}, submra = {submra:.4f}) > subnet = {subnet:.4f}",
        'subrho_condition': f"rho = {rHo:.4f} > subrho = {subrho:.4f}",
        'subthr_condition': f"Em = {Em:.4f} > subnorm = {subnorm:.4f} * network_energy_threshold = {network_energy_threshold:.4f}"
    }


@njit(cache=True)
def optimze_sky_loc(n_ifo, n_pix, n_sky, FP, FX, rms, td00, td90, td_energy, ml, network_energy_threshold, e2or,
                    subcut):
    Es = float32(2 * e2or)
    network_energy_threshold = float32(network_energy_threshold)
    offset = int(td00.shape[0] / 2)
    # print("offset: ", offset, td00.shape, ml.shape, td_energy.shape)

    rNRG = np.zeros(n_pix, dtype=float32)  # _rE
    pNRG = np.zeros(n_pix, dtype=float32)  # _pE
    # print("En = ", network_energy_threshold, ', Es = ', Es, ", n_pix = ", n_pix, ", n_sky = ", n_sky)
    l_max = 0
    stat = float32(0.0)
    Em = float32(0.0)
    Am = float32(0.0)
    lm = 0
    Vm = 0
    suball = float32(0.0)
    EE = float32(0.0)
    AA_max = float32(0.0)

    for l in range(n_sky):
        # TODO: sky sky mask
        # get time delayed data slice at sky location l, make sure it is numpy float32 array
        v_energy = np.zeros((n_ifo, n_pix), dtype=float32)  # pe
        v00 = np.zeros((n_ifo, n_pix), dtype=float32)  # pa
        v90 = np.zeros((n_ifo, n_pix), dtype=float32)  # pA

        for i in range(n_ifo):
            v_energy[i] = td_energy[ml[i, l] + offset, i]

        m = float32(0)  # pixels above threshold
        Eo = float32(0)  # total network energy
        Ls = float32(0)  # subnetwork energy
        Ln = float32(0)  # network energy above subnet threshold
        for j in range(n_pix):
            _rE = 0
            for i in range(n_ifo):  # get pixel energy
                _rE += v_energy[i, j]
            rNRG[j] = _rE  # store pixel energy
            _msk = float32(1.0) if rNRG[j] > network_energy_threshold else float32(0.0)  # E>En  0/1 mask
            m += _msk  # count pixels above threshold
            pNRG[j] = rNRG[j] * _msk  # zero sub-threshold pixels
            Eo += pNRG[j]
            for i in range(n_ifo):
                pNRG[j] = min(rNRG[j] - v_energy[i, j], pNRG[j])  # subnetwork energy
            Ls += pNRG[j]  # subnetwork energy
            _msk = float32(1.0) if pNRG[j] > Es else float32(0.0)  # subnet energy > Es 0/1 mask
            Ln += rNRG[j] * _msk  # network energy

        Eo = Eo + float32(0.01)
        m = int(2 * m + 0.01)
        aa = float32(Ls * Ln / (Eo - Ls))
        # if l in [0, 22, 1000, 1860, 1967, 2000]: print("l = ", l); print("Ln = ", Ln, ", Eo = ", Eo, ", Ls = ", Ls, ", m = ", m)
        if subcut >= 0 and (aa - m) / (aa + m + float32(1e-16)) < subcut:
            continue

        for i in range(n_ifo):
            v00[i] = td00[ml[i, l] + offset, i]
            v90[i] = td90[ml[i, l] + offset, i]

        m = 0
        Ls = Ln = Eo = float32(0.0)
        reduced_rms = np.zeros((n_pix, n_ifo), dtype=float32)
        reduced_v00 = np.zeros((n_pix, n_ifo), dtype=float32)
        reduced_v90 = np.zeros((n_pix, n_ifo), dtype=float32)
        for j in range(n_pix):
            ee = float32(0.)
            for i in range(n_ifo):
                ee += v00[i, j] * v00[i, j] + v90[i, j] * v90[i, j]
            if ee < network_energy_threshold:
                continue

            for i in range(n_ifo):
                reduced_rms[m, i] = rms[j, i]
                reduced_v00[m, i] = v00[i, j]
                reduced_v90[m, i] = v90[i, j]
            m += 1

            # dominant pixel energy
            em = float32(0.0)
            for i in range(n_ifo):
                _em = v00[i, j] * v00[i, j] + v90[i, j] * v90[i, j]
                if _em > em:
                    em = _em

            Ls += ee - em
            Eo += ee  # subnetwork energy, network energy
            if ee - em > Es:
                Ln += ee  # network energy above subnet threshold

        if Eo <= 0:
            continue

        Lo = 0
        # calculate dpf
        # TODO: check if the dpf is the same as the one in the likelihood module
        _, f, F, _, _, _, _, _ = dpf_np_loops_vec(FP[l], FX[l], reduced_rms[:m, :])

        for j in range(m):
            # calculate likelihood
            Lo += sse_like_ps(f[j], F[j], reduced_v00[j], reduced_v90[j])
        # if l in [0, 22, 1000, 1860, 1967, 2000]: print("Ln = ", Ln, ", Eo = ", Eo, ", Ls = ", Ls, ", Lo = ", Lo, ", m = ", m)

        AA = aa / (abs(aa) + abs(Eo - Lo) + 2 * m * (Eo - Ln) / Eo)  # subnet stat with threshold
        # if l in [0, 22, 1000, 1860, 1967, 2000]: print("AA = ", AA, ", aa = ", aa, ", l = ", l)
        ee = Ls * Eo / (Eo - Ls)
        em = abs(Eo - Lo) + 2 * m  # suball NULL
        ee = ee / (ee + em)  # subnet stat without threshold
        aa = (aa - m) / (aa + m)
        if AA > AA_max:
            AA_max = AA
            l_max = l
            stat = AA
            Em = Eo
            Am = aa
            lm = l_max
            Vm = m
            suball = ee
            EE = em

    return l_max, stat, Em, Am, lm, Vm, suball, EE


@njit(cache=True)
def mra_statistics(n_ifo, n_pix, FP, FX, rms, td00, td90, td_energy, ml,
                   network_energy_threshold, e2or, subcut, xtalks, xtalks_lookup, l_max):
    Es = float32(2 * e2or)
    network_energy_threshold = float32(network_energy_threshold)
    offset = int(td00.shape[0] / 2)
    # print("offset: ", offset, td00.shape, ml.shape, td_energy.shape)

    rNRG = np.zeros(n_pix, dtype=float32)  # _rE
    # pNRG = np.zeros(n_pix, dtype=float32)  # _pE

    # get time delayed data slice at sky location l, make sure it is numpy float32 array
    v_energy = np.empty((n_ifo, n_pix), dtype=float32)  # pe
    v00 = np.empty((n_ifo, n_pix), dtype=float32)  # pa
    v90 = np.empty((n_ifo, n_pix), dtype=float32)  # pA

    for i in range(n_ifo):
        v_energy[i] = td_energy[ml[i, l_max] + offset, i]

    m = float32(0)  # pixels above threshold
    # Eo = float32(0)  # total network energy
    # Ls = float32(0)  # subnetwork energy
    # Ln = float32(0)  # network energy above subnet threshold
    for j in range(n_pix):
        _rE = 0
        for i in range(n_ifo):  # get pixel energy
            _rE += v_energy[i, j]
        rNRG[j] = _rE  # store pixel energy
        _msk = float32(1.0) if rNRG[j] > network_energy_threshold else float32(0.0)  # E>En  0/1 mask
        m += _msk  # count pixels above threshold
        # pNRG[j] = rNRG[j] * _msk  # zero sub-threshold pixels
        # Eo += pNRG[j]
        # for i in range(n_ifo):
            # pNRG[j] = min(rNRG[j] - v_energy[i, j], pNRG[j])  # subnetwork energy
        # Ls += pNRG[j]  # subnetwork energy
        # _msk = float32(1.0) if pNRG[j] > Es else float32(0.0)  # subnet energy > Es 0/1 mask
        # Ln += rNRG[j] * _msk  # network energy

    # Eo = Eo + float32(0.01)
    m = int(2 * m)
    # aa = float32(Ls * Ln / (Eo - Ls))

    for i in range(n_ifo):
        v00[i] = td00[ml[i, l_max] + offset, i]
        v90[i] = td90[ml[i, l_max] + offset, i]

    xi, XI, _, _ = sse_MRA_ps(network_energy_threshold, m, rNRG,
                                    v00, v90, xtalks, xtalks_lookup)


    m = 0
    Ls = Ln = Eo = float32(0.0)
    reduced_rms = np.zeros((n_pix, n_ifo), dtype=float32)
    reduced_v00 = np.zeros((n_pix, n_ifo), dtype=float32)
    reduced_v90 = np.zeros((n_pix, n_ifo), dtype=float32)
    for j in range(n_pix):
        ee = float32(0.)
        for i in range(n_ifo):
            ee += xi[i, j] * xi[i, j] + XI[i, j] * XI[i, j]
        if ee < network_energy_threshold:
            continue

        for i in range(n_ifo):
            reduced_rms[m, i] = rms[j, i]
            reduced_v00[m, i] = v00[i, j]
            reduced_v90[m, i] = v90[i, j]
        m += 1

        # dominant pixel energy
        em = float32(0.0)
        for i in range(n_ifo):
            _em = xi[i, j] * xi[i, j] + XI[i, j] * XI[i, j]
            if _em > em:
                em = _em

        Ls += ee - em
        Eo += ee  # subnetwork energy, network energy
        if ee - em > Es:
            Ln += ee  # network energy above subnet threshold

    Lo = 0
    _, f, F, _, _, _, _, _ = dpf_np_loops_vec(FP[l_max], FX[l_max], reduced_rms[:m, :])

    # calculate likelihood
    for j in range(m):
        Lo += sse_like_ps(f[j], F[j], reduced_v00[j], reduced_v90[j])
    # print("Ln = ", Ln, ", Eo = ", Eo, ", Ls = ", Ls, ", Lo = ", Lo, ", m = ", m)
    # AA = aa / (abs(aa) + abs(Eo - Lo) + 2 * m * (Eo - Ln) / Eo)  # subnet stat with threshold
    # print("AA = ", AA, ", aa = ", aa, ", l = ", l_max)
    # ee = Ls * Eo / (Eo - Ls)
    # em = abs(Eo - Lo) + 2 * m  # suball NULL
    # ee = ee / (ee + em)  # subnet stat without threshold
    # aa = (aa - m) / (aa + m)

    submra = Ls * Eo / (Eo - Ls + float32(1e-16))  # MRA subnet statistic
    submra /= abs(submra) + abs(Eo - Lo) + 2 * (m + 6)  # MRA subnet coefficient
    rHo = np.sqrt(Lo * Lo / (Eo + 2 * m + float32(1e-16)) / 2) # MRA subnet residual
    return submra, rHo, Eo, Lo, Ls, m



@njit(cache=True)
def sse_like_ps(fp, fx, am, AM):
    """
    input fp,fx - antenna patterns in DPF
    input am,AM - network amplitude vectors
    returns: (xp*xp+XP*XP)/|f+|^2+(xx*xx+XX*XX)/(|fx|^2)
    """
    xp = np.dot(fp, am)  # fp*am
    XP = np.dot(fp, AM)  # fp*AM
    xx = np.dot(fx, am)  # fx*am
    XX = np.dot(fx, AM)  # fx*AM
    gp = np.dot(fp, fp) + float32(1.e-12)  # fx*fx + epsilon
    gx = np.dot(fx, fx) + float32(1.e-12)  # fx*fx + epsilon
    xp = xp * xp + XP * XP  # xp=xp*xp+XP*XP
    xx = xx * xx + XX * XX  # xx=xx*xx+XX*XX
    return xp / gp + xx / gx  # regularized projected energy


@njit(cache=True)
def sse_MRA_ps(Eo, K, rNRG, v_00, v_90, xtalks, xtalks_lookup, DEBUG=False):
    """
    fast multi-resolution analysis inside sky loop
    select max E pixel and either scale or skip it based on the value of residual
    """
    a_00 = v_00.copy()
    a_90 = v_90.copy()

    n_ifo, n_pix = a_00.shape

    # ee = rNRG  # residual energy
    pNRG = np.full(n_pix, float32(-1.0))  # Initialize pp with -1, assuming it's the purpose of pNRG in this context
    EE = 0.0  # extracted energy
    mam = np.zeros(n_ifo)
    mAM = np.zeros(n_ifo)

    amp = np.zeros((n_ifo, n_pix), dtype=float32)
    AMP = np.zeros((n_ifo, n_pix), dtype=float32)

    for j in range(n_pix):
        if rNRG[j] > Eo:
            pNRG[j] = 0

    k = 0
    m = 0

    while k < K:
        m = np.argmax(rNRG)  # find max pixel
        # if DEBUG:
        #     print("m = ", m)
        if rNRG[m] <= Eo:
            # if DEBUG: print("!!!!! rNRG[m] <= Eo: ", rNRG[m], Eo, k)
            break

        # get PC energy
        E = 0
        for i in range(n_ifo):
            E += a_00[i][m] * a_00[i][m] + a_90[i][m] * a_90[i][m]
        EE += E

        if E / EE < 0.01:  # ignore small PC
            # if DEBUG:
            #     print("E / EE < 0.01: ", E, EE, k)
            break

        for i in range(n_ifo):
            mam[i] = a_00[i][m]  # store a00 for max pixel
            mAM[i] = a_90[i][m]  # store a90 for max pixel

        for i in range(n_ifo):
            amp[i][m] += mam[i]  # update 00 PC
            AMP[i][m] += mAM[i]  # update 90 PC

        xtalk_range = xtalks_lookup[m]
        xtalk = xtalks[xtalk_range[0]:xtalk_range[1]]
        xtalk_indexes = xtalk[:, 0].astype(np.int32)
        xtalk_cc = np.vstack((xtalk[:, 4], xtalk[:, 5], xtalk[:, 6], xtalk[:, 7]))  # 4xM matrix

        for j in range(len(xtalk_indexes)):
            n = xtalk_indexes[j]
            if rNRG[n] > Eo:
                #  _sse_rotsub_ps(__m128* _u, float c, __m128* _v, float s, __m128* _a)
                #  calculate a -= u*c + v*s and return a*a
                # _sse_rotsub_ps(_m00,c[4],_m90,c[5],_a00+n*f)
                # _sse_rotsub_ps(_m00,c[6],_m90,c[7],_a90+n*f)
                rNRG[n] = 0
                for i in range(n_ifo):
                    a_00[i][n] -= mam[i] * xtalk_cc[0][j] + mAM[i] * xtalk_cc[1][j]
                    a_90[i][n] -= mam[i] * xtalk_cc[2][j] + mAM[i] * xtalk_cc[3][j]
                    rNRG[n] += a_00[i][n] * a_00[i][n] + a_90[i][n] * a_90[i][n]

        # store PC energy
        pp = 0
        for i in range(n_ifo):
            pp += amp[i][m] * amp[i][m] + AMP[i][m] * AMP[i][m]
        pNRG[m] = pp

        k += 1
    # if DEBUG:
    #     print("k = ", k, ", K = ", K)

    return amp, AMP, rNRG, pNRG
