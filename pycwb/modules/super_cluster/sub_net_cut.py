import numpy as np
# from numba import float32
from numba import njit
from numpy import float32
from pycwb.modules.likelihoodWP.dpf import dpf_np_loops_vec
from pycwb.modules.xtalk.monster import getXTalk_pixels
from pycwb.modules.likelihoodWP.likelihood import load_data_from_pixels


def sub_net_cut(pixels, ml, FP, FX, acor, e2or, n_ifo, n_sky, lag, subnet, subcut, subnorm,
                xtalk_coeff, xtalk_lookup_table, layers, nRes, mra=True):
    """
    This function is used to cut the subnet from the lag and return the new lag
    """
    network_energy_threshold = 2 * acor * acor * n_ifo
    n_pix = len(pixels)

    rms, td00, td90, td_energy = load_data_from_pixels(pixels, n_ifo)

    cluster_xtalk_lookup, cluster_xtalk = getXTalk_pixels(pixels, True, layers, xtalk_coeff, xtalk_lookup_table)

    td00 = np.transpose(td00.astype(np.float32), (2, 0, 1))  # (ndelay, nifo, npix)
    td90 = np.transpose(td90.astype(np.float32), (2, 0, 1))  # (ndelay, nifo, npix)
    td_energy = np.transpose(td_energy.astype(np.float32), (2, 0, 1))  # (ndelay, nifo, npix)
    FP = FP.T.astype(np.float32)
    FX = FX.T.astype(np.float32)
    rms = rms.T.astype(np.float32)

    l_max = optimze_sky_loc(n_ifo, n_pix, n_sky, FP, FX, rms, td00, td90, td_energy, ml,
                            network_energy_threshold, e2or,
                            subcut,
                            cluster_xtalk, cluster_xtalk_lookup, mra)
    print(f"l_max: {l_max}")


@njit(cache=True)
def optimze_sky_loc(n_ifo, n_pix, n_sky, FP, FX, rms, td00, td90, td_energy, ml, network_energy_threshold, e2or, subcut,
                    xtalks, xtalks_lookup, mra):
    Es = 2 * e2or
    offset = int(td00.shape[0] / 2)
    print("offset: ", offset, td00.shape, ml.shape, td_energy.shape)

    rNRG = np.zeros(n_pix, dtype=float32)  # _rE
    pNRG = np.zeros(n_pix, dtype=float32)  # _pE
    print("En = ", network_energy_threshold, ', Es = ', Es, ", n_pix = ", n_pix, ", n_sky = ", n_sky)
    AA_array = np.zeros(n_sky, dtype=float32)
    for l in range(n_sky):
        # TODO: sky sky mask
        # get time delayed data slice at sky location l, make sure it is numpy float32 array
        v_energy = np.empty((n_ifo, n_pix), dtype=float32)  # pe
        v00 = np.empty((n_ifo, n_pix), dtype=float32)  # pa
        v90 = np.empty((n_ifo, n_pix), dtype=float32)  # pA
        Fp = np.empty((n_ifo, n_pix), dtype=float32)
        Fx = np.empty((n_ifo, n_pix), dtype=float32)
        fp = np.empty(n_ifo, dtype=float32)
        fx = np.empty(n_ifo, dtype=float32)

        for i in range(n_ifo):
            v_energy[i] = td_energy[ml[i, l] + offset, i]

        m = 0  # pixels above threshold
        Eo = 0  # total network energy
        Ls = 0  # subnetwork energy
        Ln = 0  # network energy above subnet threshold
        for j in range(n_pix):
            _rE = 0
            for i in range(n_ifo):  # get pixel energy
                _rE += v_energy[i, j]
            rNRG[j] = _rE  # store pixel energy
            _msk = 1.0 # if rNRG[j] > network_energy_threshold else 0.0  # E>En  0/1 mask
            m += _msk  # count pixels above threshold
            pNRG[j] = rNRG[j] * _msk  # zero sub-threshold pixels
            Eo += pNRG[j]
            for i in range(n_ifo):
                pNRG[j] = min(rNRG[j] - v_energy[i, j], pNRG[j])  # subnetwork energy
            Ls += pNRG[j]  # subnetwork energy
            _msk = 1.0 if pNRG[j] > Es else 0.0  # subnet energy > Es 0/1 mask
            Ln += rNRG[j] * _msk  # network energy

        Eo = Eo + 0.01
        m = int(2 * m + 0.01)
        aa = Ls * Ln / (Eo - Ls)
        if l in [0, 1000, 2000]:
            print("l = ", l)
            print("Ln = ", Ln, ", Eo = ", Eo, ", Ls = ", Ls, ", m = ", m)
        if subcut >= 0 and (aa - m) / (aa + m) < subcut:
            continue

        for i in range(n_ifo):
            v00[i] = td00[ml[i, l] + offset, i]
            v90[i] = td90[ml[i, l] + offset, i]

        if mra:
            xi, XI, rNRG, pNRG = sse_MRA_ps(Eo, m, rNRG, v00, v90, xtalks, xtalks_lookup)
        else:
            # TODO: check if copy is necessary
            xi, XI = v00, v90  # pp, PP

        mask = np.zeros(n_pix, dtype=float32)
        Ls = Ln = Eo = 0
        for i in range(n_ifo):
            fp[i] = FP[l, i]
            fx[i] = FX[l, i]

        for j in range(n_pix):
            ee = 0
            for i in range(n_ifo):
                ee += xi[i, j] * xi[i, j] + XI[i, j] * XI[i, j]
            if ee < Eo:
                continue
            mask[j] = 1.0
            # normalize f+ by rms
            for i in range(n_ifo):
                Fp[i, j] = fp[i] * rms[j, i]
                Fx[i, j] = fx[i] * rms[j, i]

            # dominant pixel energy
            em = 0
            for i in range(n_ifo):
                _em = xi[i, j] * xi[i, j] + XI[i, j] * XI[i, j]
                if _em > em:
                    em = _em

            Ls += ee - em
            Eo += ee  # subnetwork energy, network energy
            if ee - em > Ls:
                Ln += ee  # network energy above subnet threshold

        if l in [0, 1000, 2000]: print("Ln = ", Ln, ", Eo = ", Eo, ", Ls = ", Ls, ", m = ", m)

        if Eo <= 0:
            continue

        Lo = 0
        for j in range(n_pix):
            if mask[j] == 0.0:
                continue
            # calculate dpf
            _, f, F, fp, fx, si, co, ni = dpf_np_loops_vec(FP[l], FX[l], rms)
            # calculate likelihood
            Es = sse_like_ps(f[j], F[j], xi[:, j], XI[:, j])
            Lo += Es

        AA = aa / (abs(aa) + abs(Eo - Lo) + 2 * m * (Eo - Ln) / Eo)  # subnet stat with threshold
        AA_array[l] = AA
        ee = Ls * Eo / (Eo - Ls)
        em = abs(Eo - Lo) + 2 * m  # suball NULL
        ee = ee / (ee + em)  # subnet stat without threshold
        aa = (aa - m) / (aa + m)
        if not mra:
            Lt = Lo  # store total coherent energy for all resolution levels
    l_max = np.argmax(AA_array)
    return l_max


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
    gp = np.dot(fp, fp) + 1.e-12  # fx*fx + epsilon
    gx = np.dot(fx, fx) + 1.e-12  # fx*fx + epsilon
    xp = xp * xp + XP * XP  # xp=xp*xp+XP*XP
    xx = xx * xx + XX * XX  # xx=xx*xx+XX*XX
    return xp / gp + xx / gx  # regularized projected energy


@njit(cache=True)
def sse_MRA_ps(Eo, K, rNRG, a_00, a_90, xtalks, xtalks_lookup):
    """
    fast multi-resolution analysis inside sky loop
    select max E pixel and either scale or skip it based on the value of residual
    """

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
        if rNRG[m] <= Eo:
            break

        # get PC energy
        E = 0
        for i in range(n_ifo):
            E += a_00[i][m] * a_00[i][m] + a_90[i][m] * a_90[i][m]
        EE += E

        if E / EE < 0.01:  # ignore small PC
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
                for i in range(n_ifo):
                    a_00[i][n] -= mam[i] * xtalk_cc[0][j] + mAM[i] * xtalk_cc[1][j]
                    a_90[i][n] -= mam[i] * xtalk_cc[2][j] + mAM[i] * xtalk_cc[3][j]
                    rNRG[n] = a_00[i][n] * a_00[i][n] + a_90[i][n] * a_90[i][n]

        # store PC energy
        pp = 0
        for i in range(n_ifo):
            pp += amp[i][m] * amp[i][m] + AMP[i][m] * AMP[i][m]
        pNRG[m] = pp

        k += 1

    return amp, AMP, rNRG, pNRG
