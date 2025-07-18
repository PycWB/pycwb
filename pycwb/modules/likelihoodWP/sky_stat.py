from math import sqrt

import numpy as np
from numba import njit, prange, float32, int32


@njit(cache=True)
def _avx_loadata_ps(p, q, En):
    n_ifo = len(p)  # Number of interferometers
    n_pix = len(p[0])  # Number of pixels

    energy_total = np.empty(n_pix, dtype=float32)
    mask = np.empty(n_pix, dtype=int32)
    ee = float32(0.0)
    EE = float32(0.0)
    NN = int32(0)

    for i in range(n_pix):
        aa = float32(0.0)
        AA = float32(0.0)

        for j in range(n_ifo):
            aa += p[j][i] * p[j][i]
            AA += q[j][i] * q[j][i]

        energy_total[i] = aa + AA + float32(1e-12)
        mask[i] = energy_total[i] > En
        NN += mask[i]
        ee += energy_total[i]
        energy_total[i] *= mask[i]
        EE += energy_total[i]

    return EE / float32(2.), NN, energy_total, mask


@njit(cache=True)
def load_data_from_td(v00, v90, network_energy_threshold):
    """
    Compute the total energy for each pixel and the mask based on the network energy threshold.

    TODO: rename this function to something more descriptive, like `compute_pixel_energy_and_mask`. And split the logic into smaller functions if necessary.

    Parameters:
    -----------
    v00 : np.ndarray
        The 00 polarization data for each interferometer and pixel.
    v90 : np.ndarray
        The 90 polarization data for each interferometer and pixel.
    network_energy_threshold : float
        The threshold for the network energy to determine if a pixel is active.

    Returns:
    --------
    tuple
        - EE : float
            The total energy of the network.
        - NN : int
            The number of active pixels.
        - energy_total : np.ndarray
            The total energy for each pixel.
        - mask : np.ndarray
            A mask indicating which pixels are active based on the network energy threshold.
    """

    n_ifo = len(v00)  # Number of interferometers
    n_pix = len(v00[0])  # Number of pixels

    energy_total = np.empty(n_pix, dtype=float32)
    mask = np.empty(n_pix, dtype=int32)
    ee = float32(0.0)
    EE = float32(0.0)
    NN = int32(0)

    for i in range(n_pix):
        aa = float32(0.0)
        AA = float32(0.0)

        for j in range(n_ifo):
            aa += v00[j][i] * v00[j][i]
            AA += v90[j][i] * v90[j][i]

        energy_total[i] = aa + AA + float32(1e-12)
        mask[i] = energy_total[i] > network_energy_threshold
        NN += mask[i]
        ee += energy_total[i]
        energy_total[i] *= mask[i]
        EE += energy_total[i]

    return EE / float32(2.), NN, energy_total, mask


@njit(cache=True)
def avx_GW_ps(v00, v90, f, F, fp, fx, ni, et, mask, reg):
    """
    GW strain packet

    Parameters
    ----------
    v00 : np.ndarray
        The 00 polarization component of the packet. v00[ifo][pixel]
    v90 : np.ndarray
        The 90 polarization component of the packet. v90[ifo][pixel]
    f : np.ndarray
        The plus polarization component in the DPF. f[pixel][ifo]
    F : np.ndarray
        The cross polarization component in the DPF. F[pixel][ifo]
    fp : np.ndarray
        The plus polarization component in the DPF, normalized. |f+|^2. fp[pixel][ifo]
    fx : np.ndarray
        The cross polarization component in the DPF, normalized. |fx|^2. fx[pixel][ifo]
    ni : np.ndarray
        The noise index for each pixel. ni[pixel][ifo]
    et : np.ndarray
        The total energy for each pixel. et[pixel][ifo]
    mask : np.ndarray
        The mask indicating active pixels. mask[pixel]
    reg : tuple
        The regularization parameters.

    Returns
    -------
    tuple
        - NN : int
            The number of pixels above threshold
        - p_updated : np.ndarray
            The updated 00 component of the packet. p_updated[ifo][pixel]
        - q_updated : np.ndarray
            The updated 90 component of the packet. q_updated[ifo][pixel]
        - mask_updated : np.ndarray
            The updated mask for the pixels. mask_updated[pixel]
        - au : np.ndarray
            Amplitude component
        - AU : np.ndarray
            Amplitude component
        - av : np.ndarray
            Amplitude component
        - AV : np.ndarray
            Amplitude component
    """
    n_ifo = len(v00)  # Number of interferometers
    n_pix = len(v00[0])  # Number of pixels

    au = np.empty(n_pix, dtype=np.float32)
    AU = np.empty(n_pix, dtype=np.float32)
    av = np.empty(n_pix, dtype=np.float32)
    AV = np.empty(n_pix, dtype=np.float32)
    mask_updated = np.empty(n_pix, dtype=np.float32)
    p_updated = np.empty((n_ifo, n_pix), dtype=np.float32)
    q_updated = np.empty((n_ifo, n_pix), dtype=np.float32)

    _o = np.float32(1e-5)
    _rr = np.float32(reg[0])
    _RR = np.float32(reg[1])
    NN = np.int32(0)

    for i in range(n_pix):
        _xp, _XP, _xx, _XX = float32(0), float32(0), float32(0), float32(0)
        for j in range(n_ifo):
            _xp += v00[j][i] * f[i][j]
            _XP += v90[j][i] * f[i][j]
            _xx += v00[j][i] * F[i][j]
            _XX += v90[j][i] * F[i][j]

        _f = sqrt(ni[i] * (_xp * _xp + _XP * _XP) / (et[i] + _o)) * _rr - fp[i]
        _f = _f if _f > float32(0.) else float32(0.0)
        _f = mask[i] / (fp[i] + _f + _o)

        _h = _xp * _f
        _H = _XP * _f
        _h = _h * _h + _H * _H
        _H = _xx * _xx + _XX * _XX
        _F = sqrt(_H / (_h + _o))
        _R = float32(0.1) + _RR / (et[i] + _o)  # dynamic x-regulator
        _F = _F * _R - fx[i]
        _F = _F if _F > float32(0.) else float32(0.0)
        _F = mask[i] / (fx[i] + _F + _o)

        au[i] = _xp * _f
        AU[i] = _XP * _f
        av[i] = _xx * _F
        AV[i] = _XX * _F

        _a = _f * fp[i] + _F * fx[i]  # Gaussin noise correction
        NN += mask[i]  # number of pixels
        mask_updated[i] = _a + mask[i] - float32(1.0)  # -1 - rejected, >=0 accepted

        for j in range(n_ifo):
            p_updated[j][i] = f[i][j] * au[i] + F[i][j] * av[i]
            q_updated[j][i] = f[i][j] * AU[i] + F[i][j] * AV[i]

    # su, sv, uu, UU, vv, VV = float32(0), float32(0), float32(0), float32(0), float32(0), float32(0)
    # for i in range(n_pix):
    #     su += au[i] * AU[i]
    #     sv += av[i] * AV[i]
    #     uu += au[i] * au[i]
    #     UU += AU[i] * AU[i]
    #     vv += av[i] * av[i]
    #     VV += AV[i] * AV[i]

    # nn = sqrt((uu - UU) * (uu - UU) + 4 * su * su) + float32(0.0001)  # co/si norm
    # cu = (uu - UU) / nn
    # et = uu + UU  # rotation cos(2p) and sin(2p)
    # uu = sqrt((et + nn) / float32(2.0))
    # UU = 0 if et > nn else sqrt((et - nn) / float32(2.0))  # amplitude of first/second component
    # nn = float32(1.0) if su > float32(0.0) else float32(-1.0)  # norm^2 of 2*cos^2 and 2*sin*cos
    # su = sqrt((float32(1.0) - cu) / float32(2.0))
    # cu = nn * sqrt((float32(1.0) + cu) / float32(2.0))  # normalized rotation sin/cos

    # second packet
    # nn = sqrt((vv - VV) * (vv - VV) + float32(4.0) * sv * sv) + float32(0.0001)  # co/si norm
    # cv = (vv - VV) / nn
    # ET = vv + VV  # rotation cos(2p) and sin(2p)
    # vv = sqrt((ET + nn) / float32(2.0))
    # VV = float32(0.) if ET > nn else sqrt((ET - nn) / float32(2.0))  # first/second component energy
    # nn = float32(1.0) if sv > float32(0.0) else float32(-1.0)  # norm^2 of 2*cos^2 and 2*sin*cos
    # sv = sqrt((float32(1.0) - cv) / float32(2.0))
    # cv = nn * sqrt((float32(1.0) + cv) / float32(2.0))  # normalized rotation sin/cos// first packet

    return NN, p_updated, q_updated, mask_updated, au, AU, av, AV


@njit(cache=True)
def avx_ort_ps(v00, v90, mask):
    """
    orthogonalize data vectors v00 and v90, calculate norms of orthogonal vectors and rotation sin & cos

    Parameters
    ----------
    v00 : np.ndarray
        The 00 polarization component of the packet. v00[ifo][pixel]
    v90 : np.ndarray
        The 90 polarization component of the packet. v90[ifo][pixel]
    mask : np.ndarray
        The mask indicating active pixels. mask[pixel]

    Returns
    -------
    tuple
        - E: float
            signal energy
        - si: np.ndarray
            sin of the rotation angle for each pixel. si[pixel]
        - co: np.ndarray
            cos of the rotation angle for each pixel. co[pixel]
        - ee: np.ndarray
            plus component energy for each pixel. ee[pixel]
        - EE: np.ndarray
            cross component energy for each pixel. EE[pixel]
    """
    n_ifo = len(v00)  # Number of interferometers
    n_pix = len(v00[0])  # Number of pixels
    _0 = np.float32(0)
    _1 = np.float32(1)
    _o = np.float32(1e-21)

    si = np.empty(n_pix, dtype=np.float32)
    co = np.empty(n_pix, dtype=np.float32)
    ee = np.empty(n_pix, dtype=np.float32)
    EE = np.empty(n_pix, dtype=np.float32)

    e = np.float32(0)
    E = np.float32(0)

    for i in range(n_pix):
        aa = np.float32(0)
        AA = np.float32(0)
        aA = np.float32(0)

        for j in range(n_ifo):
            aa += v00[j][i] * v00[j][i]
            AA += v90[j][i] * v90[j][i]
            aA += v00[j][i] * v90[j][i]

        # Orthogonalization sin and cos calculations
        si[i] = aA * float32(2.)  # rotation 2*sin*cos*norm
        co[i] = aa - AA  # rotation (cos^2-sin^2)*norm
        et = aa + AA + _o  # total energy
        cc = co[i] * co[i]  # cos^2
        ss = si[i] * si[i]  # sin^2
        nn = np.sqrt(cc + ss)  # co/si norm
        ee[i] = (et + nn) / float32(2.)  # first component energy
        EE[i] = (et - nn) / float32(2.)  # second component energy
        cc = co[i] / (nn + _o)  # cos(2p)
        nn = 1 if si[i] > _0 else 0  # 1 if sin(2p)>0. or 0 if sin(2p)<0.
        ss = 2 * nn - 1  # 1 if sin(2p)>0. or-1 if sin(2p)<0.
        si[i] = np.sqrt((float32(1.) - cc) / float32(2.))  # |sin(p)|
        co[i] = np.sqrt((float32(1.) + cc) / float32(2.))  # |cos(p)|
        co[i] *= ss  # cos(p)

        mk = 1 if mask[i] > _0 else 0  # event mask
        e += mk * ee[i]
        E += mk * EE[i]

    return e + E, si, co, ee, EE


@njit(cache=True)
def avx_stat_ps(v00, v90, s, S, si, co, mask):
    """
    returns coherent statistics in the format {cc,ec,ed,gn}

    Parameters
    ----------
    v00 : np.ndarray
        The 00 polarization component of the packet. v00[ifo][pixel]
    v90 : np.ndarray
        The 90 polarization component of the packet. v90[ifo][pixel]
    s : np.ndarray
        The updated 00 component of the packet. s[ifo][pixel]
    S : np.ndarray
        The updated 90 component of the packet. S[ifo][pixel]
    si : np.ndarray
        The sin of the rotation angle for each pixel. si[pixel]
    co : np.ndarray
        The cos of the rotation angle for each pixel. co[pixel]
    mask : np.ndarray
        The mask indicating active pixels. mask[pixel]

    Returns
    -------
    tuple
        - corr_coeff : float
            The network correlation coefficient.
        - EC: float
            The total coherent energy.
        - NN: int
            The number of pixels
        - total_noise: float
            The total noise
        - ec: np.ndarray
            The coherent energy for each pixel.
        - gn: np.ndarray
            The G-noise correction for each pixel.
        - rn: np.ndarray
            The residual noise in the TF domain for each pixel.
    """
    n_ifo = len(v00)  # Number of interferometers
    n_pix = len(v00[0])  # Number of pixels

    _o = np.float32(0.001)
    _0 = np.float32(0)
    _1 = np.float32(1)
    _2 = np.float32(2)
    # _k = 2 * (1 - k)

    ec = np.empty(n_pix, dtype=np.float32)
    gn = np.empty(n_pix, dtype=np.float32)
    rn = np.empty(n_pix, dtype=np.float32)

    LL = np.float32(0)
    Lr = np.float32(0)
    EC = np.float32(0)
    GN = np.float32(0)
    RN = np.float32(0)
    NN = np.float32(0)

    for i in range(n_pix):
        c = np.float32(0)
        C = np.float32(0)
        ss = np.float32(0)
        SS = np.float32(0)
        rr = np.float32(0)
        RR = np.float32(0)
        xs = np.float32(0)
        XS = np.float32(0)

        for j in range(n_ifo):
            s_ = s[j][i] * co[i] + S[j][i] * si[i]
            x_ = v00[j][i] * co[i] + v90[j][i] * si[i]
            S_ = S[j][i] * co[i] - s[j][i] * si[i]
            X_ = v90[j][i] * co[i] - v00[j][i] * si[i]

            a = s_ * x_
            A = S_ * X_
            xs += a
            XS += A

            c += a * a
            C += A * A
            ss += s_ * s_
            SS += S_ * S_
            rr += (s[j][i] - v00[j][i]) ** 2
            RR += (S[j][i] - v90[j][i]) ** 2

        mk = 1 if mask[i] >= _0 else 0  # event mask
        c = c / (xs * xs + _o)  # first component incoherent energy
        C = C / (XS * XS + _o)  # second component incoherent energy
        ll = mk * (ss + SS)  # signal energy
        ss = ss * (float(1.) - c)  # 00 coherent energy
        SS = SS * (float(1.) - C)  # 90 coherent energy
        ec[i] = mk * (ss + SS)  # coherent energy
        gn[i] = mk * float(2.) * mask[i]  # G-noise correction
        rn[i] = mk * (rr + RR)  # residual noise in TF domain

        a = float(2.) * abs(ec[i])  # 2*|ec|
        A = rn[i] + gn[i] + _o  # NULL
        cc = ec[i] / (a + A)  # correlation coefficient
        Lr += ll * cc  # reduced likelihood
        mm = 1 if ec[i] > _o else 0  # coherent energy mask

        LL += ll  # total signal energy
        EC += ec[i]  # total coherent energy
        GN += gn[i]  # total G-noise correction
        RN += rn[i]  # residual noise in TF domain
        NN += mm  # number of pixel in TF domain with Ec>0

    corr_coeff = float32(2.0) * Lr / (LL + _o)  # network correlation coefficient
    total_noise = (GN + RN) / float32(2.0)

    return corr_coeff, EC, NN, total_noise, ec, gn, rn
