"""Caller-owned buffers for sky-scan helpers.

Arithmetic and return contracts follow dpf.py and sky_stat.py. Every output
is overwritten before use; DPF accumulation arrays are explicitly reset.
Buffers belong to one delay-group worker and must not alias inputs.
"""

from math import sqrt
import numpy as np
from numba import njit, float32, uint32
from .dpf import mul_vec, sub_vec, add_vec, norm_vec, div_vec, avg_vec, sin_from_cc, cos_from_cc, pos_sign_vec


@njit(cache=True)
def dpf_np_loops_vec_into(Fp0, Fx0, rms, scratch):
    """Compute the dominant polarization frame into caller-owned arrays.

    Parameters
    ----------
    Fp0 : np.ndarray
        The Fp0 vector for the current sky location.
    Fx0 : np.ndarray
        The Fx0 vector for the current sky location.
    rms : np.ndarray
        The rms values for the pixels, shape (NPIX, NIFO).

    scratch : tuple of numpy.ndarray
        Writable float32 buffers: (f, F, si, co, fp, fx, ni): f/F have shape (n_pix, n_ifo);
        the remaining arrays have shape (n_pix,). fx/ni are reset before accumulation.
        Every buffer is overwritten before use and must not alias any input
        or another scratch buffer. Returned arrays borrow these buffers and
        remain valid only until the next call with the same scratch tuple.

    Returns
    -------
    tuple
        - NI : float
            The normalized index. (?)
        - f: np.ndarray
            The plus polarization component in the DPF.
        - F: np.ndarray
            The cross polarization component in the DPF.
        - fp: np.ndarray
            |f+|^2
        - fx: np.ndarray
            |fx|^2
        - si: np.ndarray
            The sine component of the DPF.
        - co: np.ndarray
            The cosine component of the DPF.
        - ni: np.ndarray
            The network index for each pixel.
    """
    NPIX, NIFO = rms.shape
    NPIX = uint32(NPIX)
    NIFO = uint32(NIFO)

    # variables for return
    f, F, si, co, fp, fx, ni = scratch

    fx.fill(0)
    ni.fill(0)

    _o = float32(1e-9)

    # Compute f and F
    for j in range(NIFO):
        for i in range(NPIX):
            f[i, j] = mul_vec(rms[i, j], Fp0[j])
            F[i, j] = mul_vec(rms[i, j], Fx0[j])

    # Compute ff, FF, and fF
    for i in range(NPIX):
        _ff = float32(0.0)
        _FF = float32(0.0)
        _fF = float32(0.0)

        for j in range(NIFO):
            _ff += f[i, j] * f[i, j]
            _FF += F[i, j] * F[i, j]
            _fF += F[i, j] * f[i, j]

        # Compute si, co, AP, nn, fp, and cc
        _si = mul_vec(float32(2.0), _fF)  # rotation 2*sin*cos*norm
        _co = sub_vec(_ff, _FF)  # rotation (cos^2-sin^2)*norm
        _AP = add_vec(_ff, _FF)  # total antenna norm
        _nn = norm_vec(_co, _si)  # co/si norm    np.sqrt(_co * _co + _si * _si)
        _cc = div_vec(_co, _nn)  # cos(2p)       _co / (_nn + 1e-9)
        fp[i] = avg_vec(_AP, _nn)  # |f+|^2        (_AP + _nn) / 2.
        si[i] = sin_from_cc(_cc)  # |sin(p)|      sqrt((1. - _cc) / 2.)
        co[i] = cos_from_cc(_cc, _si)  # cos(p)        (sqrt((1. + _cc) / 2.) if _si > 0.0 else - sqrt((1. + _cc) / 2.))

    # Compute f_new, F_new, fF_new, F_new, fx, ni
    for i in range(NPIX):
        for j in range(NIFO):
            f[i, j], F[i, j] = f[i, j] * co[i] + F[i, j] * si[i], F[i, j] * co[i] - f[i, j] * si[i]

        fF_new = float32(0.0)
        for j in range(NIFO):
            fF_new += f[i, j] * F[i, j]
        fF_new = div_vec(fF_new, fp[i])

        for j in range(NIFO):
            F[i, j] -= f[i, j] * fF_new
            fx[i] += F[i, j] * F[i, j]
            ni[i] += f[i, j] ** 4

    NI, NN = float32(0.0), uint32(0)

    # Compute NI and NN
    for i in range(NPIX):
        ni[i] = div_vec(ni[i], mul_vec(fp[i], fp[i]))
        NI += div_vec(fx[i], ni[i])  # sum of |fx|^2/2/ni
        NN += pos_sign_vec(fp[i])  # pixel count
        # NN += 1 if fp[i] > 0.0 else 0

    return sqrt(NI / (NN + 0.01)), f, F, fp, fx, si, co, ni


@njit(cache=True)
def avx_GW_ps_into(v00, v90, f, F, fp, fx, ni, et, mask, reg, scratch):
    """Project a GW strain packet into caller-owned arrays.

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
        The noise index for each pixel, shape (n_pix,).
    et : np.ndarray
        The total energy for each pixel, shape (n_pix,).
    mask : np.ndarray
        The mask indicating active pixels. mask[pixel]
    reg : tuple
        The regularization parameters.

    scratch : tuple of numpy.ndarray
        Writable float32 buffers: (au, AU, av, AV, mask_updated, p_updated, q_updated): the first
        five arrays have shape (n_pix,); the final two have shape (n_ifo, n_pix).
        Every buffer is overwritten before use and must not alias any input
        or another scratch buffer. Returned arrays borrow these buffers and
        remain valid only until the next call with the same scratch tuple.

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

    au, AU, av, AV, mask_updated, p_updated, q_updated = scratch

    _o = np.float32(1e-9)
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
        _f = _f if _f > float32(0.0) else float32(0.0)
        _f = mask[i] / (fp[i] + _f + _o)

        _h = _xp * _f
        _H = _XP * _f
        _h = _h * _h + _H * _H
        _H = _xx * _xx + _XX * _XX
        _F = sqrt(_H / (_h + _o))
        _R = float32(0.1) + _RR / (et[i] + _o)  # dynamic x-regulator
        _F = _F * _R - fx[i]
        _F = _F if _F > float32(0.0) else float32(0.0)
        _F = mask[i] / (fx[i] + _F + _o)

        au[i] = _xp * _f
        AU[i] = _XP * _f
        av[i] = _xx * _F
        AV[i] = _XX * _F

        _a = _f * fp[i] + _F * fx[i]  # Gaussian noise correction
        NN += mask[i]  # number of pixels
        mask_updated[i] = _a + mask[i] - float32(1.0)  # -1 - rejected, >=0 accepted

        for j in range(n_ifo):
            p_updated[j][i] = f[i][j] * au[i] + F[i][j] * av[i]
            q_updated[j][i] = f[i][j] * AU[i] + F[i][j] * AV[i]

    return NN, p_updated, q_updated, mask_updated, au, AU, av, AV


@njit(cache=True)
def avx_ort_ps_into(v00, v90, mask, scratch):
    """Orthogonalize quadratures into caller-owned rotation and energy arrays.

    Parameters
    ----------
    v00 : np.ndarray
        The 00 polarization component of the packet. v00[ifo][pixel]
    v90 : np.ndarray
        The 90 polarization component of the packet. v90[ifo][pixel]
    mask : np.ndarray
        The mask indicating active pixels. mask[pixel]

    scratch : tuple of numpy.ndarray
        Writable float32 buffers: (si, co, ee, EE), each with shape (n_pix,).
        Every buffer is overwritten before use and must not alias any input
        or another scratch buffer. Returned arrays borrow these buffers and
        remain valid only until the next call with the same scratch tuple.

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

    si, co, ee, EE = scratch

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
        si[i] = aA * float32(2.0)  # rotation 2*sin*cos*norm
        co[i] = aa - AA  # rotation (cos^2-sin^2)*norm
        et = aa + AA + _o  # total energy
        cc = co[i] * co[i]  # cos^2
        ss = si[i] * si[i]  # sin^2
        nn = np.sqrt(cc + ss)  # co/si norm
        ee[i] = (et + nn) / float32(2.0)  # first component energy
        EE[i] = (et - nn) / float32(2.0)  # second component energy
        cc = co[i] / (nn + _o)  # cos(2p)
        nn = 1 if si[i] > _0 else 0  # 1 if sin(2p)>0. or 0 if sin(2p)<0.
        ss = 2 * nn - 1  # 1 if sin(2p)>0. or-1 if sin(2p)<0.
        si[i] = np.sqrt((float32(1.0) - cc) / float32(2.0))  # |sin(p)|
        co[i] = np.sqrt((float32(1.0) + cc) / float32(2.0))  # |cos(p)|
        co[i] *= ss  # cos(p)

        mk = 1 if mask[i] > _0 else 0  # event mask
        e += mk * ee[i]
        E += mk * EE[i]

    return e + E, si, co, ee, EE


@njit(cache=True)
def avx_stat_ps_into(v00, v90, s, S, si, co, mask, scratch):
    """Compute coherent statistics into caller-owned per-pixel arrays.

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

    scratch : tuple of numpy.ndarray
        Writable float32 buffers: (ec, gn, rn), each with shape (n_pix,).
        Every buffer is overwritten before use and must not alias any input
        or another scratch buffer. Returned arrays borrow these buffers and
        remain valid only until the next call with the same scratch tuple.

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

    _o = np.float32(1.0e-9)
    _0 = np.float32(0)
    _1 = np.float32(1)
    _2 = np.float32(2)

    ec, gn, rn = scratch

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
        ss = ss * (float(1.0) - c)  # 00 coherent energy
        SS = SS * (float(1.0) - C)  # 90 coherent energy
        ec[i] = mk * (ss + SS)  # coherent energy
        gn[i] = mk * float(2.0) * mask[i]  # G-noise correction
        rn[i] = mk * (rr + RR)  # residual noise in TF domain

        a = float(2.0) * abs(ec[i])  # 2*|ec|
        A = rn[i] + gn[i] + _o  # NULL
        cc = ec[i] / (a + A)  # correlation coefficient
        Lr += ll * cc  # reduced likelihood
        mm = 1 if ec[i] > _o else 0  # coherent energy mask

        LL += ll  # total signal energy
        EC += ec[i]  # total coherent energy
        GN += gn[i]  # total G-noise correction
        RN += rn[i]  # residual noise in TF domain
        NN += mm  # number of pixel in TF domain with Ec>0

    # watavx.hh keeps this 0.001 energy offset even with its division bugfix
    # enabled. It is distinct from the 1e-9 per-pixel division epsilon.
    corr_coeff = float32(2.0) * float32(Lr / (LL + 0.001))
    total_noise = (GN + RN) / float32(2.0)

    return corr_coeff, EC, NN, total_noise, ec, gn, rn
