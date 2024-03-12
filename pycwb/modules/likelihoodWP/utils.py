from math import sqrt

import numpy as np
from numba import njit, float32


@njit(cache=True)
def avx_packet_ps(p, q, mask):
    n_ifo = len(p)
    n_pix = len(p[0])
    _o = float(0.0001)

    mk = np.empty(n_pix, dtype=np.float32)
    aa = np.zeros(n_ifo, dtype=np.float32)
    AA = np.zeros(n_ifo, dtype=np.float32)
    aA = np.zeros(n_ifo, dtype=np.float32)

    si = np.empty(n_ifo, dtype=np.float32)
    co = np.empty(n_ifo, dtype=np.float32)
    a = np.empty(n_ifo, dtype=np.float32)
    A = np.empty(n_ifo, dtype=np.float32)
    a_save = np.empty(n_ifo, dtype=np.float32)
    A_save = np.empty(n_ifo, dtype=np.float32)

    for i in range(n_pix):
        mk[i] = float32(1.0) if mask[i] > 0 else float32(0.)

    for j in range(n_ifo):
        for i in range(n_pix):
            aa[j] += mk[i] * (p[j][i] * p[j][i])
            AA[j] += mk[i] * (q[j][i] * q[j][i])
            aA[j] += mk[i] * (p[j][i] * q[j][i])

    E = np.empty(n_ifo, dtype=np.float32)
    for i in range(n_ifo):
        _si = float32(2.) * aA[i]   # rotation 2*sin*cos*norm
        _co = aa[i] - AA[i]   # rotation (cos^2-sin^2)*norm
        # print(f"_si[{i}]: ", _si, f"_co[{i}]: ", _co)
        _x = aa[i] + AA[i] + _o  # total energy
        _cc = _co * _co   # cos^2
        _ss = _si * _si   # sin^2
        _nn = sqrt(_cc + _ss)   # co/si norm
        a[i] = sqrt((_x + _nn) / float32(2.))  # first component amplitude
        A[i] = sqrt(abs((_x - _nn) / float32(2.)))  # second component energy
        _cc = _co / (_nn + _o)  # cos(2p)
        _ss = float32(1.) if _si > float32(0.) else float32(-1.)  # 1 if sin(2p)>0. or-1 if sin(2p)<0.
        si[i] = sqrt((float32(1.) - _cc) / float32(2.))  # |sin(p)|
        co[i] = sqrt((float32(1.) + _cc) / float32(2.)) * _ss  # cos(p)

        E[i] = (a[i] + A[i]) ** 2 / float32(2.)
        a_save[i] = a[i]
        A_save[i] = A[i]
        a[i] = float(1.0) / (a[i] + _o)
        A[i] = float(1.0) / (A[i] + _o)

    Ep = 0.
    for i in range(n_ifo):
        Ep += E[i]

    p_updated = np.empty((n_ifo, n_pix), dtype=np.float32)
    q_updated = np.empty((n_ifo, n_pix), dtype=np.float32)
    for j in range(n_ifo):
        for i in range(n_pix):
            _a = p[j][i] * co[j] + q[j][i] * si[j]
            _A = q[j][i] * co[j] - p[j][i] * si[j]
            p_updated[j][i] = mk[i] * _a * a[j]
            q_updated[j][i] = mk[i] * _A * A[j]

    return Ep/float32(2.), p_updated, q_updated, E, si, co, a_save, A_save


@njit(cache=True)
def packet_norm_numpy(p, q, xtalks, xtalks_lookup, mk, q_E):
    """Compute the norm of a packet of pixels.

    Parameters
    ----------
    p : np.ndarray
        The p component of the packet. p[ifo][pixel]
    q : np.ndarray
        The q component of the packet. q[ifo][pixel]
    xtalks : np.ndarray
        The cross-talk matrix. xtalks[pixel]
    """
    n_pixels = len(p[0])
    n_ifos = len(p)
    _o = float32(1.e-12)

    q_norm = np.zeros((n_ifos, n_pixels))
    norm = np.zeros(n_ifos)
    rn = np.zeros(n_pixels)
    for i in range(n_pixels):
        if mk[i] <= 0.:
            continue
        xtalk_range = xtalks_lookup[i]
        xtalk = xtalks[xtalk_range[0]:xtalk_range[1]]
        xtalk_indexes = xtalk[:,0].astype(np.int32)
        xtalk_cc = np.vstack((xtalk[:,4], xtalk[:,5], xtalk[:,6], xtalk[:,7]))  # 4xM matrix
        # Select elements from p and q based on xtalk_indexes
        p_vec = p[:, xtalk_indexes]  # N*M matrix
        q_vec = q[:, xtalk_indexes]  # N*M matrix
        # Compute the sums using a vectorized approach
        # x = np.sum(xtalk_cc * np.array([q_vec, p_vec, q_vec, p_vec]), axis=1)  # 4-d vector

        # h = x * np.array([q[:, i], p[:, i], q[:, i], p[:, i]])
        x = np.vstack((np.dot(p_vec, xtalk_cc[0].T),
                      np.dot(p_vec, xtalk_cc[1].T),
                      np.dot(q_vec, xtalk_cc[2].T),
                      np.dot(q_vec, xtalk_cc[3].T)))  # 4xN matrix

        # Summing all components together
        t = (x[0] * p[:, i]) + (x[1] * q[:, i]) + (x[2] * p[:, i]) + (x[3] * q[:, i])

        # if i == 0:
        #     print('xtalk_cc: ', xtalk_cc)
        #     print('xtalk: ', xtalk[0], xtalk[4], xtalk[5], xtalk[6], xtalk[7])
        #     print('x: ', x)
        #     print('t: ', t)
        #     print('p[0, i]: ', p[0, i])
        #     print('q[0, i]: ', q[0, i])


        # set t to 0 if t < 0
        norm += np.where(t < 0, 0, t)

        e = (p[:, i] * p[:, i] + q[:, i] * q[:, i]) / (t + _o)  # 1-d vector

        q_norm[:, i] = np.where(e > 1, 0, e)

        u = x[0] + x[2]
        v = x[1] + x[3]
        rn[i] = np.sum(u * u + v * v)

    # print('q: ', norm)
    e = q_E * 2   # TF-Domain SNR
    norm = np.where(norm < float32(2.), float32(2.), norm)  # set norm to 2 if norm < 2
    # print('norm: ', norm)
    # print('e: ', e)
    detector_snr = e / norm  # detector {0:NIFO} SNR

    return detector_snr, norm, rn, q_norm


@njit(cache=True)
def gw_norm_numpy(data_norm, q_norm, p_E, ec):
    n_pixels = len(data_norm[0])
    n_ifos = len(data_norm)

    norm = np.zeros(n_ifos)
    p_norm = np.zeros(n_ifos)

    signal_norm = np.zeros((n_ifos, n_pixels))
    for i in range(n_ifos):
        norm[i] = q_norm[i]  # get data norms
        p_norm[i] = norm[i]  # save norms
        e = p_E[i] * 2  # TF-Domain SNR
        norm[i] = e / norm[i]  # detector {0:NIFO} SNR
        signal_norm[i] = np.where(ec > 0, data_norm[i], 0)  # set signal norms

    return norm, signal_norm


@njit
def orthogonalize_and_rotate(p, q, pAVX, length):
    event_mask = pAVX[1]
    rotation_sin = pAVX[4]
    rotation_cos = pAVX[5]
    first_component_energy = pAVX[15]
    second_component_energy = pAVX[16]
    energy_accumulated_first = np.zeros_like(p[0])
    energy_accumulated_second = np.zeros_like(q[0])

    for i in range(0, length, 4):
        accumulated_p_square = np.zeros_like(p[0])
        accumulated_q_square = np.zeros_like(q[0])
        accumulated_pq_product = np.zeros_like(p[0])

        for j in range(8):
            partial_p = p[j][i:i+4]
            partial_q = q[j][i:i+4]

            accumulated_p_square += partial_p * partial_p
            accumulated_q_square += partial_q * partial_q
            accumulated_pq_product += partial_p * partial_q

        event_occurance = (event_mask[i//4] > 0.0) * 1.0
        rotation_sin[i//4] = accumulated_pq_product * 2.0
        rotation_cos[i//4] = accumulated_p_square - accumulated_q_square

        total_energy = accumulated_p_square + accumulated_q_square + 1.e-21
        cos_square = rotation_cos[i//4]**2
        sin_square = rotation_sin[i//4]**2
        cos_sin_norm = np.sqrt(cos_square + sin_square)

        first_component_energy[i//4] = (total_energy + cos_sin_norm) / 2.0
        second_component_energy[i//4] = (total_energy - cos_sin_norm) / 2.0

        cos_divided = rotation_cos[i//4] / (cos_sin_norm + 1.e-21)
        sin_positive = (rotation_sin[i//4] > 0.0) * 1.0
        sin_value = 2.0 * sin_positive - 1.0

        rotation_sin[i//4] = np.sqrt((1.0 - cos_divided) / 2.0)
        rotation_cos[i//4] = np.sqrt((1.0 + cos_divided) / 2.0) * sin_value

        energy_accumulated_first += event_occurance * first_component_energy[i//4]
        energy_accumulated_second += event_occurance * second_component_energy[i//4]

    pAVX[1] = event_mask
    pAVX[4] = rotation_sin
    pAVX[5] = rotation_cos
    pAVX[15] = first_component_energy
    pAVX[16] = second_component_energy

    return np.sum(energy_accumulated_first) + np.sum(energy_accumulated_second), pAVX
