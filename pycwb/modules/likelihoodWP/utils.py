import numpy as np
from numba import njit


def packet_norm(p, q, xtalks):
    pass


def packet_norm_numpy(p, q, xtalks):
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

    g = np.zeros(n_ifos)
    for i in range(n_pixels):
        xtalk = xtalks[i]
        xtalk_indexes = xtalk[0::8]
        xtalk_cc = np.array([xtalk[4::8], xtalk[5::8], xtalk[6::8], xtalk[7::8]])  # 4xM matrix

        # Select elements from p and q based on xtalk_indexes
        p_vec = p[:, xtalk_indexes]  # N*M matrix
        q_vec = q[:, xtalk_indexes]  # N*M matrix

        # Compute the sums using a vectorized approach
        # x = np.sum(xtalk_cc * np.array([q_vec, p_vec, q_vec, p_vec]), axis=1)  # 4-d vector

        # h = x * np.array([q[:, i], p[:, i], q[:, i], p[:, i]])
        x = np.array([np.dot(q_vec, xtalk_cc[0].T),
                      np.dot(q_vec, xtalk_cc[1].T),
                      np.dot(p_vec, xtalk_cc[2].T),
                      np.dot(p_vec, xtalk_cc[3].T)])  # 4xN matrix

        # Summing all components together
        t = (x[0] * q[:, i]) + (x[1] * p[:, i]) + (x[2] * q[:, i]) + (x[3] * p[:, i])
        # t = np.sum(h, axis=0)

        g += t

        e = (p[:, i] ** 2 + q[:, i] ** 2) / t  # 1-d vector
        # todo: what is Q = q[M+m]?
        Q = np.where(e > 1, 0, e)

        # update halo energy


def load_data(p, q, En, pAVX, I):
    # Assuming p, q, u, v are lists of NumPy arrays and pAVX is a list containing NumPy arrays
    # p, q are the input arrays; u, v are the output arrays; En is the energy threshold
    # I is the number of elements to process

    # Initialize arrays for total energy and masks
    total_energy = np.zeros_like(pAVX[0])
    mask = np.zeros_like(pAVX[1])

    nifo = len(p)
    # Process each set of arrays
    for idx in range(nifo):
        _p = p[idx][:I]
        _q = q[idx][:I]

        # Load data vectors into temporary arrays and compute energies
        energy_p = _p ** 2
        energy_q = _q ** 2
        total_energy[:I] += energy_p + energy_q

    # Apply energy threshold and initialize pixel mask
    mask[:I] = np.where(total_energy[:I] > En, 1, 0)

    # Calculate total energy and store in pAVX
    pAVX[0][:I] = total_energy[:I] * mask[:I]
    pAVX[1][:I] = mask[:I]

    # Compute and return the total energy above threshold
    total_energy_above_threshold = np.sum(pAVX[0][:I])
    return total_energy_above_threshold / 2


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


@njit
def avx_dpf_ps(Fp, Fx, l, pAPN, pAVX, I):
    # Prepare constants
    _0 = np.float32(0)
    _1 = np.float32(1)
    _2 = np.float32(2)
    _o = np.float32(0.0001)

    _NI = np.float32(0)
    _NN = np.float32(0)

    # Extract pointers
    fp_vec = pAVX[2]
    fx_vec = pAVX[3]
    sin_vec = pAVX[4]
    cos_vec = pAVX[5]
    ni_vec = pAVX[18]

    # Calculate sign
    sign = 0
    for k in range(8):
        if pAPN[k][2*I] > 0:
            sign += Fp[k][l] * Fx[k][l]
    sign = _1 if sign > 0 else -1
    _sign = np.full(I, sign, dtype=np.float32)

    # Main loop
    for i in range(I):
        ff_sum, FF_sum, fF_sum = 0, 0, 0
        for k in range(8):
            f_val = pAPN[k][i] * Fp[k][l]
            F_val = pAPN[k][I + i] * Fx[k][l]

            ff_sum += f_val ** 2
            FF_sum += F_val ** 2
            fF_sum += f_val * F_val

            pAPN[k][i] = f_val
            pAPN[k][I + i] = F_val

        rotation_term = 2 * fF_sum
        rotation_diff = ff_sum - FF_sum
        antenna_norm = ff_sum + FF_sum

        cos_term = rotation_diff ** 2
        sin_term = rotation_term ** 2
        norm_term = np.sqrt(cos_term + sin_term)

        fp_vec[i] = (antenna_norm + norm_term) / 2
        cos_divisor = rotation_diff / (norm_term + _o)
        sin_condition = rotation_term > _0

        sin_term = np.sqrt((_1 - cos_divisor) / 2)
        cos_term = np.sqrt((_1 + cos_divisor) / 2)
        cos_term *= 2 * sin_condition - _1

        sin_vec[i] = sin_term
        cos_vec[i] = cos_term

        for k in range(8):
            f_val = pAPN[k][i]
            F_val = pAPN[k][I + i]

            f_plus = f_val * cos_term + F_val * sin_term
            f_minus = F_val * cos_term - f_val * sin_term

            ni_term = f_plus ** 4
            fF_term = f_plus * f_minus

            pAPN[k][i] = f_plus
            pAPN[k][I + i] = f_minus

            ni_vec[i] += ni_term
            _NI += ni_term
            _NN += fF_term
    return ni_vec, _NI, _NN
