import numpy as np
from numba import njit

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
