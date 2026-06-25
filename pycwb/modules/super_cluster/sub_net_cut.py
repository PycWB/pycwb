import numpy as np
import time
# from numba import float32
from numba import njit
from numpy import float32
from pycwb.modules.likelihoodWP.dpf import dpf_np_loops_vec
from pycwb.modules.xtalk.monster import getXTalk_pixels_fast
from pycwb.modules.xtalk.type import XTalk
from pycwb.modules.likelihoodWP.likelihood import load_data_from_pixels


def sub_net_cut(
    pixels, ml, FP, FX, acor, e2or, n_ifo, n_sky,
    subnet, subcut, subnorm, subrho, xtalk: XTalk,
    timing: dict | None = None,
):
    """
    This function is used to cut the subnet from the lag and return the new lag
    """
    from pycwb.types.pixel_arrays import PixelArrays

    if isinstance(pixels, PixelArrays):
        return sub_net_cut_from_pixel_arrays(
            pixels, None, ml, FP, FX, acor, e2or, n_ifo, n_sky,
            subnet, subcut, subnorm, subrho, xtalk,
            arrays_prepared=False,
            timing=timing,
        )

    t_total = time.perf_counter()
    n_pix = len(pixels)
    if n_pix == 0:
        _add_timing(timing, "total", time.perf_counter() - t_total)
        return _format_subnet_result(False, False, False, 0.0, 0.0, 0.0, 0.0, subnet, subrho, subnorm)

    t_stage = time.perf_counter()
    rms, td00, td90, _ = load_data_from_pixels(pixels, n_ifo)

    td00 = np.ascontiguousarray(np.transpose(td00.astype(np.float32), (2, 0, 1)))  # (ndelay, nifo, npix)
    td90 = np.ascontiguousarray(np.transpose(td90.astype(np.float32), (2, 0, 1)))  # (ndelay, nifo, npix)
    FP = np.ascontiguousarray(FP.T, dtype=np.float32)
    FX = np.ascontiguousarray(FX.T, dtype=np.float32)
    ml = np.ascontiguousarray(ml, dtype=np.int32)
    rms = np.ascontiguousarray(rms.T, dtype=np.float32)
    _add_timing(timing, "data_prep", time.perf_counter() - t_stage)

    result = _sub_net_cut_prepared_packets(
        rms, td00, td90, ml, FP, FX, acor, e2or, n_ifo, n_sky,
        subnet, subcut, subnorm, subrho, xtalk=xtalk, pixels=pixels,
        timing=timing,
    )
    _add_timing(timing, "total", time.perf_counter() - t_total)
    return result


def sub_net_cut_from_pixel_arrays(
    pixels,
    pixel_indices,
    ml,
    FP,
    FX,
    acor,
    e2or,
    n_ifo,
    n_sky,
    subnet,
    subcut,
    subnorm,
    subrho,
    xtalk: XTalk,
    arrays_prepared: bool = False,
    timing: dict | None = None,
):
    """Internal fast path that avoids building a sliced ``PixelArrays`` object."""
    t_total = time.perf_counter()
    if pixel_indices is None:
        rows = np.arange(len(pixels), dtype=np.int64)
    else:
        rows = np.asarray(pixel_indices, dtype=np.int64)

    n_pix = len(rows)
    if n_pix == 0:
        _add_timing(timing, "total", time.perf_counter() - t_total)
        return _format_subnet_result(False, False, False, 0.0, 0.0, 0.0, 0.0, subnet, subrho, subnorm)

    t_stage = time.perf_counter()
    rms, td00, td90 = _load_selected_pixel_arrays(pixels, rows)
    _add_timing(timing, "data_prep", time.perf_counter() - t_stage)

    if arrays_prepared:
        ml_p = np.ascontiguousarray(ml, dtype=np.int32)
        FP_p = np.ascontiguousarray(FP, dtype=np.float32)
        FX_p = np.ascontiguousarray(FX, dtype=np.float32)
    else:
        ml_p = np.ascontiguousarray(ml, dtype=np.int32)
        FP_p = np.ascontiguousarray(FP.T, dtype=np.float32)
        FX_p = np.ascontiguousarray(FX.T, dtype=np.float32)

    result = _sub_net_cut_prepared_packets(
        rms, td00, td90, ml_p, FP_p, FX_p, acor, e2or, n_ifo, n_sky,
        subnet, subcut, subnorm, subrho, xtalk=xtalk,
        layers=pixels.layers[rows], times=pixels.time[rows],
        timing=timing,
    )
    _add_timing(timing, "total", time.perf_counter() - t_total)
    return result


def _load_selected_pixel_arrays(pa, rows: np.ndarray):
    inv_rms = 1.0 / pa.noise_rms[:, rows].astype(np.float64)
    rms_pix = 1.0 / np.sqrt(np.sum(inv_rms ** 2, axis=0))
    rms = np.ascontiguousarray((inv_rms * rms_pix[np.newaxis, :]).T, dtype=np.float32)

    n_rows = len(pa.time) * pa._n_ifo
    if n_rows == 0 or len(pa.td_amp_flat) == 0:
        return (
            rms,
            np.zeros((0, pa._n_ifo, len(rows)), dtype=np.float32),
            np.zeros((0, pa._n_ifo, len(rows)), dtype=np.float32),
        )
    sizes = pa.td_amp_offsets[1: n_rows + 1] - pa.td_amp_offsets[:n_rows]
    tsize = int(sizes[0])
    if not np.all(sizes == tsize):
        raise ValueError(
            "td_amp vectors have non-uniform lengths; subnet cut requires dense TD amplitudes"
        )
    td00, td90 = _gather_selected_td_halves(
        pa.td_amp_flat, pa.td_amp_offsets, rows, pa._n_ifo, tsize // 2
    )
    return rms, td00, td90


def _get_xtalk_from_arrays(xtalk: XTalk, layers: np.ndarray, times: np.ndarray):
    if hasattr(xtalk, "get_xtalk_pixels_from_arrays"):
        return xtalk.get_xtalk_pixels_from_arrays(layers, times, True)
    pix_mat = np.column_stack([layers, times]).astype(np.int64)
    return getXTalk_pixels_fast(pix_mat, True, xtalk.layers, xtalk.coeff, xtalk.lookup_table)


def _add_timing(timing: dict | None, key: str, elapsed: float) -> None:
    if timing is not None:
        timing[key] = timing.get(key, 0.0) + float(elapsed)


@njit(cache=True)
def _gather_selected_td_halves(td_amp_flat, td_amp_offsets, rows, n_ifo, tsize2):
    n_pix = len(rows)
    td00 = np.empty((tsize2, n_ifo, n_pix), dtype=np.float32)
    td90 = np.empty((tsize2, n_ifo, n_pix), dtype=np.float32)
    for p in range(n_pix):
        pix_idx = int(rows[p])
        for i in range(n_ifo):
            row = pix_idx * n_ifo + i
            start = int(td_amp_offsets[row])
            for k in range(tsize2):
                td00[k, i, p] = td_amp_flat[start + k]
                td90[k, i, p] = td_amp_flat[start + tsize2 + k]
    return td00, td90


def _sub_net_cut_prepared_arrays(
    rms,
    td00,
    td90,
    td_energy,
    cluster_xtalk_lookup,
    cluster_xtalk,
    ml,
    FP,
    FX,
    acor,
    e2or,
    n_ifo,
    n_sky,
    subnet,
    subcut,
    subnorm,
    subrho,
):
    network_energy_threshold = np.float32(2 * acor * acor * n_ifo)
    n_pix = int(rms.shape[0])

    l_max, stat, Em, Am, lm, Vm, suball, EE = optimze_sky_loc(n_ifo, n_pix, n_sky, FP, FX, rms, td00, td90, td_energy,
                                                              ml, network_energy_threshold, e2or, subcut)

    submra, rHo, Eo, Lo, Ls, m = mra_statistics(n_ifo, n_pix, FP, FX, rms, td00, td90, td_energy, ml,
                                                      network_energy_threshold, e2or, subcut,
                                                      cluster_xtalk, cluster_xtalk_lookup, l_max)
    subnet_pass = min(suball, submra) > subnet
    subrho_pass = rHo > subrho
    subthr_pass = Em > subnorm * Eo

    return _format_subnet_result(subnet_pass, subrho_pass, subthr_pass, suball, submra, rHo, Em, subnet, subrho, subnorm, Eo)


def _sub_net_cut_prepared_packets(
    rms,
    td00,
    td90,
    ml,
    FP,
    FX,
    acor,
    e2or,
    n_ifo,
    n_sky,
    subnet,
    subcut,
    subnorm,
    subrho,
    xtalk: XTalk,
    pixels=None,
    layers=None,
    times=None,
    timing: dict | None = None,
):
    network_energy_threshold = np.float32(2 * acor * acor * n_ifo)
    n_pix = int(rms.shape[0])

    t_stage = time.perf_counter()
    l_max, stat, Em, Am, lm, Vm, suball, EE = optimze_sky_loc_from_td(
        n_ifo, n_pix, n_sky, FP, FX, rms, td00, td90,
        ml, network_energy_threshold, e2or, subcut,
    )
    _add_timing(timing, "sky_scan", time.perf_counter() - t_stage)

    # MRA feeds only the second subnet operand.  If suball already fails the
    # configured subnet threshold, min(suball, submra) cannot pass.
    if suball <= subnet:
        return _format_subnet_result(
            False, True, True, suball, 0.0, 0.0, Em, subnet, subrho, subnorm, 0.0
        )

    t_stage = time.perf_counter()
    if pixels is not None:
        cluster_xtalk_lookup, cluster_xtalk = xtalk.get_xtalk_pixels(pixels, True)
    else:
        cluster_xtalk_lookup, cluster_xtalk = _get_xtalk_from_arrays(xtalk, layers, times)
    _add_timing(timing, "xtalk", time.perf_counter() - t_stage)

    t_stage = time.perf_counter()
    submra, rHo, Eo, Lo, Ls, m = mra_statistics_from_td(
        n_ifo, n_pix, FP, FX, rms, td00, td90, ml,
        network_energy_threshold, e2or, subcut,
        cluster_xtalk, cluster_xtalk_lookup, l_max,
    )
    _add_timing(timing, "mra", time.perf_counter() - t_stage)

    subnet_pass = min(suball, submra) > subnet
    subrho_pass = rHo > subrho
    subthr_pass = Em > subnorm * Eo

    return _format_subnet_result(
        subnet_pass, subrho_pass, subthr_pass,
        suball, submra, rHo, Em, subnet, subrho, subnorm, Eo,
    )


def _format_subnet_result(
    subnet_pass,
    subrho_pass,
    subthr_pass,
    suball,
    submra,
    rHo,
    Em,
    subnet,
    subrho,
    subnorm,
    Eo=0.0,
):
    return {
        'subnet_passed': subnet_pass,
        'subrho_passed': subrho_pass,
        'subthr_passed': subthr_pass,
        'subnet_condition': f"min(suball = {suball:.4f}, submra = {submra:.4f}) > subnet = {subnet:.4f}",
        'subrho_condition': f"rho = {rHo:.4f} > subrho = {subrho:.4f}",
        'subthr_condition': f"Em = {Em:.4f} > (subnorm = {subnorm:.4f} * Eo = {Eo:.4f})"
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
    reduced_rms = np.empty((n_pix, n_ifo), dtype=float32)
    reduced_v00 = np.empty((n_pix, n_ifo), dtype=float32)
    reduced_v90 = np.empty((n_pix, n_ifo), dtype=float32)

    for l in range(n_sky):
        m = float32(0)  # pixels above threshold
        Eo = float32(0)  # total network energy
        Ls = float32(0)  # subnetwork energy
        Ln = float32(0)  # network energy above subnet threshold
        for j in range(n_pix):
            _rE = float32(0.0)
            for i in range(n_ifo):  # get pixel energy
                _rE += td_energy[ml[i, l] + offset, i, j]
            rNRG[j] = _rE  # store pixel energy
            _msk = float32(1.0) if rNRG[j] > network_energy_threshold else float32(0.0)  # E>En  0/1 mask
            m += _msk  # count pixels above threshold
            pNRG[j] = rNRG[j] * _msk  # zero sub-threshold pixels
            Eo += pNRG[j]
            for i in range(n_ifo):
                pNRG[j] = min(rNRG[j] - td_energy[ml[i, l] + offset, i, j], pNRG[j])  # subnetwork energy
            Ls += pNRG[j]  # subnetwork energy
            _msk = float32(1.0) if pNRG[j] > Es else float32(0.0)  # subnet energy > Es 0/1 mask
            Ln += rNRG[j] * _msk  # network energy

        Eo = Eo + float32(0.01)
        m = int(2 * m + 0.01)
        aa = float32(Ls * Ln / (Eo - Ls))
        # if l in [0, 22, 1000, 1860, 1967, 2000]: print("l = ", l); print("Ln = ", Ln, ", Eo = ", Eo, ", Ls = ", Ls, ", m = ", m)
        if subcut >= 0 and (aa - m) / (aa + m + float32(1e-16)) < subcut:
            continue

        m = 0
        Ls = Ln = Eo = float32(0.0)
        for j in range(n_pix):
            ee = float32(0.)
            for i in range(n_ifo):
                v00_ij = td00[ml[i, l] + offset, i, j]
                v90_ij = td90[ml[i, l] + offset, i, j]
                ee += v00_ij * v00_ij + v90_ij * v90_ij
            if ee < network_energy_threshold:
                continue

            em = float32(0.0)
            for i in range(n_ifo):
                reduced_rms[m, i] = rms[j, i]
                v00_ij = td00[ml[i, l] + offset, i, j]
                v90_ij = td90[ml[i, l] + offset, i, j]
                reduced_v00[m, i] = v00_ij
                reduced_v90[m, i] = v90_ij
                _em = v00_ij * v00_ij + v90_ij * v90_ij
                if _em > em:
                    em = _em
            m += 1

            Ls += ee - em
            Eo += ee  # subnetwork energy, network energy
            if ee - em > Es:
                Ln += ee  # network energy above subnet threshold

        if Eo <= 0:
            continue

        Lo = float32(0.0)
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
def optimze_sky_loc_from_td(n_ifo, n_pix, n_sky, FP, FX, rms, td00, td90, ml, network_energy_threshold, e2or,
                            subcut):
    Es = float32(2 * e2or)
    network_energy_threshold = float32(network_energy_threshold)
    offset = int(td00.shape[0] / 2)

    rNRG = np.zeros(n_pix, dtype=float32)
    pNRG = np.zeros(n_pix, dtype=float32)
    l_max = 0
    stat = float32(0.0)
    Em = float32(0.0)
    Am = float32(0.0)
    lm = 0
    Vm = 0
    suball = float32(0.0)
    EE = float32(0.0)
    AA_max = float32(0.0)
    reduced_rms = np.empty((n_pix, n_ifo), dtype=float32)
    reduced_v00 = np.empty((n_pix, n_ifo), dtype=float32)
    reduced_v90 = np.empty((n_pix, n_ifo), dtype=float32)

    for l in range(n_sky):
        m = float32(0)
        Eo = float32(0)
        Ls = float32(0)
        Ln = float32(0)
        for j in range(n_pix):
            _rE = float32(0.0)
            for i in range(n_ifo):
                delay_idx = ml[i, l] + offset
                v00_ij = td00[delay_idx, i, j]
                v90_ij = td90[delay_idx, i, j]
                _rE += v00_ij * v00_ij + v90_ij * v90_ij
            rNRG[j] = _rE
            _msk = float32(1.0) if rNRG[j] > network_energy_threshold else float32(0.0)
            m += _msk
            pNRG[j] = rNRG[j] * _msk
            Eo += pNRG[j]
            for i in range(n_ifo):
                delay_idx = ml[i, l] + offset
                v00_ij = td00[delay_idx, i, j]
                v90_ij = td90[delay_idx, i, j]
                detector_energy = v00_ij * v00_ij + v90_ij * v90_ij
                pNRG[j] = min(rNRG[j] - detector_energy, pNRG[j])
            Ls += pNRG[j]
            _msk = float32(1.0) if pNRG[j] > Es else float32(0.0)
            Ln += rNRG[j] * _msk

        Eo = Eo + float32(0.01)
        m = int(2 * m + 0.01)
        aa = float32(Ls * Ln / (Eo - Ls))
        if subcut >= 0 and (aa - m) / (aa + m + float32(1e-16)) < subcut:
            continue

        m = 0
        Ls = Ln = Eo = float32(0.0)
        for j in range(n_pix):
            ee = float32(0.)
            for i in range(n_ifo):
                delay_idx = ml[i, l] + offset
                v00_ij = td00[delay_idx, i, j]
                v90_ij = td90[delay_idx, i, j]
                ee += v00_ij * v00_ij + v90_ij * v90_ij
            if ee < network_energy_threshold:
                continue

            em = float32(0.0)
            for i in range(n_ifo):
                reduced_rms[m, i] = rms[j, i]
                delay_idx = ml[i, l] + offset
                v00_ij = td00[delay_idx, i, j]
                v90_ij = td90[delay_idx, i, j]
                reduced_v00[m, i] = v00_ij
                reduced_v90[m, i] = v90_ij
                _em = v00_ij * v00_ij + v90_ij * v90_ij
                if _em > em:
                    em = _em
            m += 1

            Ls += ee - em
            Eo += ee
            if ee - em > Es:
                Ln += ee

        if Eo <= 0:
            continue

        Lo = float32(0.0)
        _, f, F, _, _, _, _, _ = dpf_np_loops_vec(FP[l], FX[l], reduced_rms[:m, :])

        for j in range(m):
            Lo += sse_like_ps(f[j], F[j], reduced_v00[j], reduced_v90[j])

        AA = aa / (abs(aa) + abs(Eo - Lo) + 2 * m * (Eo - Ln) / Eo)
        ee = Ls * Eo / (Eo - Ls)
        em = abs(Eo - Lo) + 2 * m
        ee = ee / (ee + em)
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

    v00 = np.empty((n_ifo, n_pix), dtype=float32)  # pa
    v90 = np.empty((n_ifo, n_pix), dtype=float32)  # pA

    m = float32(0)  # pixels above threshold
    # Eo = float32(0)  # total network energy
    # Ls = float32(0)  # subnetwork energy
    # Ln = float32(0)  # network energy above subnet threshold
    for j in range(n_pix):
        _rE = float32(0.0)
        for i in range(n_ifo):  # get pixel energy
            delay_idx = ml[i, l_max] + offset
            _rE += td_energy[delay_idx, i, j]
            v00[i, j] = td00[delay_idx, i, j]
            v90[i, j] = td90[delay_idx, i, j]
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
    m = int(m)  # undoubled count of above-threshold pixels, matching C++ _sse_MRA_ps call
    # aa = float32(Ls * Ln / (Eo - Ls))

    xi, XI, _, _ = sse_MRA_ps(network_energy_threshold, m, rNRG,
                                    v00, v90, xtalks, xtalks_lookup)


    m = 0
    Ls = Ln = Eo = float32(0.0)
    reduced_rms = np.empty((n_pix, n_ifo), dtype=float32)
    reduced_v00 = np.empty((n_pix, n_ifo), dtype=float32)
    reduced_v90 = np.empty((n_pix, n_ifo), dtype=float32)
    for j in range(n_pix):
        ee = float32(0.)
        for i in range(n_ifo):
            ee += xi[i, j] * xi[i, j] + XI[i, j] * XI[i, j]
        if ee < network_energy_threshold:
            continue

        em = float32(0.0)
        for i in range(n_ifo):
            reduced_rms[m, i] = rms[j, i]
            reduced_v00[m, i] = xi[i, j]   # use MRA principal components, not original v00
            reduced_v90[m, i] = XI[i, j]   # use MRA principal components, not original v90
            _em = xi[i, j] * xi[i, j] + XI[i, j] * XI[i, j]
            if _em > em:
                em = _em
        m += 1

        Ls += ee - em
        Eo += ee  # subnetwork energy, network energy
        if ee - em > Es:
            Ln += ee  # network energy above subnet threshold

    Lo = float32(0.0)
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
def mra_statistics_from_td(n_ifo, n_pix, FP, FX, rms, td00, td90, ml,
                           network_energy_threshold, e2or, subcut, xtalks, xtalks_lookup, l_max):
    Es = float32(2 * e2or)
    network_energy_threshold = float32(network_energy_threshold)
    offset = int(td00.shape[0] / 2)

    rNRG = np.zeros(n_pix, dtype=float32)
    v00 = np.empty((n_ifo, n_pix), dtype=float32)
    v90 = np.empty((n_ifo, n_pix), dtype=float32)

    m = float32(0)
    for j in range(n_pix):
        _rE = float32(0.0)
        for i in range(n_ifo):
            delay_idx = ml[i, l_max] + offset
            v00_ij = td00[delay_idx, i, j]
            v90_ij = td90[delay_idx, i, j]
            _rE += v00_ij * v00_ij + v90_ij * v90_ij
            v00[i, j] = v00_ij
            v90[i, j] = v90_ij
        rNRG[j] = _rE
        _msk = float32(1.0) if rNRG[j] > network_energy_threshold else float32(0.0)
        m += _msk

    m = int(m)

    xi, XI, _, _ = sse_MRA_ps(network_energy_threshold, m, rNRG,
                              v00, v90, xtalks, xtalks_lookup)

    m = 0
    Ls = Ln = Eo = float32(0.0)
    reduced_rms = np.empty((n_pix, n_ifo), dtype=float32)
    reduced_v00 = np.empty((n_pix, n_ifo), dtype=float32)
    reduced_v90 = np.empty((n_pix, n_ifo), dtype=float32)
    for j in range(n_pix):
        ee = float32(0.)
        for i in range(n_ifo):
            ee += xi[i, j] * xi[i, j] + XI[i, j] * XI[i, j]
        if ee < network_energy_threshold:
            continue

        em = float32(0.0)
        for i in range(n_ifo):
            reduced_rms[m, i] = rms[j, i]
            reduced_v00[m, i] = xi[i, j]
            reduced_v90[m, i] = XI[i, j]
            _em = xi[i, j] * xi[i, j] + XI[i, j] * XI[i, j]
            if _em > em:
                em = _em
        m += 1

        Ls += ee - em
        Eo += ee
        if ee - em > Es:
            Ln += ee

    Lo = float32(0.0)
    _, f, F, _, _, _, _, _ = dpf_np_loops_vec(FP[l_max], FX[l_max], reduced_rms[:m, :])

    for j in range(m):
        Lo += sse_like_ps(f[j], F[j], reduced_v00[j], reduced_v90[j])

    submra = Ls * Eo / (Eo - Ls + float32(1e-16))
    submra /= abs(submra) + abs(Eo - Lo) + 2 * (m + 6)
    rHo = np.sqrt(Lo * Lo / (Eo + 2 * m + float32(1e-16)) / 2)
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
    EE = float32(0.0)  # extracted energy
    mam = np.zeros(n_ifo, dtype=float32)
    mAM = np.zeros(n_ifo, dtype=float32)

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
        E = float32(0.0)
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

        xtalk_start = int(xtalks_lookup[m, 0])
        xtalk_end = int(xtalks_lookup[m, 1])

        for j in range(xtalk_start, xtalk_end):
            n = int(xtalks[j, 0])
            if rNRG[n] > Eo:
                #  _sse_rotsub_ps(__m128* _u, float c, __m128* _v, float s, __m128* _a)
                #  calculate a -= u*c + v*s and return a*a
                # _sse_rotsub_ps(_m00,c[4],_m90,c[5],_a00+n*f)
                # _sse_rotsub_ps(_m00,c[6],_m90,c[7],_a90+n*f)
                rNRG[n] = 0
                for i in range(n_ifo):
                    a_00[i][n] -= mam[i] * xtalks[j, 4] + mAM[i] * xtalks[j, 5]
                    a_90[i][n] -= mam[i] * xtalks[j, 6] + mAM[i] * xtalks[j, 7]
                    rNRG[n] += a_00[i][n] * a_00[i][n] + a_90[i][n] * a_90[i][n]

        # store PC energy
        pp = float32(0.0)
        for i in range(n_ifo):
            pp += amp[i][m] * amp[i][m] + AMP[i][m] * AMP[i][m]
        pNRG[m] = pp

        k += 1
    # if DEBUG:
    #     print("k = ", k, ", K = ", K)

    return amp, AMP, rNRG, pNRG
