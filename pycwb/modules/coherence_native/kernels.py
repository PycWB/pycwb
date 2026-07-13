"""Numba kernels used by native coherence selection and clustering."""

from __future__ import annotations

import numpy as np
from numba import njit


# ---------------------------------------------------------------------------
# Numba JIT helpers (compiled once on first call; cached to disk)
# ---------------------------------------------------------------------------


@njit(cache=True)
def _uf_find(parent: np.ndarray, x: int) -> int:
    """Find a union-find root with path compression."""
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


@njit(cache=True)
def _label_components_grid(
    f_arr: np.ndarray,
    t_arr: np.ndarray,
    n_freq: int,
    n_time: int,
    kf: int,
    kt: int,
) -> np.ndarray:
    """Label rectangular-connectivity components for sparse TF pixels.

    This is equivalent to building an undirected graph with an edge whenever
    ``abs(df_bin) <= kf`` and ``abs(dt_bin) <= kt``, then running connected
    components.  The grid index lets each pixel inspect only its bounded local
    neighbourhood.
    """
    n = len(f_arr)
    labels = np.empty(n, dtype=np.int64)
    if n == 0:
        return labels

    parent = np.arange(n, dtype=np.int64)
    index_map = np.full((n_freq, n_time), -1, dtype=np.int64)
    for i in range(n):
        f = f_arr[i]
        t = t_arr[i]
        if 0 <= f < n_freq and 0 <= t < n_time:
            old = index_map[f, t]
            if old >= 0:
                ri = _uf_find(parent, i)
                rj = _uf_find(parent, old)
                if ri != rj:
                    if ri < rj:
                        parent[rj] = ri
                    else:
                        parent[ri] = rj
            index_map[f, t] = i

    for i in range(n):
        f = f_arr[i]
        t = t_arr[i]
        if f < 0 or f >= n_freq or t < 0 or t >= n_time:
            continue

        f0 = f - kf
        if f0 < 0:
            f0 = 0
        f1 = f + kf + 1
        if f1 > n_freq:
            f1 = n_freq
        t0 = t - kt
        if t0 < 0:
            t0 = 0
        t1 = t + kt + 1
        if t1 > n_time:
            t1 = n_time

        for ff in range(f0, f1):
            for tt in range(t0, t1):
                j = index_map[ff, tt]
                if j >= 0 and j < i:
                    ri = _uf_find(parent, i)
                    rj = _uf_find(parent, j)
                    if ri != rj:
                        if ri < rj:
                            parent[rj] = ri
                        else:
                            parent[ri] = rj

    root_to_label = np.full(n, -1, dtype=np.int64)
    next_label = 1
    for i in range(n):
        root = _uf_find(parent, i)
        lbl = root_to_label[root]
        if lbl < 0:
            lbl = next_label
            root_to_label[root] = lbl
            next_label += 1
        labels[i] = lbl

    return labels


@njit(cache=True)
def _candidate_passes_support_numba(
    combined: np.ndarray,
    fi: int,
    t: int,
    ii: int,
    eo: float,
    em: float,
    eh: float,
) -> bool:
    """Return True when one clipped support-map pixel passes cWB selection."""
    e_val = combined[fi, t]
    if e_val < eo:
        return False

    ct = combined[fi + 1, t] + combined[fi, t + 1] + combined[fi + 1, t + 1]
    cb = combined[fi - 1, t] + combined[fi, t - 1] + combined[fi - 1, t - 1]

    ht = combined[fi + 1, t + 2]
    if fi < ii:
        ht += combined[fi + 2, t + 2] + combined[fi + 2, t + 1]

    hb = combined[fi - 1, t - 2]
    if fi >= 2:
        hb += combined[fi - 2, t - 2] + combined[fi - 2, t - 1]

    return not (
        (ct + cb) * e_val < eh
        and (ct + ht) * e_val < eh
        and (cb + hb) * e_val < eh
        and e_val < em
    )


@njit(cache=True)
def _subnet_subrho_numba(
    asnr_arr: np.ndarray, noise_rms_arr: np.ndarray, n_sub: float
) -> tuple[float, float]:
    """Compute subnet and subrho statistics for one cluster.

    Parameters
    ----------
    asnr_arr      : float64[n_pix, n_ifo]
    noise_rms_arr : float64[n_pix, n_ifo]
    n_sub         : float  — threshold constant 2 * iGamma^{-1}(n_ifo-1, 0.314)

    Returns
    -------
    subnet : float
    subrho : float
    """
    n_pix = asnr_arr.shape[0]
    n_ifo = asnr_arr.shape[1]
    rho = 0.0
    e_sub_total = 0.0
    e_max_sum = 0.0
    subnet_acc = 0.0
    for p in range(n_pix):
        amp_sum = 0.0
        e_max = 0.0
        e_tot = 0.0
        nsd = 0.0
        msd = 0.0
        for d in range(n_ifo):
            amp = asnr_arr[p, d]
            x = amp * amp
            v = noise_rms_arr[p, d] * noise_rms_arr[p, d]
            amp_sum += abs(amp)
            if x > e_max:
                e_max = x
                msd = v
            e_tot += x
            if v > 0.0:
                nsd += 1.0 / v
        a_fac = (e_tot / (amp_sum * amp_sum)) if amp_sum > 0.0 else 1.0
        rho += (1.0 - a_fac) * (e_tot - n_sub * 2.0)
        y_val = e_tot - e_max
        x_corr = y_val * (1.0 + y_val / (e_max + 1.0e-5))
        if msd > 0.0:
            nsd -= 1.0 / msd
        v_corr = (2.0 * e_max - e_tot) * msd * nsd / 10.0
        e_sub_total += e_tot - e_max
        e_max_sum += e_max
        a_cut = x_corr / (x_corr + n_sub) if (x_corr + n_sub) != 0.0 else 0.0
        denom = x_corr + (v_corr if v_corr > 0.0 else 1.0e-5)
        subnet_acc += (e_tot * x_corr / denom) * (a_cut if a_cut > 0.5 else 0.0)
    subnet = subnet_acc / (e_max_sum + e_sub_total + 0.01)
    subrho = np.sqrt(rho) if rho >= 0.0 else np.nan
    return subnet, subrho


@njit(cache=True)
def _subnet_subrho_batch_numba(
    asnr_all: np.ndarray,
    noise_rms_all: np.ndarray,
    offsets: np.ndarray,
    n_sub: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute subnet and subrho for all clusters in one Numba pass.

    Parameters
    ----------
    asnr_all      : float64[n_pix_total, n_ifo]  — flattened across all clusters
    noise_rms_all : float64[n_pix_total, n_ifo]
    offsets       : int64[n_clusters + 1]  — CSR row-pointer style start/end per cluster
    n_sub         : float  — 2 * iGamma^{-1}(n_ifo-1, 0.314), same for all clusters

    Returns
    -------
    subnet_arr : float64[n_clusters]
    subrho_arr : float64[n_clusters]
    """
    n_clusters = len(offsets) - 1
    subnet_arr = np.empty(n_clusters, dtype=np.float64)
    subrho_arr = np.empty(n_clusters, dtype=np.float64)
    for c in range(n_clusters):
        s = offsets[c]
        e = offsets[c + 1]
        subnet_arr[c], subrho_arr[c] = _subnet_subrho_numba(
            asnr_all[s:e], noise_rms_all[s:e], n_sub
        )
    return subnet_arr, subrho_arr


@njit(cache=True)
def _align_threshold_map_numba(
    arrays_stack: np.ndarray,
    shift_bins: np.ndarray,
    valid_start: int,
    nn_valid: int,
    veto: np.ndarray,
    has_veto: bool,
    edge_bins: int,
    ib: int,
    ie: int,
    eo: float,
    em: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Align detectors and build the clipped support map used for selection.

    Parameters
    ----------
    arrays_stack : float64[n_ifo, n_freq, n_time]
    shift_bins   : int64[n_ifo]
    valid_start  : int
    nn_valid     : int
    veto         : int16[n_time] or empty  — 0 = reject, 1 = keep
    has_veto     : bool
    edge_bins    : int
    ib, ie       : int  — inclusive first / exclusive last valid freq indices
    eo, em       : float  — energy threshold, hard cap (2*eo)

    Returns
    -------
    combined  : float64[n_freq, n_time]  — clipped support map
    live_mask : bool[n_time]             — shifted-veto live output bins
    """
    n_ifo = arrays_stack.shape[0]
    n_freq = arrays_stack.shape[1]
    n_time = arrays_stack.shape[2]

    combined = np.zeros((n_freq, n_time), dtype=np.float64)
    live_mask = np.zeros(n_time, dtype=np.bool_)

    for t in range(valid_start, valid_start + nn_valid):
        u = t - valid_start
        live = True
        if has_veto:
            for d in range(n_ifo):
                src_t = valid_start + (u + shift_bins[d]) % nn_valid
                if veto[src_t] == 0:
                    live = False
                    break
        live_mask[t] = live

    valid_stop = valid_start + nn_valid
    for fi in range(n_freq):
        if fi < ib:
            continue
        for t in range(valid_start, valid_stop):
            if has_veto and not live_mask[t]:
                continue
            u = t - valid_start
            v = 0.0
            for d in range(n_ifo):
                src_t = valid_start + (u + shift_bins[d]) % nn_valid
                v += arrays_stack[d, fi, src_t]

            if v < eo:
                continue
            elif v > em:
                combined[fi, t] = em + 0.1
            else:
                combined[fi, t] = v

    return combined, live_mask


@njit(cache=True)
def _select_candidates_numba(
    combined: np.ndarray,
    arrays_stack: np.ndarray,
    shift_bins: np.ndarray,
    valid_start: int,
    valid_stop: int,
    nn_valid: int,
    n_freq: int,
    edge_bins: int,
    ib: int,
    ie: int,
    eo: float,
    em: float,
    eh: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Emit selected pixels and per-detector payload directly from support map."""
    n_ifo = arrays_stack.shape[0]
    n_time = combined.shape[1]

    ii = n_freq - 2
    margin = max(edge_bins, 2)
    t_s = margin
    t_e = n_time - margin
    f_s = ib
    f_e = min(max(ie, f_s), n_freq - 1)

    selected = np.zeros((n_freq, n_time), dtype=np.bool_)

    n_pix = 0
    for fi in range(f_s, f_e):
        for t in range(t_s, t_e):
            if _candidate_passes_support_numba(combined, fi, t, ii, eo, em, eh):
                selected[fi, t] = True
                n_pix += 1

    freq_idx = np.empty(n_pix, dtype=np.int64)
    time_idx = np.empty(n_pix, dtype=np.int64)
    values = np.empty(n_pix, dtype=np.float64)
    pix_det_energy = np.empty((n_pix, n_ifo), dtype=np.float64)
    pix_det_index = np.empty((n_pix, n_ifo), dtype=np.int64)

    k = 0
    for fi in range(f_s, f_e):
        for t in range(t_s, t_e):
            if selected[fi, t]:
                freq_idx[k] = fi
                time_idx[k] = t

                raw_energy = 0.0
                for d in range(n_ifo):
                    if nn_valid > 0 and valid_start <= t < valid_stop:
                        u = t - valid_start
                        det_t = valid_start + (u + shift_bins[d]) % nn_valid
                    else:
                        det_t = t
                    e = arrays_stack[d, fi, det_t]
                    raw_energy += e
                    pix_det_energy[k, d] = e if e > 0.0 else 0.0
                    pix_det_index[k, d] = det_t * n_freq + fi

                values[k] = raw_energy
                k += 1

    return selected, freq_idx, time_idx, values, pix_det_energy, pix_det_index
