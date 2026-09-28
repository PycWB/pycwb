"""Hough-track chirp fitting and cluster chirp-statistic updates.

This is the netcluster::mchirp path. The distinct micropixel estimator lives
in chirp_micropixel.py. Both consume likelihood-populated cluster pixels.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange
from pycwb.types.network_cluster import Cluster

@njit(cache=True, parallel=True)
def _count_chirp_track_overlaps_numba(x, y, xerr, yerr, kk, m_vals):
    """Phase 1 of mchirp Hough transform: compute the max interval-overlap count
    for each mass value (independent → parallelised with prange).

    For each mass m the t-f locus is a line  y = sl*x + b  in (time, F^{-8/3})
    space.  Each pixel defines an error ellipse that, projected onto the b-axis,
    gives an interval [bmin, bmax].  The maximum number of overlapping intervals
    is the Hough vote count for that mass.

    Parameters
    ----------
    x, y, xerr, yerr : 1-D float64 arrays, length n_pts
        Pixel coordinates and their uncertainties.
    kk : float
        Pre-computed chirp-mass constant.
    m_vals : 1-D float64 array, length n_mass
        Mass grid to scan.

    Returns
    -------
    nsel_arr : 1-D int64 array, length n_mass
        Maximum overlap count per mass value.
    """
    n_mass = len(m_vals)
    n_pts = len(x)
    nsel_arr = np.zeros(n_mass, dtype=np.int64)

    for mi in prange(n_mass):
        m = m_vals[mi]
        sl = kk * np.abs(m) ** (5.0 / 3.0)
        if m > 0.0:
            sl = -sl

        Db = np.sqrt(2.0 * (sl * sl * xerr * xerr + yerr * yerr))
        bmin = y - sl * x - Db
        bmax = bmin + 2.0 * Db

        # Build flat endpoint list: opens (+1) followed by closes (-1)
        ep_val = np.empty(2 * n_pts, dtype=np.float64)
        ep_type = np.empty(2 * n_pts, dtype=np.float64)
        for i in range(n_pts):
            ep_val[i] = bmin[i]
            ep_type[i] = 1.0
            ep_val[n_pts + i] = bmax[i]
            ep_type[n_pts + i] = -1.0

        order = np.argsort(ep_val)

        # Walk sorted endpoints; track running overlap count
        cum = 0
        maxcum = 0
        for i in range(2 * n_pts):
            idx = order[i]
            cum += int(ep_type[idx])
            if cum > maxcum:
                maxcum = cum

        nsel_arr[mi] = maxcum

    return nsel_arr


@njit(cache=True)
def _fit_chirp_track_candidates_numba(x, y, xerr, yerr, wgt, kk, m_vals, cand_indices, nselmax, chi2_thr):
    """Phase 2 of mchirp Hough transform: fine b-grid search among candidate masses.

    For each candidate mass (those achieving *nselmax* votes in phase 1) the
    b-axis is scanned at step 0.0025 within segments that attain the maximum
    overlap.  The (m, b) pair minimising the likelihood-weighted mean chi2 is
    returned.

    Parameters
    ----------
    x, y, xerr, yerr, wgt : 1-D float64 arrays, length n_pts
    kk : float
    m_vals : 1-D float64 array (full mass grid)
    cand_indices : 1-D int64 array — indices into m_vals with nsel == nselmax
    nselmax : int
    chi2_thr : float

    Returns
    -------
    m0, b0 : float
        Best-fit chirp-mass slope and intercept.
    """
    n_pts = len(x)
    b_step = 0.0025
    chi2min = 1e100
    m0 = m_vals[cand_indices[0]]
    b0 = 0.0

    for jj in range(len(cand_indices)):
        mi = cand_indices[jj]
        m = m_vals[mi]
        sl = kk * np.abs(m) ** (5.0 / 3.0)
        if m > 0.0:
            sl = -sl

        # Per-pixel chi2 denominator
        eps = sl * sl * xerr * xerr + yerr * yerr

        # Recompute sorted endpoints (cheap — only a few candidate masses)
        Db = np.sqrt(2.0 * (sl * sl * xerr * xerr + yerr * yerr))
        bmin = y - sl * x - Db
        bmax = bmin + 2.0 * Db

        ep_val = np.empty(2 * n_pts, dtype=np.float64)
        ep_type = np.empty(2 * n_pts, dtype=np.float64)
        for i in range(n_pts):
            ep_val[i] = bmin[i]
            ep_type[i] = 1.0
            ep_val[n_pts + i] = bmax[i]
            ep_type[n_pts + i] = -1.0

        order = np.argsort(ep_val)
        sorted_val = np.empty(2 * n_pts, dtype=np.float64)
        sorted_type = np.empty(2 * n_pts, dtype=np.float64)
        for i in range(2 * n_pts):
            sorted_val[i] = ep_val[order[i]]
            sorted_type[i] = ep_type[order[i]]

        # Build cumulative-type array
        cum_types = np.empty(2 * n_pts, dtype=np.int64)
        cum = 0
        for i in range(2 * n_pts):
            cum += int(sorted_type[i])
            cum_types[i] = cum

        # Walk segments that achieve nselmax and scan b grid
        for k in range(2 * n_pts - 1):
            if cum_types[k] != nselmax:
                continue
            b_lo = sorted_val[k]
            b_hi = sorted_val[k + 1]
            if b_hi <= b_lo:
                continue

            n_b_steps = int((b_hi - b_lo) / b_step)
            if n_b_steps < 1:
                n_b_steps = 1

            for bi in range(n_b_steps + 1):
                b = b_lo + bi * b_step
                if b > b_hi:
                    b = b_hi

                chi2_sum = 0.0
                wgt_sum = 0.0
                for i in range(n_pts):
                    res = y[i] - sl * x[i] - b
                    chi2_val = res * res / eps[i]
                    if chi2_val <= chi2_thr:
                        chi2_sum += chi2_val * wgt[i]
                        wgt_sum += wgt[i]

                if wgt_sum > 0.0:
                    totchi = chi2_sum / wgt_sum
                    if totchi < chi2min:
                        chi2min = totchi
                        m0 = m
                        b0 = b

    return m0, b0


def update_chirp_mass_statistics(cluster: Cluster, xgb_rho_mode: bool = False, pat0: bool = False):
    """Python implementation of C++ netcluster::mchirp().

    Computes chirpEllip and chirpEfrac via Hough-transform + PCA ellipticity
    on the cluster's pixels (which must have .likelihood already set by
    populate_detection_statistics).  Updates cluster.cluster_meta.net_rho2 with
    rho1 = rho0 * chirpEllip * sqrt(chirpEfrac), matching netevent.cc line 977:
        rho[1] = pcd->netRHO * chirp[3] * sqrt(chirp[5])   (pat0=false branch)

    net_rho2 is only updated for original 2G mode (xgb_rho_mode=False)
    with pat0=False.  In XGB mode or pat0=True the value set by
    populate_detection_statistics is preserved (mirrors netevent.cc lines 974-981).
    """
    import math

    # --- C++ watconstants (same as in netcluster::mchirp, from constants.hh) ---
    G = 6.67259e-11  # WAT_G_SI: gravitational constant [N m^2 kg^-2]
    SM = 1.98892e30  # solar mass [kg]
    C = 299792458.0  # speed of light [m/s]
    Pi = math.pi
    sF = 128.0  # frequency scaling (units of 128 Hz)
    chi2_thr = 2.5  # default threshold

    kk = 256.0 * Pi / 5.0 * math.pow(G * SM * Pi / (C * C * C), 5.0 / 3.0)
    kk *= math.pow(sF, 8.0 / 3.0)

    # --- Collect pixels (vectorised — no per-pixel object construction) ---
    _pa = cluster.pixel_arrays
    _valid = (_pa.likelihood > 0.0) & (_pa.frequency > 0)

    _rate_v = _pa.rate[_valid].astype(float)
    _layers_v = _pa.layers[_valid].astype(float)
    _time_v = _pa.time[_valid].astype(float)
    _freq_v = _pa.frequency[_valid].astype(float)
    _lh_v = _pa.likelihood[_valid].astype(float)

    T_v = np.floor(_time_v / _layers_v) / _rate_v
    eT_v = (0.5 / _rate_v) * math.sqrt(2.0)

    F_raw_v = _freq_v * _rate_v / 2.0 / sF
    _pos = F_raw_v > 0.0
    T_v, eT_v, F_raw_v, _rate_v, _lh_v = (T_v[_pos], eT_v[_pos], F_raw_v[_pos], _rate_v[_pos], _lh_v[_pos])

    eF_v = (_rate_v / 4.0 / math.sqrt(3.0)) / sF
    eF_v *= 8.0 / 3.0 / np.power(F_raw_v, 11.0 / 3.0)
    F_t_v = 1.0 / np.power(F_raw_v, 8.0 / 3.0)

    np_pts = len(T_v)
    if np_pts < 5:
        return  # insufficient pixels — leave net_rho2 unchanged

    x = T_v
    y = F_t_v
    xerr = eT_v
    yerr = eF_v
    wgt = _lh_v

    # --- Hough transform: find mass(es) with maximum pixel-overlap ---
    maxM = 100.0
    stepM = 0.2
    m_vals = np.arange(-maxM, maxM + 1e-9, stepM)  # 1001 values

    # Phase 1: parallel Numba scan — O(n_mass * n_pts * log n_pts) with prange
    nsel_arr = _count_chirp_track_overlaps_numba(
        x.astype(np.float64),
        y.astype(np.float64),
        xerr.astype(np.float64),
        yerr.astype(np.float64),
        float(kk),
        m_vals.astype(np.float64),
    )

    nselmax = int(np.max(nsel_arr))
    cand_indices = np.where(nsel_arr == nselmax)[0].astype(np.int64)

    # Phase 2: fine b-grid search over candidate masses — tight Numba inner loop
    m0, b0 = _fit_chirp_track_candidates_numba(
        x.astype(np.float64),
        y.astype(np.float64),
        xerr.astype(np.float64),
        yerr.astype(np.float64),
        wgt.astype(np.float64),
        float(kk),
        m_vals.astype(np.float64),
        cand_indices,
        int(nselmax),
        float(chi2_thr),
    )

    # --- Compute Efrac ---
    sl = kk * math.pow(abs(m0), 5.0 / 3.0)
    if m0 > 0:
        sl = -sl

    eps = sl * sl * xerr * xerr + yerr * yerr
    residuals = y - sl * x - b0
    chi2_all = residuals * residuals / eps
    sel_mask = chi2_all <= chi2_thr

    totEn = float(np.sum(wgt))
    selEn = float(np.sum(wgt[sel_mask]))
    Efrac = selEn / totEn if totEn > 0.0 else 0.0

    # --- Filter to selected pixels and compute PCA ellipticity ---
    x_sel = x[sel_mask]
    y_sel = y[sel_mask]
    np_sel = len(x_sel)

    if np_sel >= 2:
        xcm = np.mean(x_sel)
        ycm = np.mean(y_sel)
        dx = x_sel - xcm
        dy = y_sel - ycm
        qxx = float(np.sum(dx * dx))
        qyy = float(np.sum(dy * dy))
        qxy = float(np.sum(dx * dy))

        sq_delta = math.sqrt((qxx - qyy) ** 2 + 4.0 * qxy * qxy)
        lam1 = math.sqrt((qxx + qyy + sq_delta) / 2.0)
        lam2_sq = (qxx + qyy - sq_delta) / 2.0
        lam2 = math.sqrt(max(lam2_sq, 0.0))
        denom = lam1 + lam2
        chirpEllip = abs(lam1 - lam2) / denom if denom > 0.0 else 0.0
    else:
        chirpEllip = 0.0

    # --- Update cluster metadata ---
    chrho = chirpEllip * math.sqrt(Efrac)
    rho1 = cluster.cluster_meta.net_rho * chrho

    # C++ netevent.cc lines 974-981:
    #   chrho = chirp[3] * sqrt(chirp[5])                    (always computed)
    #   if netRHO >= 0 (original 2G):
    #       rho[1] = pat0 ? netrho : netRHO * chrho           (only pat0=false uses chirp)
    #   else (XGB.rho0):
    #       rho[1] = netrho                                   (chirp result ignored)
    if not xgb_rho_mode and not pat0:
        cluster.cluster_meta.net_rho2 = rho1
    # else: net_rho2 already set correctly by populate_detection_statistics; do not overwrite

__all__ = [
    "update_chirp_mass_statistics",
]
