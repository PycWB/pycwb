"""Veto-mask and threshold helpers for native coherence."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import gammaincc as _scipy_gammaincc
from scipy.special import gammainccinv

from pycwb.types.time_frequency_map import TimeFrequencyMap


def build_veto_mask(
    tf_map, veto_windows: list[tuple[float, float]], edge: float | None = None
) -> np.ndarray:
    """
    Build a binary keep-mask from GPS time windows for a TF map timeline.

    Bins inside any of *veto_windows* are marked 1 (keep); everything else
    is 0 (reject).  The resulting array can be passed as the ``veto``
    argument to :func:`select_network_pixels`.

    Parameters
    ----------
    tf_map : object
        Reference TF map that defines the time axis (must expose ``data``,
        ``dt``, and ``start`` attributes).
    veto_windows : list[tuple[float, float]]
        GPS intervals ``(start, end)`` to keep.  Overlapping windows are
        handled correctly (union).
    edge : float or None
        Ignored for mask construction but accepted for API consistency.

    Returns
    -------
    np.ndarray
        1-D int16 array of length ``n_time`` (mask: 1=keep, 0=reject).
    """
    data = np.asarray(getattr(tf_map, "data", []))
    n_samples = int(data.shape[1]) if data.ndim == 2 else int(data.size)
    if n_samples <= 0:
        return np.zeros(0, dtype=np.int16)

    dt = float(getattr(tf_map, "dt", 0.0))
    if dt <= 0:
        raise ValueError("tf_map.dt must be positive for veto mask construction")

    rate = 1.0 / dt
    start = float(getattr(tf_map, "start", 0.0))
    stop = float(getattr(tf_map, "stop", start + n_samples * dt))

    mask = np.zeros(n_samples, dtype=np.int16)
    for seg_start, seg_end in veto_windows:
        s = min(max(float(seg_start), start), stop)
        e = min(max(float(seg_end), start), stop)
        jb = max(0, int((s - start) * rate))
        je = min(n_samples, int((e - start) * rate))
        if je > jb:
            mask[jb:je] = 1
    return mask


def _igamma_inv_upper(shape: float, p: float) -> float:
    """Inverse of the upper regularized incomplete gamma function.

    Returns x such that the upper regularized incomplete gamma
    Q(shape, x) = p, using scipy's ``gammainccinv``.
    """
    p = float(np.clip(p, 1.0e-12, 1.0 - 1.0e-12))
    s = float(max(shape, 1.0e-12))
    return float(gammainccinv(s, p))


def _get_tf_energy_array(tf_map: Any, edge: float | None = None) -> np.ndarray:
    """
    Return a real-valued TF energy array from a map-like object.

    :param tf_map: time-frequency map exposing `data` and optional `dt`
    :type tf_map: object
    :param edge: optional edge (seconds) cropped on both time sides
    :type edge: float | None
    :return: 2D or 1D float64 energy array
    :rtype: np.ndarray
    """
    arr = np.asarray(tf_map.data)
    if np.iscomplexobj(arr):
        arr = arr.real
    arr = np.asarray(arr, dtype=np.float64)

    if arr.ndim == 2 and edge is not None and hasattr(tf_map, "dt") and tf_map.dt > 0:
        e = int(max(0, round(float(edge) / float(tf_map.dt))))
        if e > 0 and arr.shape[1] > 2 * e:
            arr = arr[:, e:-e]
    return arr


def compute_threshold(
    tf_maps: list[TimeFrequencyMap],
    bpp: float,
    alp: float | None = None,
    edge: float | None = None,
) -> float:
    """
    Compute pixel energy threshold from time-frequency map statistics.

    Uses a Python-native implementation inspired by cWB ``network::THRESHOLD``
    logic based on the black-pixel probability (false-alarm rate).

    Parameters
    ----------
    tf_maps : list[TimeFrequencyMap]
        Per-detector time-frequency maps.
    bpp : float
        Black-pixel probability (target false-alarm rate).
    alp : float | None, optional
        Packet-shape scaling parameter (``None`` for the no-shape path).
    edge : float | None, optional
        Edge margin in seconds excluded from threshold statistics.

    Returns
    -------
    float
        Computed pixel energy threshold.
    """
    if not tf_maps or not hasattr(tf_maps[0], "data"):
        raise ValueError("compute_threshold requires TF maps")
    n_ifo = len(tf_maps)

    if alp is not None:
        # C++ THRESHOLD(p, shape) works on flat 1-D time-major data with
        # nL / nR indices, NOT on 2-D edge-cropped arrays.  Replicate that.
        pw0 = tf_maps[0]
        arr0 = np.asarray(pw0.data, dtype=np.float64)
        if np.iscomplexobj(arr0):
            arr0 = arr0.real
        M = int(arr0.shape[0]) if arr0.ndim == 2 else 1
        # C++ stores time-major: flat[t*M + m].  Python (M, T) -> transpose then ravel.
        if arr0.ndim == 2:
            flat0 = arr0.T.ravel()
        else:
            flat0 = arr0.ravel()
        w = flat0.copy()
        for tfm in tf_maps[1:]:
            a = np.asarray(tfm.data, dtype=np.float64)
            if np.iscomplexobj(a):
                a = a.real
            if a.ndim == 2:
                w += a.T.ravel()
            else:
                w += a.ravel()

        nL = int(float(edge or 0.0) * pw0.wavelet_rate * M)
        nR = int(w.size) - nL - 1  # C++: nR = pw->size() - nL - 1

        region = w[nL:nR]  # C++ loop: for(i=nL; i<nR; i++)
        region = np.clip(region, 0.0, n_ifo * 100.0)
        positive = region[region > 1.0e-3]
        if positive.size == 0:
            return 0.0

        avr = float(np.mean(positive))
        bbb = float(np.mean(np.log(positive)))
        alp_fit = np.log(avr) - bbb
        alp_fit = (
            3 - alp_fit + np.sqrt((alp_fit - 3) * (alp_fit - 3) + 24 * alp_fit)
        ) / (12 * alp_fit)
        bpp_corr = float(bpp) * alp_fit / float(alp)
        result = avr * _igamma_inv_upper(alp_fit, bpp_corr) / alp_fit / 2.0
        return result

    energies = [_get_tf_energy_array(tfm, edge=edge) for tfm in tf_maps]
    combined = np.sum(energies, axis=0)
    work = combined.ravel()
    if work.size == 0:
        return 0.0

    work = np.clip(work, 0.0, n_ifo * 100.0)
    positive = work[work > 1.0e-3]
    if positive.size == 0:
        return 0.0

    # CWB THRESHOLD(p) exact algorithm: iterative search for Gamma shape m
    # fff = fill fraction (fraction of pixels with energy > 0.0001, matching CWB wavecount)
    fff = float(np.sum(work > 1.0e-4) / work.size)
    if fff <= 0.0:
        return 0.0
    n_total = work.size
    sorted_work = np.sort(work)

    # CWB waveSplit(nL, nR, nR-k): returns (k+1)-th largest in range, i.e., sorted_work[n_total-k-1]
    k_val = int(float(bpp) * fff * n_total)
    k_med = int(0.2 * fff * n_total)
    val = (
        float(sorted_work[max(0, n_total - k_val - 1)])
        if k_val > 0
        else float(sorted_work[-1])
    )
    med = (
        float(sorted_work[max(0, n_total - k_med - 1)])
        if k_med > 0
        else float(sorted_work[-1])
    )

    # Find smallest m >= 1.0 (in 0.01 steps) where P(Gamma(N*m) >= med) >= 0.2
    # Matches CWB: while(p00<0.2) {p00 = 1-Gamma(N*m,med); m+=0.01;} if(m>1) m-=0.01;
    m = 1.0
    p00 = 0.0
    while p00 < 0.2:
        p00 = float(_scipy_gammaincc(n_ifo * m, med))
        m += 0.01
    if m > 1.01:
        m -= 0.01

    result = 0.3 * (_igamma_inv_upper(n_ifo * m, float(bpp)) + val) + n_ifo * np.log(m)
    return result


def apply_veto(
    tf_map: TimeFrequencyMap,
    tw: float,
    segment_list: list[tuple[float, float]] | None = None,
    injection_times: list[float] | None = None,
    edge: float | None = None,
    return_mask: bool = False,
) -> float | tuple[float, np.ndarray]:
    """
    Compute live time and optional veto mask from segments and injections.

    Parameters
    ----------
    tf_map : TimeFrequencyMap
        Reference TF map for timeline definition.
    tw : float
        Injection exclusion window in seconds.
    segment_list : list[tuple[float, float]] | None, optional
        Accepted live segments ``(start, stop)`` in GPS seconds.
    injection_times : list[float] | None, optional
        Injection GPS times to exclude within ±tw/2 seconds.
    edge : float | None, optional
        Edge margin in seconds excluded from live-time integration.
    return_mask : bool, optional
        If True, return ``(live_time, veto_mask)`` instead of just live time.

    Returns
    -------
    float or tuple[float, np.ndarray]
        Live time in seconds. If ``return_mask=True``, returns
        ``(live_time, veto_mask)`` where veto_mask is a 1-D int16 array.
    """
    data = np.asarray(getattr(tf_map, "data", []))
    if data.ndim == 2:
        n_samples = int(data.shape[1])
    else:
        n_samples = int(data.size)
    if n_samples <= 0:
        return (0.0, np.zeros(0, dtype=np.int16)) if return_mask else 0.0

    dt = float(getattr(tf_map, "dt", 0.0))
    if dt <= 0:
        raise ValueError("tf_map.dt must be positive for veto construction")

    rate = 1.0 / dt
    start = float(getattr(tf_map, "start", 0.0))
    stop = float(getattr(tf_map, "stop", start + n_samples * dt))

    veto = np.zeros(n_samples, dtype=np.int16)

    if not segment_list:
        veto[:] = 1
    else:
        for seg_start, seg_stop in segment_list:
            s = min(max(float(seg_start), start), stop)
            e = min(max(float(seg_stop), start), stop)
            jb = max(0, int((s - start) * rate))
            je = min(n_samples, int((e - start) * rate))
            if je > jb:
                veto[jb:je] = 1

    if injection_times:
        w = np.zeros_like(veto)
        tw = max(2.0, float(tw))
        half_window = int(tw * rate / 2.0 + 0.5)
        for gps in injection_times:
            j = int((float(gps) - start) * rate)
            jb = max(0, j - half_window)
            je = min(n_samples, j + half_window)
            if je - jb >= int(
                rate
            ):  # skip injection windows shorter than 1 s (likely GPS edge artefacts)
                w[jb:je] = 1
        veto = (veto * w).astype(np.int16)

    if edge is None:
        edge = 0.0
    n_edge = int(max(0, edge * rate + 0.5))
    if 2 * n_edge >= n_samples:
        live = 0.0
    else:
        live = float(np.sum(veto[n_edge : n_samples - n_edge])) / rate

    return (live, veto) if return_mask else live
