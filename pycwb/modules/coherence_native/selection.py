"""Network-pixel selection for native coherence."""

from __future__ import annotations

import numpy as np

from pycwb.types.time_frequency_map import TimeFrequencyMap

from .kernels import _align_threshold_map_numba, _select_candidates_numba
from .veto_threshold import _get_tf_energy_array


def _shift_bins_from_lag_shifts(
    lag_shifts: np.ndarray | list | None, n_ifo: int, rate: float
) -> np.ndarray:
    """Convert per-detector lag shifts in seconds to circular TF-bin shifts."""
    if lag_shifts is None:
        shifts_sec = np.zeros(n_ifo, dtype=float)
    else:
        shifts_sec = np.asarray(lag_shifts, dtype=float)
        if shifts_sec.size != n_ifo:
            raise ValueError("lag_shifts size mismatch with number of detectors")
    ref = float(np.min(shifts_sec)) if shifts_sec.size else 0.0
    return np.asarray(
        [int((float(s) - ref) * rate + 0.001) for s in shifts_sec], dtype=np.int64
    )


def _build_selection_cache(
    tf_maps: list[TimeFrequencyMap],
    edge: float = 0.0,
    lag_shifts_by_lag: np.ndarray | list | None = None,
) -> dict:
    """Build lag-invariant numeric inputs for :func:`select_network_pixels`."""
    if not tf_maps or not hasattr(tf_maps[0], "data"):
        raise ValueError("select_network_pixels requires TF maps")

    arrays = [_get_tf_energy_array(tfm, edge=None) for tfm in tf_maps]
    if not all(arr.ndim == 2 for arr in arrays):
        raise ValueError("python get_network_pixels expects 2D TF arrays")

    n_freq, n_time = arrays[0].shape
    for arr in arrays[1:]:
        if arr.shape != (n_freq, n_time):
            raise ValueError("all detector TF maps must have the same shape")

    dt = float(tf_maps[0].dt)
    if dt <= 0.0:
        raise ValueError("tf_map.dt must be positive")
    rate = 1.0 / dt

    edge_bins = int(max(0, float(edge) * rate + 0.001))
    valid_start = edge_bins
    valid_stop = n_time - edge_bins
    nn_valid = valid_stop - valid_start

    f_low = float(getattr(tf_maps[0], "f_low", 0.0) or 0.0)
    f_high_attr = getattr(tf_maps[0], "f_high", None)
    f_high = (
        float(f_high_attr)
        if f_high_attr is not None
        else float((n_freq - 1) * tf_maps[0].df)
    )
    df = float(getattr(tf_maps[0], "df", 0.0) or 0.0)

    ib = 1
    ie = n_freq
    if df > 0:
        freqs = np.arange(n_freq, dtype=np.float64) * df
        for idx_f, freq in enumerate(freqs):
            if freq <= f_high:
                ie = idx_f
            if freq <= f_low:
                ib = idx_f + 1
    ie = min(ie, n_freq - 1)
    ib = max(ib, 1)

    shift_bins_by_lag = None
    if lag_shifts_by_lag is not None:
        lag_shift_rows = list(lag_shifts_by_lag)
        shift_bins_by_lag = (
            np.vstack(
                [
                    _shift_bins_from_lag_shifts(lag_shifts, len(arrays), rate)
                    for lag_shifts in lag_shift_rows
                ]
            ).astype(np.int64, copy=False)
            if lag_shift_rows
            else np.empty((0, len(arrays)), dtype=np.int64)
        )

    tf0 = tf_maps[0]
    return {
        "arrays_stack": np.ascontiguousarray(
            np.stack(arrays, axis=0), dtype=np.float64
        ),
        "n_ifo": len(arrays),
        "n_freq": n_freq,
        "n_time": n_time,
        "dt": dt,
        "rate": rate,
        "edge_bins": edge_bins,
        "valid_start": valid_start,
        "valid_stop": valid_stop,
        "nn_valid": nn_valid,
        "ib": ib,
        "ie": ie,
        "start": float(getattr(tf0, "start", 0.0)),
        "stop": float(getattr(tf0, "stop", 0.0)),
        "f_low": f_low,
        "f_high": float(getattr(tf0, "f_high", (n_freq - 1) * df) or 0.0),
        "shift_bins_by_lag": shift_bins_by_lag,
    }


def select_network_pixels(
    tf_maps: list[TimeFrequencyMap],
    lag_index: int,
    energy_threshold: float,
    lag_shifts: np.ndarray | list | None = None,
    veto: np.ndarray | None = None,
    edge: float = 0.0,
    selection_cache: dict | None = None,
) -> dict:
    """
    Select significant pixels above energy threshold for one lag.

    Parameters
    ----------
    tf_maps : list[TimeFrequencyMap]
        Per-detector time-frequency maps.
    lag_index : int
        Zero-based lag index for time-delay selection.
    energy_threshold : float
        Pixel energy threshold for significance.
    lag_shifts : np.ndarray | list[float] | None, optional
        Per-detector time shifts in seconds for this lag.
    veto : np.ndarray | None, optional
        Binary veto array in time bins (1 keep, 0 reject).
    edge : float, optional
        Edge margin in seconds. Default is 0.0.
    selection_cache : dict or None, optional
        Lag-invariant numeric cache produced by :func:`_build_selection_cache`.
        When omitted, it is built from ``tf_maps`` for backwards-compatible
        direct calls.

    Returns
    -------
    dict
        Candidate payload with keys:

        - ``mask``           : bool ndarray, shape (n_freq, n_time) — pixel selection mask
        - ``time``           : int64 ndarray, shape (n_pix,) — time-bin indices of selected pixels
        - ``frequency``      : int64 ndarray, shape (n_pix,) — frequency-bin indices
        - ``energy``         : float64 ndarray, shape (n_pix,) — summed energy across detectors
        - ``pix_det_energy`` : float64 ndarray, shape (n_pix, n_ifo) — per-detector pixel energy
        - ``pix_det_index``  : int64 ndarray, shape (n_pix, n_ifo) — flat TF index per detector
        - ``rate``           : float — sample rate (Hz)
        - ``layers``         : int — number of WDM frequency layers
        - ``start``          : float — segment GPS start
        - ``stop``           : float — segment GPS stop
        - ``f_low``          : float | None — low-frequency edge
        - ``f_high``         : float | None — high-frequency edge
    """
    if selection_cache is None:
        selection_cache = _build_selection_cache(tf_maps, edge=edge)

    arrays_stack = selection_cache["arrays_stack"]
    n_ifo = int(selection_cache["n_ifo"])
    n_freq = int(selection_cache["n_freq"])
    n_time = int(selection_cache["n_time"])
    rate = float(selection_cache["rate"])

    shift_bins_by_lag = selection_cache.get("shift_bins_by_lag")
    if shift_bins_by_lag is not None and 0 <= int(lag_index) < len(shift_bins_by_lag):
        shift_bins_arr = np.asarray(shift_bins_by_lag[int(lag_index)], dtype=np.int64)
    else:
        shift_bins_arr = _shift_bins_from_lag_shifts(lag_shifts, n_ifo, rate)
    edge_bins = int(selection_cache["edge_bins"])
    valid_start = int(selection_cache["valid_start"])
    valid_stop = int(selection_cache["valid_stop"])
    nn_valid = int(selection_cache["nn_valid"])
    ib = int(selection_cache["ib"])
    ie = int(selection_cache["ie"])

    eo = float(energy_threshold)
    em = 2.0 * eo
    eh = em * em

    veto_arr = (
        veto.astype(np.int16, copy=False)
        if (veto is not None and len(veto) == n_time)
        else np.zeros(0, dtype=np.int16)
    )

    # Build the clipped support map, then emit sparse selected pixels directly.
    combined, live_mask = _align_threshold_map_numba(
        arrays_stack,
        shift_bins_arr,
        int(valid_start),
        int(nn_valid),
        veto_arr,
        (veto is not None and len(veto) == n_time),
        int(edge_bins),
        int(ib),
        int(ie),
        eo,
        em,
    )
    selected, freq_idx, time_idx, values, pix_det_energy, pix_det_index = (
        _select_candidates_numba(
            combined,
            arrays_stack,
            shift_bins_arr,
            int(valid_start),
            int(valid_stop),
            int(nn_valid),
            int(n_freq),
            int(edge_bins),
            int(ib),
            int(ie),
            eo,
            em,
            eh,
        )
    )

    return {
        "mask": selected,
        "time": time_idx,
        "frequency": freq_idx,
        "energy": values,
        "pix_det_energy": pix_det_energy,  # (n_pix, n_ifo) float64: energy per pixel per detector
        "pix_det_index": pix_det_index,  # (n_pix, n_ifo) int64: TF index per pixel per detector
        "rate": rate,
        "layers": n_freq,
        "start": float(selection_cache["start"]),
        "stop": float(selection_cache["stop"]),
        "f_low": float(selection_cache["f_low"]),
        "f_high": float(selection_cache["f_high"]),
        "live_mask": live_mask,
        "live_samples": int(np.sum(live_mask)),
    }
