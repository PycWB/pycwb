from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LagPlan:
    n_lag: int
    lag_shifts: np.ndarray


def build_lag_plan_from_network(net, n_ifo):
    """
    Build lag schedule from a network-like object.

    Parameters
    ----------
    net : object
        Network-like object exposing `nLag` and `get_ifo(i).lagShift.data`.
    n_ifo : int
        Number of detectors.

    Returns
    -------
    LagPlan
        Lag schedule with `n_lag` and a `(n_lag, n_ifo)` shift matrix in seconds.
    """
    n_lag = int(getattr(net, "nLag", 1))
    lag_shifts = np.zeros((n_lag, int(n_ifo)), dtype=float)

    for det_idx in range(int(n_ifo)):
        ifo = net.get_ifo(det_idx)
        shifts = np.asarray(ifo.lagShift.data, dtype=float)
        if shifts.size < n_lag:
            lag_shifts[:shifts.size, det_idx] = shifts
        else:
            lag_shifts[:, det_idx] = shifts[:n_lag]

    return LagPlan(n_lag=n_lag, lag_shifts=lag_shifts)


def _get_segment_duration_seconds(tf_maps):
    tf0 = tf_maps[0]
    data_obj = getattr(tf0, "data", None)

    # Resolve timeline samples from several possible containers
    if data_obj is None:
        data = np.asarray(tf0) if tf0 is not None else np.array([])
    elif hasattr(data_obj, "data"):
        data = np.asarray(data_obj.data)
    else:
        data = np.asarray(data_obj)

    if data.ndim == 2:
        n_time = int(data.shape[1])
    else:
        n_time = int(data.size)

    dt = 0.0
    if hasattr(tf0, "dt"):
        dt = float(getattr(tf0, "dt"))
    elif hasattr(tf0, "delta_t"):
        dt = float(getattr(tf0, "delta_t"))
    elif hasattr(tf0, "sample_rate"):
        sr = float(getattr(tf0, "sample_rate"))
        dt = 1.0 / sr if sr > 0 else 0.0
    elif data_obj is not None and hasattr(data_obj, "dt"):
        dt = float(getattr(data_obj, "dt"))
    elif data_obj is not None and hasattr(data_obj, "delta_t"):
        dt = float(getattr(data_obj, "delta_t"))
    elif data_obj is not None and hasattr(data_obj, "sample_rate"):
        sr = float(getattr(data_obj, "sample_rate"))
        dt = 1.0 / sr if sr > 0 else 0.0

    if n_time <= 0 or dt <= 0:
        raise ValueError("tf_maps must provide positive-duration timeline via data and dt")
    return float(n_time) * dt


def _generate_extended_lag_ids(n_ifo, lag_size, lag_off, lag_max, lag_site=None, max_iter=10_000_000):
    """
    Generate integer lag IDs for extended-lag mode (lagMax > 0).

    This is a deterministic Python adaptation of cWB random extended-lag generation.
    """
    n_ifo = int(n_ifo)
    lag_size = int(max(1, lag_size))
    lag_off = int(max(0, lag_off))
    lag_max = int(max(0, lag_max))

    if lag_site is not None:
        lag_site = np.asarray(lag_site, dtype=int)
        if lag_site.size != n_ifo:
            raise ValueError("lag_site size mismatch with n_ifo")
        if np.any((lag_site < 0) | (lag_site >= n_ifo)):
            raise ValueError("lag_site values must be in [0, n_ifo)")

    target = lag_off + lag_size
    rng = np.random.default_rng(13)

    lag_ids = [tuple([0] * n_ifo)]
    seen = {lag_ids[0]}

    for _ in range(int(max_iter)):
        if len(lag_ids) >= target:
            break

        sampled = np.zeros(n_ifo, dtype=int)
        sampled[1:] = rng.integers(-lag_max, lag_max + 1, size=n_ifo - 1)

        if lag_site is None:
            ids = sampled.copy()
        else:
            ids = sampled[lag_site]

        check = True
        for i in range(n_ifo - 1, -1, -1):
            for j in range(i - 1, -1, -1):
                if lag_site is not None:
                    if lag_site[i] != lag_site[j] and ids[i] == ids[j]:
                        check = False
                else:
                    if ids[i] == ids[j]:
                        check = False

        if not check:
            continue

        if np.all(ids == 0):
            continue

        ids = ids - int(np.min(ids))
        key = tuple(int(x) for x in ids)
        if key in seen:
            continue
        seen.add(key)
        lag_ids.append(key)

    return lag_ids


def build_lag_plan_from_config(config, tf_maps):
    """
    Build lag schedule from Python config and TF maps without ROOT network state.

    Parameters
    ----------
    config : object
        Configuration exposing `nIFO`, `lagSize`, `lagStep`, `lagOff`, `lagMax`, `segEdge`, and optional `lagSite`.
    tf_maps : list
        Time-frequency maps used to infer segment duration.

    Returns
    -------
    LagPlan
        Lag schedule with `n_lag` and `(n_lag, n_ifo)` shift matrix in seconds.
    """
    if tf_maps is None or len(tf_maps) == 0:
        raise ValueError("build_lag_plan_from_config requires non-empty tf_maps")

    n_ifo = int(getattr(config, "nIFO", len(tf_maps)))
    lag_size = int(max(1, int(getattr(config, "lagSize", 1))))
    lag_step = float(getattr(config, "lagStep", 1.0))
    lag_off = int(max(0, int(getattr(config, "lagOff", 0))))
    lag_max = int(max(0, int(getattr(config, "lagMax", 0))))
    seg_edge = float(getattr(config, "segEdge", 0.0))
    lag_site = getattr(config, "lagSite", None)

    if lag_step <= 0:
        raise ValueError("lagStep must be positive")

    seg_duration = _get_segment_duration_seconds(tf_maps)
    lag_max_seg = int((seg_duration - 2.0 * seg_edge) / lag_step) - 1
    if lag_max_seg < 0:
        return LagPlan(n_lag=0, lag_shifts=np.zeros((0, n_ifo), dtype=float))

    if lag_max == 0:
        full_ids = [tuple([m] + [0] * (n_ifo - 1)) for m in range(lag_off, lag_off + lag_size)]
        selected_ids = full_ids
    else:
        full_ids = _generate_extended_lag_ids(
            n_ifo=n_ifo,
            lag_size=lag_size,
            lag_off=lag_off,
            lag_max=lag_max,
            lag_site=lag_site,
        )
        if lag_off >= len(full_ids):
            selected_ids = []
        else:
            selected_ids = full_ids[lag_off:lag_off + lag_size]

    valid = []
    for ids in selected_ids:
        arr = np.asarray(ids, dtype=int)
        if np.any(arr < 0) or np.any(arr > lag_max_seg):
            continue
        valid.append(arr)

    if len(valid) == 0:
        return LagPlan(n_lag=0, lag_shifts=np.zeros((0, n_ifo), dtype=float))

    lag_shifts = np.vstack(valid).astype(float) * lag_step
    return LagPlan(n_lag=int(lag_shifts.shape[0]), lag_shifts=lag_shifts)
