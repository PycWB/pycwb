from typing import List, Optional, Dict
from dataclasses import dataclass, asdict
import numpy as np

from numba import njit as _numba_njit


@_numba_njit(cache=True)
def _generate_extended_lag_ids_core(
        n_ifo, lag_size, lag_off, lag_max, lag_site, use_lag_site, max_iter
    ):
        """Numba-compiled inner loop for extended-lag ID generation.

        Uses MT19937 (np.random.seed(13)) instead of PCG64 — lag sequences
        differ from the pure-Python fallback for the same parameters.
        Polynomial hash encodes each (n_ifo,) int64 tuple as a single int64
        key; valid for ids[i] in [0, 2*lag_max] when base = 2*lag_max+1.
        """
        target = lag_off + lag_size
        base = np.int64(2 * lag_max + 1)

        # Pre-allocate result; slot 0 = zero lag (already zeros).
        result = np.zeros((target, n_ifo), dtype=np.int64)
        count = np.int64(1)

        seen = set()
        seen.add(np.int64(0))  # zero-lag key = 0

        np.random.seed(13)

        sampled = np.zeros(n_ifo, dtype=np.int64)
        ids = np.zeros(n_ifo, dtype=np.int64)

        for _ in range(max_iter):
            if count >= target:
                break

            sampled[0] = np.int64(0)
            for i in range(1, n_ifo):
                # Uniform in [-lag_max, lag_max] via [0, 2*lag_max+1) shift.
                sampled[i] = (
                    np.int64(np.random.randint(np.int64(0), np.int64(2 * lag_max + 1)))
                    - np.int64(lag_max)
                )

            if use_lag_site:
                for i in range(n_ifo):
                    ids[i] = sampled[lag_site[i]]
            else:
                for i in range(n_ifo):
                    ids[i] = sampled[i]

            # Uniqueness check: no two IFOs from different sites share the same shift.
            check = True
            for i in range(n_ifo - 1, -1, -1):
                if not check:
                    break
                for j in range(i - 1, -1, -1):
                    if use_lag_site:
                        if lag_site[i] != lag_site[j] and ids[i] == ids[j]:
                            check = False
                            break
                    else:
                        if ids[i] == ids[j]:
                            check = False
                            break
            if not check:
                continue

            # Skip all-zero (already in slot 0).
            all_zero = True
            for i in range(n_ifo):
                if ids[i] != np.int64(0):
                    all_zero = False
                    break
            if all_zero:
                continue

            # Normalize: shift so minimum is 0.
            min_val = ids[0]
            for i in range(1, n_ifo):
                if ids[i] < min_val:
                    min_val = ids[i]
            for i in range(n_ifo):
                ids[i] -= min_val

            # Polynomial hash: sum(ids[i] * base^i).
            key = np.int64(0)
            mult = np.int64(1)
            for i in range(n_ifo):
                key += ids[i] * mult
                mult *= base

            if key in seen:
                continue
            seen.add(key)

            for i in range(n_ifo):
                result[count, i] = ids[i]
            count += 1

        return result[:count]


@dataclass
class FrameFile:
    """
    Class to store the metadata of a frame file, which contains the ifo, the path, the start time, and the duration.

    Parameters
    ----------
    ifo: str
        name of the interferometer
    path: str
        path of the frame file
    start_time: float
        start time of the frame file
    duration: float
        duration of the frame file
    """
    ifo: str
    path: str
    start_time: float
    duration: float

    @property
    def end_time(self) -> float:
        """
        Get the end time of the frame file.

        Returns
        -------
        end_time: float
            end time of the frame file
        """
        return self.start_time + self.duration


@dataclass
class WaveSegment:
    """
    Class to store the metadata of a wave segment for analysis, which contains the index of the segment,
    the start and end time of the segment, and the list of frame files that are within the segment.

    Time-window conventions
    -----------------------
    ``start_time`` / ``end_time`` bound the **analysis window** — the portion
    of data that is actually searched for events.  Because the wavelet / WDM
    transforms require boundary padding, data is always loaded over a wider
    **padded window** ``[padded_start, padded_end]`` =
    ``[start_time − seg_edge, end_time + seg_edge]``.

    Use the named properties to avoid arithmetic in call sites:

    * ``analyze_start`` / ``analyze_end``   — analysis window (= start_time / end_time)
    * ``padded_start``  / ``padded_end``    — padded data window (± seg_edge)
    * ``duration``                          — analysis window length
    * ``padded_duration``                   — padded window length (= duration + 2*seg_edge)
    * ``physical_analyze_starts/ends``      — per-IFO analysis window (after superlag shift)
    * ``physical_padded_starts/ends``       — per-IFO padded window (after superlag shift ± seg_edge)

    Parameters
    ----------
    index: int
        index of the segment
    trial_idx: int
        trial index of the segment for injections, leave it 0 for no injections
    ifos: list of str
        list of interferometers
    analyze_start: float
        GPS start of the analysis window (excluding edge padding)
    analyze_end: float
        GPS end of the analysis window (excluding edge padding)
    sample_rate: float
        sample rate of the segment
    seg_edge: float
        wavelet boundary padding in seconds added on each side of the analysis window
    lag_size: int
        number of lags to generate (default 1)
    lag_step: float
        lag step size in seconds (default 1.0)
    lag_off: int
        first lag id offset (default 0, includes zero lag)
    lag_max: int
        maximum lag id for extended-lag mode (default 0, standard mode)
    lag_site: list of int, optional
        lag site assignment per IFO for extended-lag mode
    lag_array: list of list of float, optional
        user-provided lag shift matrix; overrides computed lags
    lag_file: str, optional
        path to a text file with lag shifts; overrides lag_array and computed lags
    shift: list, optional
        list of shifts for each interferometer, used for superlags
    channels: list, optional
        list of data channels for each interferometer
    frames: list, optional
        list of frame files that are within the segment
    noise: dict, optional
        The noise configurations that are within the segment
    injections: list, optional
        list of injections that are within the segment
    """
    index: int
    ifos: List[str]
    analyze_start: float
    analyze_end: float
    sample_rate: float
    seg_edge: float
    lag_size: int = 1
    lag_step: float = 1.0
    lag_off: int = 0
    lag_max: int = 0
    lag_site: Optional[List[int]] = None
    lag_array: Optional[List[List[float]]] = None
    lag_file: Optional[str] = None
    shift: Optional[List[float]] = None
    channels: Optional[List[str]] = None
    frames: Optional[List[FrameFile]] = None
    noise: Optional[Dict] = None
    injections: Optional[List[Dict]] = None
    trial_idx: int = 0
    veto_windows: list[tuple[float, float]] | None = None

    # ------------------------------------------------------------------
    # Analysis-window accessors (no edge padding)
    # ------------------------------------------------------------------

    @property
    def duration(self) -> float:
        """Duration of the analysis window in seconds (= analyze_end − analyze_start)."""
        return self.analyze_end - self.analyze_start

    def livetime(self, lag: int = 0) -> float:
        """Post-veto livetime of the analysis window for a given lag.

        For each IFO ``i`` with lag shift ``delta_i`` seconds, the data at
        analysis time ``t`` comes from GPS time ``t + delta_i``.  A GPS
        keep-window ``(s, e)`` therefore maps to the analysis-time interval
        ``(s - delta_i, e - delta_i)`` for that IFO.

        The effective livetime is the total duration where **all** IFOs are
        simultaneously inside their lag-adjusted keep windows, intersected
        with the analysis window ``[analyze_start, analyze_end]``.

        For the zero-lag (all shifts zero) this reduces to the plain overlap
        of *veto_windows* with the analysis window.  If *veto_windows* is
        not set the full :attr:`duration` is returned.

        Parameters
        ----------
        lag : int
            Row index into :attr:`lag_shifts` identifying the time-slide.

        Returns
        -------
        float
            Effective analysed duration in seconds after applying veto windows
            for the given lag.
        """
        if not self.veto_windows:
            return self.duration

        shifts = self.lag_shifts[lag]  # (n_ifo,) in seconds

        # Start with the full analysis window as the set of live intervals.
        # Iteratively intersect with each IFO's lag-adjusted keep windows.
        live: list[tuple[float, float]] = [(self.analyze_start, self.analyze_end)]

        for ifo_idx in range(len(self.ifos)):
            delta = float(shifts[ifo_idx])
            # GPS keep-window (s, e) → analysis-time (s - delta, e - delta)
            ifo_keep = sorted((s - delta, e - delta) for s, e in self.veto_windows)
            new_live: list[tuple[float, float]] = []
            for (ls, le) in live:
                for (ks, ke) in ifo_keep:
                    lo = max(ls, ks)
                    hi = min(le, ke)
                    if hi > lo:
                        new_live.append((lo, hi))
            live = new_live
            if not live:
                return 0.0

        return sum(e - s for s, e in live)

    # ------------------------------------------------------------------
    # Padded-window accessors (analysis window ± seg_edge)
    # ------------------------------------------------------------------

    @property
    def padded_start(self) -> float:
        """GPS start of the padded data window (= analyze_start − seg_edge).

        This is the actual start of the data that is read from frames /
        generated as noise.  All TimeSeries produced by the data-reading
        layer have ``start_time == padded_start``.
        """
        return self.analyze_start - self.seg_edge

    @property
    def padded_end(self) -> float:
        """GPS end of the padded data window (= analyze_end + seg_edge)."""
        return self.analyze_end + self.seg_edge

    @property
    def padded_duration(self) -> float:
        """Duration of the padded data window (= duration + 2 * seg_edge)."""
        return self.analyze_end - self.analyze_start + 2.0 * self.seg_edge

    # ------------------------------------------------------------------
    # Per-IFO physical times (after applying superlag shifts)
    # ------------------------------------------------------------------

    @property
    def physical_analyze_starts(self) -> Dict[str, float]:
        """Per-IFO GPS start of the analysis window (superlag shifts applied).

        For zero-lag or standard lags ``shift`` is ``None`` and every IFO
        returns ``start_time``.  For superlags, IFO *i* returns
        ``start_time − shift[i]``.
        """
        if self.shift is None:
            return {ifo: self.analyze_start for ifo in self.ifos}
        return {ifo: self.analyze_start - self.shift[i] for i, ifo in enumerate(self.ifos)}

    @property
    def physical_analyze_ends(self) -> Dict[str, float]:
        """Per-IFO GPS end of the analysis window (superlag shifts applied)."""
        if self.shift is None:
            return {ifo: self.analyze_end for ifo in self.ifos}
        return {ifo: self.analyze_end - self.shift[i] for i, ifo in enumerate(self.ifos)}

    @property
    def physical_padded_starts(self) -> Dict[str, float]:
        """Per-IFO GPS start of the padded data window (= physical_analyze_starts[ifo] − seg_edge).

        This is the correct epoch for any TimeSeries that spans the full
        padded window per IFO (e.g. the epoch base for ``Event.gps``).
        """
        return {ifo: t - self.seg_edge for ifo, t in self.physical_analyze_starts.items()}

    @property
    def physical_padded_ends(self) -> Dict[str, float]:
        """Per-IFO GPS end of the padded data window (= physical_analyze_ends[ifo] + seg_edge)."""
        return {ifo: t + self.seg_edge for ifo, t in self.physical_analyze_ends.items()}

    @property
    def physical_start_times(self) -> Dict[str, float]:
        """Deprecated alias for :attr:`physical_analyze_starts`."""
        return self.physical_analyze_starts

    @property
    def physical_end_times(self) -> Dict[str, float]:
        """Deprecated alias for :attr:`physical_analyze_ends`."""
        return self.physical_analyze_ends

    @property
    def n_lag(self) -> int:
        """Number of valid time lags for this segment."""
        return self.lag_shifts.shape[0]

    @property
    def lag_shifts(self) -> np.ndarray:
        """
        Lag shift matrix of shape ``(n_lag, n_ifo)`` in seconds.

        Resolution order:
        1. ``lag_file`` — load from a text file (one row per lag, columns = IFO shifts).
        2. ``lag_array`` — user-provided list of lag vectors.
        3. Computed from ``lag_size/lag_step/lag_off/lag_max/lag_site`` and segment geometry.
        """
        if self.lag_file is not None:
            return self._load_lag_file()
        if self.lag_array is not None:
            return np.asarray(self.lag_array, dtype=float)
        return self._compute_lag_shifts()

    def _load_lag_file(self) -> np.ndarray:
        """Load lag shifts from a whitespace-delimited text file."""
        data = np.loadtxt(self.lag_file, dtype=float, ndmin=2)
        n_ifo = len(self.ifos)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.shape[1] != n_ifo:
            raise ValueError(
                f"lag_file has {data.shape[1]} columns but segment has {n_ifo} IFOs"
            )
        return data

    def _compute_lag_shifts(self) -> np.ndarray:
        """Build the lag shift matrix from lag parameters and segment geometry."""
        n_ifo = len(self.ifos)
        seg_duration = float(self.duration)
        lag_step = float(self.lag_step)

        if lag_step <= 0:
            raise ValueError("lag_step must be positive")

        # CWB computes lagMaxSeg as int((tfmap_size/rate - 2*edge) / lagStep) - 1.
        # The TFmap spans the *padded* window (analysis + 2*edge), so
        # tfmap_size/rate - 2*edge == analysis_duration == seg_duration here.
        lag_max_seg = int(seg_duration / lag_step) - 1
        if lag_max_seg < 0:
            return np.zeros((0, n_ifo), dtype=float)

        if self.lag_max == 0:
            full_ids = [
                tuple([m] + [0] * (n_ifo - 1))
                for m in range(self.lag_off, self.lag_off + self.lag_size)
            ]
            selected_ids = full_ids
        else:
            _lag_site_arr = (
                np.asarray(self.lag_site, dtype=np.int64)
                if self.lag_site is not None
                else np.zeros(n_ifo, dtype=np.int64)
            )
            _result = _generate_extended_lag_ids_core(
                np.int64(n_ifo),
                np.int64(self.lag_size),
                np.int64(self.lag_off),
                np.int64(self.lag_max),
                _lag_site_arr,
                self.lag_site is not None,
                np.int64(10_000_000),
            )
            full_ids = [tuple(int(x) for x in row) for row in _result]
            if self.lag_off >= len(full_ids):
                selected_ids = []
            else:
                selected_ids = full_ids[self.lag_off : self.lag_off + self.lag_size]

        valid = []
        for ids in selected_ids:
            arr = np.asarray(ids, dtype=int)
            if np.any(arr < 0) or np.any(arr > lag_max_seg):
                continue
            valid.append(arr)

        if len(valid) == 0:
            return np.zeros((0, n_ifo), dtype=float)

        return np.vstack(valid).astype(float) * lag_step

    to_dict = asdict


@dataclass
class SLag:
    """
    Class to store the metadata of a SLag, which contains the job id, the slag id, and the segment id.

    Parameters
    ----------
    job_id: int
        job id
    slag_id: list[int]
        slag id vector, [0]=jobId - [1]=1/0 1=header slag - [2,..,nIFO+1] ifo slag
    seg_id: list[int]
        seg id vector, [0,..,nIFO-1] ifo segment number
    """
    job_id: int
    slag_id: List[int]
    seg_id: List[int]