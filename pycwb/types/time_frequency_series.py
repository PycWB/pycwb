import numpy as np
import warnings
from dataclasses import dataclass
from .wavelet import Wavelet
from .time_series import TimeSeries

class TimeFrequencySeries:
    """
    Class for storing a time-frequency series. This class is a Python wrapper for ROOT.WSeries
    
    .. deprecated:: 
        TimeFrequencySeries is obsolete. Use TimeFrequencyMap from wdm_wavelet instead.
        TimeFrequencyMap provides a cleaner interface and better integration with pure-Python WDM operations.

    :param data: data
    :type data: pycbc.types.timeseries.TimeSeries
    :param wavelet: wavelet method
    :type wavelet: WDM
    :param whiten_mode: whiten mode
    :type whiten_mode: int
    :param bpp: black pixel probability
    :type bpp: float
    :param w_rate: wavelet zero layer rate
    :type w_rate: float
    :param f_low: low frequency cutoff
    :type f_low: float
    :param f_high: high frequency cutoff
    :type f_high: float
    """
    __slots__ = ['_wavelet', 'data', 'whiten_mode', 'bpp', 'w_rate', '_f_low', '_f_high', '_wseries', '_ptr']

    def __init__(self, data=None, wavelet=None, whiten_mode=None, bpp=None, w_rate=None, f_low=None, f_high=None, 
                 wseries=None):
        # Issue deprecation warning
        warnings.warn(
            "TimeFrequencySeries is obsolete and will be removed in a future version. "
            "Use TimeFrequencyMap from wdm_wavelet.wdm instead for better Python-native support.",
            DeprecationWarning,
            stacklevel=2
        )
        
        self._wavelet = None
        #: Time series data
        self.data = data
        #: Wavelet method, a new wavelet will be copied and data will be allocated automatically
        self.wavelet = wavelet
        #: Whiten mode
        self.whiten_mode = 0 if whiten_mode is None else whiten_mode
        #: black pixel probability
        self.bpp = 1. if bpp is None else bpp
        #: wavelet zero layer rate
        self.w_rate = (data.sample_rate if data else 0.) if w_rate is None else w_rate
        #: low frequency cutoff
        self._f_low = f_low
        #: high frequency cutoff
        self._f_high = f_high
        #: WSeries object, used for data cleaning
        self._wseries = wseries

    def __dict__(self):
        return {key: getattr(self, key) for key in self.__slots__}

    def copy(self):
        new = TimeFrequencySeries(
            data=self.data.copy(),
            wavelet=self.wavelet,
            whiten_mode=self.whiten_mode,
            bpp=self.bpp,
            w_rate=self.w_rate,
            f_low=self.f_low,
            f_high=self.f_high,
        )

        return new

    def __del__(self):
        if self._wseries:
            self._wseries.resize(0)
            del self._wseries
        if self._wavelet:
            self._wavelet.release()
            del self._wavelet

    __copy__ = copy

    def forward(self, k=-1):
        """
        Performs forward wavelet transform on data (Not working yet)
        """
        if self.wavelet.allocate():
            self.wavelet.nSTS = self.wavelet.nWWS
            # was failed every second time to copy and forward a new time series because of
            # the pointer pWWS is not malloced, so it can not be reallocated, fixed by converting
            # the list to malloced memory
            self.wavelet.t2w(k)
            # if self.wavelet.pWWS != self.data or self.wavelet.nWWS != len(self.data):
            #     # TODO: implement and convert (or not)
            #     print(' ====== Implementation missing')
            #     #     self.data = self.wavelet.pWWS
            #     #     self.Size = self.wavelet.nWWS
            #     #     self.Slice = slice(0, self.wavelet.nWWS, 1)
            self.w_rate = float(self.wavelet.get_slice_size(0) / (self.stop - self.start))
        else:
            raise ValueError('Wavelet transform failed')

    @property
    def wavelet(self):
        """
        ROOT.WDM object
        """
        return self._wavelet

    @wavelet.setter
    def wavelet(self, value):
        if not value:
            self._wavelet = None

        if self._wavelet:
            self._wavelet.release()
            del self._wavelet

        self._wavelet = value.clone()
        self._wavelet.allocate(self.data)

    @property
    def start(self):
        """
        start time
        """
        return self.data.start_time

    @property
    def stop(self):
        """
        stop time
        """
        return self.data.end_time

    @property
    def edge(self):
        """
        TODO: dummy edge
        """
        return 0.0

    @property
    def sample_rate(self):
        """
        sample rate
        """
        return self.data.sample_rate

    @property
    def f_high(self):
        """
        high frequency cutoff
        """
        return self.sample_rate / 2 if self._f_high is None else self._f_high

    @f_high.setter
    def f_high(self, value):
        self._f_high = value

    @property
    def f_low(self):
        """
        low frequency cutoff
        """
        return 0.0 if self._f_low is None else self._f_low

    @f_low.setter
    def f_low(self, value):
        self._f_low = value

from wdm_wavelet.types.time_frequency_map import TimeFrequencyMap as WDMTimeFrequencyMap

@dataclass
class TimeFrequencyMap:
    """
    Data class for storing time-frequency map information.
    """
    data: np.ndarray
    is_whitened: bool
    dt: float
    df: float
    start: float
    stop: float
    # allow none
    f_low: float | None
    f_high: float | None
    edge: float | None
    wavelet: Wavelet  # Replace 'object' with the actual type of wavelet if available

    @classmethod
    def from_timeseries(cls, ts: TimeSeries, wavelet: Wavelet, 
                        is_whitened: bool = False,
                        f_low: float = None, f_high: float = None, edge: float = None):
        """
        Create a TimeFrequencyMap from a TimeSeries and wavelet.

        :param ts: Time series
        :type ts: TimeSeries
        :param wavelet: Wavelet object
        :type wavelet: Wavelet
        :param f_low: Low frequency cutoff
        :type f_low: float | None
        :param f_high: High frequency cutoff
        :type f_high: float | None
        :param edge: Edge parameter
        :type edge: float | None
        :return: TimeFrequencyMap object
        :rtype: TimeFrequencyMap
        """
        # Perform wavelet transform (placeholder, replace with actual implementation)
        # Here we assume a function `wavelet_transform` exists that performs the transform.
        # data = wavelet_transform(ts.data, wavelet)

        # Placeholder for transformed data, replace with actual transformed data
        data = wavelet.t2w(ts.data, ts.sample_rate, ts.t0)

        return cls(
            data=data.data,
            is_whitened=is_whitened,
            dt = data.dt,
            df = data.df,
            start = ts.t0,
            stop = ts.end_time,
            f_low=f_low,
            f_high=f_high,
            edge=edge,
            wavelet=wavelet
        )
    
    @property
    def timeseries(self):
        return self.wavelet.inverse(self.data)
    
    @property
    def wavelet_rate(self):
        return int(1.0 / self.dt)
    
    @property
    def size(self):
        return int(np.asarray(self.data).size)
    
    @property
    def maxLayer(self):
        return self.wavelet.M

    def wavecount(self, threshold, edge_length=None):
        """
        Count coefficients above a threshold.

        This is a Python-native compatibility helper for
        `ROOT.WaveArray::wavecount`.

        :param threshold: threshold value
        :type threshold: float
        :param edge_length: optional flattened edge width excluded on both sides
        :type edge_length: int | None
        :return: number of coefficients above the threshold
        :rtype: int
        """
        flat = np.asarray(self.data)
        if np.iscomplexobj(flat):
            flat = flat.real
        flat = flat.ravel()

        if edge_length is not None:
            n = int(edge_length)
            if n <= 0:
                return int(np.sum(flat > threshold))
            if 2 * n >= flat.size:
                return 0
            return int(np.sum(flat[n:-n] > threshold))
        return int(np.sum(flat > threshold))

    def wavesplit(self, start_index, end_index, split_index):
        """
        Return the order-statistics value in a flattened slice.

        Uses `np.partition` for efficiency and keeps Python compatibility with
        `ROOT.WaveArray::wavesplit` semantics.

        :param start_index: start index of the segment
        :type start_index: int
        :param end_index: end index of the segment
        :type end_index: int
        :param split_index: index to split the sorted segment
        :type split_index: int
        :return: value at the split index
        :rtype: float
        """
        flat = np.asarray(self.data)
        if np.iscomplexobj(flat):
            flat = flat.real
        flat = flat.ravel()

        section = flat[start_index:end_index]
        if section.size == 0:
            raise ValueError("wavesplit() empty input segment")
        split_index = int(max(0, min(split_index, section.size - 1)))
        parted = np.partition(section, split_index)
        value = parted[split_index]
        return value
    
    def Gamma2Gauss(self, hist=None):
        """
        Apply gamma-to-Gaussian style normalization on TF energy values.

        The transform updates `self.data` in place and optionally appends
        intermediate/final values into `hist`.

        :param hist: optional list-like accumulator for diagnostics
        :type hist: list | None
        :return: scaling pivot (`ALP`) used by the transform, or 0.0 on failure
        :rtype: float
        """
        original = np.asarray(self.data)
        shape = original.shape
        flat = original.real if np.iscomplexobj(original) else original
        flat = flat.ravel()

        if flat.size < 4:
            return 0.0

        nL = int(float(self.edge or 0.0) * self.wavelet_rate)
        nn = int(flat.size)
        nL = max(0, min(nL, nn - 2))
        nR = nn - nL
        if nR <= nL + 1:
            return 0.0

        work = flat[nL:nR]
        med = float(np.median(work))
        if med <= 0:
            return 0.0

        mask = (work > 0.01) & (work < 20 * med)
        valid_data = work[mask]
        if valid_data.size == 0:
            return 0.0

        aaa = np.sum(valid_data)
        bbb = np.sum(np.log(valid_data))
        count = valid_data.size

        alp = np.log(aaa / count) - bbb / count
        if alp <= 0:
            return 0.0
        alp = (3 - alp + np.sqrt((alp - 3) * (alp - 3) + 24 * alp)) / (12 * alp)

        avr = med * (3 * alp + 0.2) / (3 * alp - 0.8)
        ALP = med * alp / avr

        amp = flat * alp / avr
        transformed = np.where(amp < ALP, 0.0, amp - ALP * (1 + np.log(amp / ALP)))

        if hist is not None:
            hist.extend(transformed[nL:nR].tolist())

        core = transformed[nL:nR]
        core = core[core > 1.0e-5]
        if core.size == 0:
            return 0.0
        q68 = float(np.quantile(core, 0.6827))
        if q68 <= 0:
            return 0.0
        rms = 1.0 / q68
        transformed *= rms

        if hist is not None:
            hist.extend(np.sqrt(np.clip(transformed[nL:nR], 0.0, None)).tolist())

        if len(shape) == 2:
            self.data = transformed.reshape(shape)
        else:
            self.data = transformed

        return ALP


    def bandpass(self, f_low=None, f_high=None):
        """
        Set the frequency band for the time-frequency series.

        :param f_low: low frequency cutoff
        :type f_low: float
        :param f_high: high frequency cutoff
        :type f_high: float
        """
        if f_low is not None:
            self._f_low = f_low
        if f_high is not None:
            self._f_high = f_high

        pass

    def time_delay_max_energy(self, dt, downsample=1, pattern=0, hist=None):
        """
        Compute delayed max-energy map for a TF series.

        Python-native port of cWB `WSeries::maxEnergy`. The method updates
        `self.data` in place with the maximum delayed-pixel statistic over
        +/- time shifts in the range `[downsample, |dt| * rate]`.

        :param dt: max time delay in seconds
        :type dt: float
        :param downsample: delay step in samples (cWB `N`)
        :type downsample: int
        :param pattern: wave-packet pattern (cWB `pattern`)
        :type pattern: int
        :param hist: optional list-like container to collect transformed samples
        :type hist: list | None
        :return: gamma-to-Gauss scaling parameter (`ALP`) if `pattern != 0`, else `1.0`
        :rtype: float
        """
        if not hasattr(self.wavelet, "t2w") or not hasattr(self.wavelet, "w2t"):
            raise ValueError("time_delay_max_energy requires a WDM wavelet with t2w/w2t APIs")

        if downsample <= 0:
            raise ValueError("downsample must be >= 1")

        if not np.isfinite(dt):
            raise ValueError("dt must be finite")

        source_tf = WDMTimeFrequencyMap(
            data=np.asarray(self.data),
            df=float(self.df),
            dt=float(self.dt),
            t0=float(self.start),
            len_timeseries=max(1, int(round((self.stop - self.start) / self.dt))),
            wdm_params=dict(getattr(self.wavelet, "params", {})),
        )

        ts = self.wavelet.w2t(source_tf)
        ts_data = np.ascontiguousarray(np.asarray(ts.value), dtype=np.float64)
        sample_rate = float(ts.sample_rate.value)
        t0 = float(ts.t0.value)
        n_samples = int(ts_data.size)

        max_delay = int(sample_rate * abs(float(dt)))
        mm_mode = -1 if abs(int(pattern)) else 0

        def _time_slide_copy(data, length=0, src_idx=0, dst_idx=0):
            """Copy a contiguous time slice with zero-padded out-of-range behavior."""
            out = np.array(data, copy=True)
            if length == 0:
                length = min(len(data) - dst_idx, len(data) - src_idx)
            length = min(length, len(data) - dst_idx, len(data) - src_idx)
            if length > 0:
                out[dst_idx:dst_idx + length] = data[src_idx:src_idx + length]
            return out

        if abs(int(pattern)):
            base_tf = self.wavelet.t2w(ts_data, sample_rate=sample_rate, t0=t0, MM=mm_mode)
            current_max = np.zeros_like(base_tf.data.real, dtype=np.float64)

            packet_energy = self.wdm_packet(pattern, mode='e', coeffs=base_tf.data, return_map=True)
            current_max = np.maximum(current_max, packet_energy)

            for k in range(int(downsample), max_delay + 1, int(downsample)):
                if k >= n_samples:
                    break

                shifted = _time_slide_copy(ts_data, length=n_samples - k, src_idx=0, dst_idx=k)
                tmp_tf = self.wavelet.t2w(shifted, sample_rate=sample_rate, t0=t0, MM=mm_mode)
                current_max = np.maximum(current_max, self.wdm_packet(pattern, mode='e', coeffs=tmp_tf.data, return_map=True))

                shifted = _time_slide_copy(ts_data, length=n_samples - k, src_idx=k, dst_idx=0)
                tmp_tf = self.wavelet.t2w(shifted, sample_rate=sample_rate, t0=t0, MM=mm_mode)
                current_max = np.maximum(current_max, self.wdm_packet(pattern, mode='e', coeffs=tmp_tf.data, return_map=True))

            n_freq = current_max.shape[0]
            current_max[0, :] = 0.0
            current_max[n_freq - 1, :] = 0.0

            if abs(int(pattern)) in {5, 6, 9} and n_freq > 3:
                current_max[1, :] = 0.0
                current_max[n_freq - 2, :] = 0.0

            self.data = current_max
            return self.Gamma2Gauss(hist=hist)

        base_tf = self.wavelet.t2w(ts_data, sample_rate=sample_rate, t0=t0, MM=mm_mode)
        current_max_real = np.array(base_tf.data.real, copy=True)
        current_max_imag = np.array(base_tf.data.imag, copy=True)

        for k in range(int(downsample), max_delay + 1, int(downsample)):
            if k >= n_samples:
                break

            shifted = _time_slide_copy(ts_data, length=n_samples - k, src_idx=0, dst_idx=k)
            tmp_tf = self.wavelet.t2w(shifted, sample_rate=sample_rate, t0=t0, MM=mm_mode)
            current_max_real = np.maximum(current_max_real, tmp_tf.data.real)
            current_max_imag = np.maximum(current_max_imag, tmp_tf.data.imag)

            shifted = _time_slide_copy(ts_data, length=n_samples - k, src_idx=k, dst_idx=0)
            tmp_tf = self.wavelet.t2w(shifted, sample_rate=sample_rate, t0=t0, MM=mm_mode)
            current_max_real = np.maximum(current_max_real, tmp_tf.data.real)
            current_max_imag = np.maximum(current_max_imag, tmp_tf.data.imag)

        self.data = current_max_real + 1j * current_max_imag
        n_freq = self.data.shape[0]
        self.data[0, :] = 0.0
        self.data[n_freq - 1, :] = 0.0
        return 1.0

    def _compute_bounds(self, n_freq=None, n_time=None):
        """
        Compute flattened/time-frequency bounds used by packet operations.

        :param n_freq: number of frequency bins
        :type n_freq: int | None
        :param n_time: number of time bins
        :type n_time: int | None
        :return: `(jb, je, mL, mH)` flattened and frequency limits
        :rtype: tuple[int, int, int, int]
        """
        if n_freq is None or n_time is None:
            coeffs = np.asarray(self.data)
            if coeffs.ndim != 2:
                raise ValueError("_compute_bounds expects a 2D time-frequency map")
            n_freq, n_time = coeffs.shape

        M = int(n_freq)
        J = int(n_freq * n_time)
        edge = float(self.edge or 0.0)
        jb = int(edge * self.wavelet_rate / 4.0) * M
        if jb < 4 * M:
            jb = 4 * M
        je = J - jb
        df = self.df
        f_low = 0.0 if self.f_low is None else float(self.f_low)
        f_high = (df * (M - 1)) if self.f_high is None else float(self.f_high)
        mL = int(f_low / df + 0.1)
        mH = int(f_high / df + 0.1)
        mL = max(0, mL)
        mH = min(M - 1, mH)
        return jb, je, mL, mH
    
    def wdm_packet(self, pattern: int, mode: str = 'e', coeffs=None, return_map: bool = False):
        """
        Compute WDM packet energy/amplitude map for a given pattern.

        Vectorized NumPy implementation with non-wrapping edge handling.
        Uses `self.data` by default, or external `coeffs` if provided.

        :param pattern: packet pattern ID (compatible with cWB-style presets)
        :type pattern: int
        :param mode: output mode: `'e'` energy, `'l'` likelihood-like, `'a'` amplitude
        :type mode: str
        :param coeffs: optional complex TF coefficient map
        :type coeffs: np.ndarray | None
        :param return_map: if True, return computed map instead of shape scalar
        :type return_map: bool
        :return: packet shape (float) or computed map when `return_map=True`
        :rtype: float | np.ndarray
        """
        PATTERNS = {
            0: [],  # single pixel
            1: [(0, 1), (0, -1)],                         # "3|" vertical
            2: [(1, 0), (-1, 0)],                         # "3-" horizontal
            3: [(1, 1), (-1, -1)],                        # "3/" chirp
            4: [(1, -1), (-1, 1)],                        # "3\\" ringdown
            5: [(1, 1), (-1, -1), (2, 2), (-2, -2)],      # "5/"
            6: [(1, -1), (-1, 1), (2, -2), (-2, 2)],      # "5\\"
            7: [(0, 1), (0, -1), (1, 0), (-1, 0)],        # "3+"
            8: [(1, 1), (1, -1), (-1, 1), (-1, -1)],      # "3x"
            9: [(0, 1), (0, -1), (1, 0), (-1, 0),
                (1, 1), (1, -1), (-1, 1), (-1, -1)]      # "9*"
        }

        pattern = abs(int(pattern))
        mode = mode.lower()
        if mode not in {'e', 'l', 'a'}:
            raise ValueError("mode must be one of {'e', 'l', 'a'}")

        complex_map = np.asarray(self.data if coeffs is None else coeffs)
        if complex_map.ndim != 2:
            raise ValueError("wdm_packet expects a 2D time-frequency map")

        M, T = complex_map.shape
        jb, je, mL, mH = self._compute_bounds(M, T)

        # Determine shape/mean like earlier
        if pattern in (1, 3, 4):
            shape = mean = 3.0
            mL += 1; mH -= 1
        elif pattern == 2:
            shape = mean = 3.0
        elif pattern in (5, 6):
            shape = mean = 5.0
            mL += 2; mH -= 2
        elif pattern in (7, 8):
            shape = mean = 5.0
            mL += 1; mH -= 1
        elif pattern == 9:
            shape = mean = 9.0
            mL += 1; mH -= 1
        else:
            shape = mean = 1.0

        offsets = PATTERNS.get(pattern, [])
        center_weight = mean - 8.0

        # Initialize accumulators
        ee = np.zeros((M, T), dtype=float)
        EE = np.zeros((M, T), dtype=float)
        ss = np.zeros((M, T), dtype=float)

        # Center contribution (scaled)
        ee += (complex_map.real ** 2) * center_weight
        EE += (complex_map.imag ** 2) * center_weight
        ss += (complex_map.real * complex_map.imag) * center_weight

        # helper: add shifted contributions of offsets without wrap (zero outside)
        def add_shift(dr: int, dt: int):
            """Add contribution of shifted TF map by (dr, dt) into ee/EE/ss."""
            src_m0 = max(0, -dr)
            src_m1 = min(M, M - dr)
            src_t0 = max(0, -dt)
            src_t1 = min(T, T - dt)

            dst_m0 = src_m0 + dr
            dst_m1 = src_m1 + dr
            dst_t0 = src_t0 + dt
            dst_t1 = src_t1 + dt

            if src_m1 <= src_m0 or src_t1 <= src_t0:
                return  # nothing overlaps

            block = complex_map[src_m0:src_m1, src_t0:src_t1]
            # add squared real/imag and real*imag properly to destination slice
            ee[dst_m0:dst_m1, dst_t0:dst_t1] += np.real(block) ** 2
            EE[dst_m0:dst_m1, dst_t0:dst_t1] += np.imag(block) ** 2
            ss[dst_m0:dst_m1, dst_t0:dst_t1] += (np.real(block) * np.imag(block))

        for dr, dt in offsets:
            add_shift(dr, dt)

        # Now compute cc, ss2, nn arrays vectorized
        cc = ee - EE
        ss2 = ss * 2.0
        cc2 = cc

        # compute nn = sqrt(cc^2 + ss2^2), but ensure numeric stability
        nn = np.sqrt(cc2 * cc2 + ss2 * ss2)
        sum_eeEE = ee + EE
        # condition where sum_eeEE < nn -> nn = sum_eeEE
        mask = sum_eeEE < nn
        if mask.any():
            nn[mask] = sum_eeEE[mask]

        # compute aa elementwise safely
        a1 = np.sqrt(np.clip((sum_eeEE + nn) / 2.0, 0.0, None))
        a2 = np.sqrt(np.clip((sum_eeEE - nn) / 2.0, 0.0, None))
        aa = a1 + a2

        # compute em array
        if (mode == 'e') or (mode == 'l') or (mean == 1.0):
            em = sum_eeEE / 2.0
        else:
            em = (aa * aa) / 4.0

        alp = shape - np.log(shape) / 3.0 if shape > 0 else shape

        if mode == 'l':
            em = em * (shape / mean)
            # where em < alp, set to 0; otherwise apply the correction
            mask2 = em < alp
            em2 = em.copy()
            # avoid log of zero or negative
            pos_mask = ~mask2
            if pos_mask.any():
                em2[pos_mask] = em[pos_mask] - alp * (1.0 + np.log(em[pos_mask] / alp))
            em2[mask2] = 0.0
            em = em2

        # amplitude branch: compute amplitude complex array
        amplitudes = np.zeros((M, T), dtype=np.complex128)
        if mode == 'a':
            # avoid division by zero: where nn==0, amplitudes remain 0
            safe_nn = nn.copy()
            safe_nn[safe_nn == 0.0] = 1.0  # temporary to avoid division by zero
            cc_norm = cc2 / safe_nn
            ss_norm = ss2 / safe_nn
            denom = np.sqrt((1.0 + cc_norm) * (1.0 + cc_norm) + ss_norm * ss_norm) / 2.0
            # components:
            real_part = aa * cc_norm
            imag_part = aa * ss_norm / 2.0
            amplitudes = real_part + 1j * imag_part
            # fix locations where nn was zero to zero amplitude
            amplitudes[nn == 0.0] = 0+0j

        # apply frequency masks (mL..mH) and jb/je flattened semantics:
        # For vectorized version we produce full matrices but then zero rows outside mL..mH
        energy_out = em.copy()
        amp_out = amplitudes.copy()

        # zero frequency rows outside [mL, mH]
        if mL > 0:
            energy_out[:mL, :] = 0.0
            amp_out[:mL, :] = 0+0j
        if mH < M - 1:
            energy_out[mH+1:, :] = 0.0
            amp_out[mH+1:, :] = 0+0j

        # mimic jb/je flattened skipping by zeroing time bins whose flattened j are out of range
        # flattened j = m + t*M; compute min and max t allowed per m: we will set to zero where j<jb or j>=je
        # vectorized construction:
        m_idx = np.arange(M).reshape(M, 1)
        t_idx = np.arange(T).reshape(1, T)
        j_flat = m_idx + t_idx * M
        invalid_mask = (j_flat < jb) | (j_flat >= je)
        energy_out[invalid_mask] = 0.0
        amp_out[invalid_mask] = 0+0j

        self.last_energy = energy_out
        self.last_amplitude = amp_out

        if coeffs is None:
            self.data = amp_out if mode == 'a' else energy_out

        if return_map:
            return amp_out if mode == 'a' else energy_out

        return shape



def whiten_slice(data, rate, t, mode=1, offset=0.0, stride=0.0):
    """
    Robust time-domain whitening by local variance normalization (NumPy optimized).
    """
    data = np.ascontiguousarray(data, dtype=np.float64)
    N = len(data)
    segT = N / rate

    if t <= 0:
        t = segT - 2 * offset

    offset_samples = int(offset * rate + 0.5)
    if offset_samples % 2:
        offset_samples -= 1

    if stride > t or stride <= 0:
        stride = t

    K = int((segT - 2 * offset) / stride)
    if K == 0:
        K = 1

    n = N - 2 * offset_samples
    k = n // K
    if k % 2:
        k -= 1

    m = int(t * rate + 0.5)
    mL = int(0.15865 * m + 0.5)
    mR = m - mL - 1

    if m < 3 or mL < 2 or mR > m - 2:
        raise ValueError("whiten_timeseries: input array too short")

    # ---- build starting indices of blocks ----
    jL = (N - k * K) // 2
    jR = N - offset_samples - m
    jj = jL - m // 2
    starts = []
    for j in range(K + 1):
        if jj < offset_samples:
            starts.append(offset_samples)
        elif jj >= jR:
            starts.append(jR)
        else:
            starts.append(jj)
        jj += k
    starts = np.array(starts)

    # ---- extract all windows at once ----
    # shape: (K+1, m)
    windows = np.stack([data[s:s+m] for s in starts])

    # ---- compute robust stats per window ----
    q16, med, q84 = np.quantile(windows, [0.15865, 0.5, 0.84135], axis=1)
    medians = med if mode else np.sqrt(med * 0.7191)
    norms = (q84 - q16) / 2.0

    if mode == 0:
        return medians  # only noise estimates

    # ---- interpolation of median and norm across samples ----
    out = np.empty_like(data)

    # left boundary
    left_len = jL
    if left_len > 0:
        x = data[:left_len] - medians[0]
        r = norms[0]
        out[:left_len] = x / r if mode == 1 else x / (r * r)

    # main blocks (vectorized over each block of length k)
    p = left_len
    for j in range(K):
        idx = np.arange(k)  # [0, 1, ..., k-1]
        w = idx / k
        med_interp = medians[j] * (1 - w) + medians[j+1] * w
        norm_interp = norms[j] * (1 - w) + norms[j+1] * w
        x = data[p:p+k] - med_interp
        out[p:p+k] = x / norm_interp if mode == 1 else x / (norm_interp * norm_interp)
        p += k

    # right boundary
    if p < N:
        x = data[p:] - medians[-1]
        r = norms[-1]
        out[p:] = x / r if mode == 1 else x / (r * r)

    return out


def compute_rms(tf_map, t, mode, offset, stride):
    """
    Compute noise RMS for each layer in the time-frequency map.

    :param tf_map: Time-frequency map
    :type tf_map: TimeFrequencyMap
    :param t: whitening interval length in seconds (if <=0 use full duration minus offset*2)
    :param mode: whitening mode (0 = return medians, 1 = whitened, 2 = power-normalized)
    :param offset: boundary offset in seconds
    :param stride: step length in seconds
    :return: noise RMS for each layer
    :rtype: np.ndarray
    """
    duration = tf_map.stop - tf_map.start 
    if t <= 0:
        t = duration - 2. * offset
    w_mode = abs(mode)

    K = int((duration - 2 * offset) / stride) + 1 # number of noise measurements

    n_layers = len(tf_map.data)
    nRMS = np.zeros((n_layers, K))

    for i, layer in enumerate(tf_map.data):
        nRMS[i] = whiten_slice(np.abs(layer)**2, 1.0 / tf_map.dt, t, w_mode, offset, stride)

    return nRMS