import numpy as np
from dataclasses import dataclass
from .wavelet import Wavelet
from .time_series import TimeSeries

class TimeFrequencySeries:
    """
    Class for storing a time-frequency series. This class is a Python wrapper for ROOT.WSeries

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
        return len(self.data)
    
    @property
    def maxLayer(self):
        return self.wavelet.M

    def wavecount(self, threshold, edge_length=None):
        """
        Count the number of wavelet coefficients above a certain threshold.
        Backward compatibility with ROOT.WaveArray::wavecount

        :param threshold: threshold value
        :type threshold: float
        :param edge_length: edge length to exclude
        :type edge_length: int
        :return: number of coefficients above the threshold
        :rtype: int
        """
        # np_array[np_array > 0.01].size
        if edge_length is not None:
            return np.sum(self.data[edge_length:-edge_length] > threshold)
        return np.sum(self.data > threshold)

    def wavesplit(self, start_index, end_index, split_index):
        """
        Find the value at the split index in the sorted array segment.
        Backward compatibility with ROOT.WaveArray::wavesplit

        :param start_index: start index of the segment
        :type start_index: int
        :param end_index: end index of the segment
        :type end_index: int
        :param split_index: index to split the sorted segment
        :type split_index: int
        :return: value at the split index
        :rtype: float
        """
        split_index = split_index - 1 # don't know why, it is consistent with ROOT
        parted = np.partition(self.data[start_index:end_index], split_index)
        value = parted[split_index]
        return value
    
    def Gamma2Gauss(self, hist=None):
        M = self.maxLayer() + 1
        nL = int(self.edge * self.wavelet_rate * M)
        nn = len(self.data)
        nR = nn - nL - 1

        # fraction of near-zero values
        fff = (nR - nL) * self.wavecount(0.001) / float(nn)

        # median estimate
        med = self.wavesplit(nL, nR, nR - int(0.5 * fff))

        # Vectorized computation for Gamma parameter estimation
        data_slice = self.data[nL:nR]
        mask = (data_slice > 0.01) & (data_slice < 20 * med)
        valid_data = data_slice[mask]
        
        if len(valid_data) == 0:
            return 0.0
            
        aaa = np.sum(valid_data)
        bbb = np.sum(np.log(valid_data))
        count = len(valid_data)

        # Estimate Gamma shape parameter (alpha)
        alp = np.log(aaa / count) - bbb / count
        alp = (3 - alp + np.sqrt((alp - 3) * (alp - 3) + 24 * alp)) / (12 * alp)

        # Gamma mean estimate
        avr = med * (3 * alp + 0.2) / (3 * alp - 0.8)

        # Scaling parameter
        ALP = med * alp / avr

        # Vectorized nonlinear Gamma → Gaussian transform
        amp = self.data * alp / avr
        self.data = np.where(amp < ALP, 0.0, amp - ALP * (1 + np.log(amp / ALP)))
        
        if hist is not None:
            hist.extend(self.data[nL:nR].tolist())

        # Renormalization
        fff = self.wavecount(1.0e-5, nL)
        rms = 1.0 / self.wavesplit(nL, nR, nR - int(0.3173 * fff))

        self.data *= rms
        
        if hist is not None:
            hist.extend(np.sqrt(self.data[nL:nR]).tolist())

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

    def time_delay_max_energy(self, dt):
        """
        Calculate the time delay with maximum energy.

        :param dt: time delay
        :type dt: float
        :return: time delay with maximum energy
        :rtype: float
        """
        
        pass

    def _compute_bounds(self):
        M = self.maxLayer()
        J = self.size()
        jb = int(self.edge * self.wavelet_rate / 4.0) * M
        if jb < 4 * M:
            jb = 4 * M
        je = J - jb
        df = self.df
        mL = int(self.f_low / df + 0.1)
        mH = int(self.f_high / df + 0.1)
        return jb, je, mL, mH
    
    def wdm_packet(self, pattern: int, mode: str = 'e') -> float:
        """
        Vectorized implementation using numpy slicing (no Python-for loops over pixels).
        Edge handling is non-wrapping (shifts produce zeros outside bounds).
        This version aims for maximal speed on large arrays.
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
        M, T = self.M, self.T
        jb, je, mL, mH = self._compute_bounds()

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
        ee += (self.c.real ** 2) * center_weight
        EE += (self.c.imag ** 2) * center_weight
        ss += (self.c.real * self.c.imag) * center_weight

        # helper: add shifted contributions of offsets without wrap (zero outside)
        def add_shift(dr: int, dt: int):
            """Add contribution of self.c shifted by (dr, dt) into ee/EE/ss without wrap."""
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

            block = self.c[src_m0:src_m1, src_t0:src_t1]
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