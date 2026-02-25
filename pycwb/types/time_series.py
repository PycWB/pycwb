from dataclasses import dataclass
import numpy as np

from functools import partial
import jax
import jax.numpy as _jnp


@partial(jax.jit, static_argnames=("new_n",))
def _cwb_resample_jax(data, new_n):
    old_n = data.shape[0]
    old_n2 = old_n // 2

    old_fft = _jnp.fft.fft(data)
    packed = _jnp.zeros((old_n,), dtype=_jnp.float64)

    idx_old = _jnp.arange(old_n2, dtype=_jnp.int32)
    packed = packed.at[2 * idx_old].set(_jnp.real(old_fft[:old_n2]) / old_n)
    packed = packed.at[2 * idx_old + 1].set(_jnp.imag(old_fft[:old_n2]) / old_n)
    packed = packed.at[1].set(_jnp.real(old_fft[old_n2]) / old_n)
    if old_n & 1:
        packed = packed.at[old_n - 1].set(_jnp.imag(old_fft[old_n2]) / old_n)

    if new_n > old_n:
        packed_resized = _jnp.pad(packed, (0, new_n - old_n), mode="constant")
    else:
        packed_resized = packed[:new_n]

    new_n2 = new_n // 2
    spec = _jnp.zeros((new_n,), dtype=_jnp.complex128)

    idx_new = _jnp.arange(1, new_n2, dtype=_jnp.int32)
    re = packed_resized[2 * idx_new]
    im = packed_resized[2 * idx_new + 1]

    spec = spec.at[idx_new].set(re + 1j * im)
    spec = spec.at[new_n - idx_new].set(re - 1j * im)
    spec = spec.at[0].set(packed_resized[0])

    if new_n & 1:
        nyquist = packed_resized[1] + 1j * packed_resized[new_n - 1]
    else:
        nyquist = packed_resized[1] + 0j
    spec = spec.at[new_n2].set(nyquist)

    return _jnp.real(_jnp.fft.ifft(spec)) * new_n


@dataclass
class TimeSeries:
    data: np.ndarray
    t0: float
    dt: float

    def __post_init__(self):
        # Ensure the data is stored as a contiguous float64 NumPy array
        arr = np.ascontiguousarray(self.data, dtype=np.float64)
        if not arr.flags.writeable:
            arr = arr.copy()
        self.data = arr

    @property
    def times(self) -> np.ndarray:
        """Returns the array of time values."""
        return self.t0 + self.dt * np.arange(len(self.data))
    
    @property
    def sample_rate(self) -> float:
        """Returns the sampling rate."""
        return 1.0 / self.dt
    
    @property
    def end_time(self) -> float:
        """Returns the end time of the time series."""
        return self.t0 + self.dt * len(self.data)

    def __array__(self) -> np.ndarray:
        """Allows implicit conversion to a NumPy array."""
        return self.data

    def __getitem__(self, key):
        """Allow indexing into the time series."""
        return self.data[key]

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return (f"TimeSeries(data=..., t0={self.t0}, dt={self.dt}, "
                f"length={len(self.data)})")

    def __str__(self):
        return str(self.data)

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
    
    def to_pycbc(self):
        """
        Convert to a PyCBC TimeSeries object.
        
        :return: PyCBC TimeSeries object
        :rtype: pycbc.types.TimeSeries
        """
        from pycbc.types import TimeSeries as PyCBCTimeSeries
        return PyCBCTimeSeries(self.data, delta_t=self.dt, epoch=self.t0)
    
    def to_gwpy(self):
        """
        Convert to a GWPy TimeSeries object.
        
        :return: GWPy TimeSeries object
        :rtype: gwpy.timeseries.TimeSeries
        """
        from gwpy.timeseries import TimeSeries as GWPyTimeSeries
        return GWPyTimeSeries(self.data, t0=self.t0, dt=self.dt)

    def cwb_resampling(self, target_rate: float):
        """
        Resample using the same algorithmic path as cWB `wavearray::Resample(double)`.

        Parameters
        ----------
        target_rate : float
            Target sampling rate.

        Returns
        -------
        TimeSeries
            A new resampled time series.
        """
        old_n = int(self.data.size)
        if old_n == 0:
            return TimeSeries(data=self.data.copy(), t0=self.t0, dt=self.dt)

        current_rate = float(self.sample_rate)
        rsize = target_rate / current_rate * old_n
        new_n = int(round(rsize))

        if not np.isclose(rsize, new_n, atol=1e-9):
            raise ValueError(f"Resample frequency ({target_rate}) not allowed: non-integer target size {rsize}")
        if new_n % 2 != 0:
            raise ValueError(f"Resample frequency ({target_rate}) not allowed: target size must be even")

        if new_n == old_n:
            return TimeSeries(data=self.data.copy(), t0=self.t0, dt=self.dt)

        resampled = np.asarray(_cwb_resample_jax(_jnp.asarray(self.data), new_n), dtype=np.float64)

        return TimeSeries(data=resampled, t0=self.t0, dt=1.0 / float(target_rate))

    @classmethod
    def time_slide_copy(cls, ts_in, length: int = 0, src_idx: int = 0, dst_idx: int = 0):
        """
        Classmethod version of C++ wavearray::cpf(), but always operating on
        `ts_in` as both source and destination, returning a new modified copy.

        Parameters
        ----------
        ts_in : TimeSeries
            The input TimeSeries; used as both the source and destination.
        length : int, optional
            Number of samples to copy. 
            If 0, automatically computed from available array length.
        src_idx : int
            Starting index in `ts_in.data` from which samples are copied.
        dst_idx : int
            Starting index in the output TimeSeries where samples are written.

        Returns
        -------
        TimeSeries
            A new TimeSeries whose data contains the copied segment.
        """
        data = ts_in.data

        # Same auto-length logic as C++ cpf()
        if length == 0:
            length = min(len(data) - dst_idx, len(data) - src_idx)

        # Create a fresh copy of the full input time series
        new_ts = cls(
            data=np.copy(data),
            t0=ts_in.t0,
            dt=ts_in.dt
        )

        # Vectorized segment copy (NumPy handles any boundary issues gracefully)
        new_ts.data[dst_idx:dst_idx + length] = data[src_idx:src_idx + length]

        return new_ts
    
    @classmethod
    def from_pycbc(cls, pycbc_ts):
        """
        Create a TimeSeries from a PyCBC TimeSeries object.
        
        :param pycbc_ts: PyCBC TimeSeries object
        :type pycbc_ts: pycbc.types.TimeSeries
        :return: TimeSeries object
        :rtype: TimeSeries
        """
        return cls(data=pycbc_ts.data, t0=float(pycbc_ts.start_time), dt=pycbc_ts.delta_t)
    
    @classmethod
    def from_gwpy(cls, gwpy_ts):
        """
        Create a TimeSeries from a GWPy TimeSeries object.
        
        :param gwpy_ts: GWPy TimeSeries object
        :type gwpy_ts: gwpy.timeseries.TimeSeries
        :return: TimeSeries object
        :rtype: TimeSeries
        """
        return cls(data=gwpy_ts.value, t0=gwpy_ts.t0, dt=gwpy_ts.dt)

    @classmethod
    def from_input(cls, ts):
        """
        Create a TimeSeries from supported input types.

        Supported inputs:
        - pycwb.types.time_series.TimeSeries
        - pycbc.types.TimeSeries
        - gwpy.timeseries.TimeSeries

        :param ts: input time series object
        :type ts: object
        :return: TimeSeries object
        :rtype: TimeSeries
        """
        if isinstance(ts, cls):
            return ts

        if hasattr(ts, "delta_t") and hasattr(ts, "start_time"):
            return cls.from_pycbc(ts)

        if hasattr(ts, "value") and hasattr(ts, "t0") and hasattr(ts, "dt"):
            return cls.from_gwpy(ts)

        raise ValueError("input must be pycwb TimeSeries, pycbc TimeSeries, or gwpy TimeSeries")