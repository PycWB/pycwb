from dataclasses import dataclass, field
import numpy as np


@dataclass
class TimeSeries:
    data: np.ndarray
    t0: float
    dt: float

    def __post_init__(self):
        # Ensure the data is stored as a contiguous float64 NumPy array
        self.data = np.ascontiguousarray(self.data, dtype=np.float64)

    @property
    def times(self) -> np.ndarray:
        """Returns the array of time values."""
        return self.t0 + self.dt * np.arange(len(self.data))

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
    
    @classmethod
    def from_pycbc(cls, pycbc_ts):
        """
        Create a TimeSeries from a PyCBC TimeSeries object.
        
        :param pycbc_ts: PyCBC TimeSeries object
        :type pycbc_ts: pycbc.types.TimeSeries
        :return: TimeSeries object
        :rtype: TimeSeries
        """
        return cls(data=pycbc_ts.data, t0=pycbc_ts.start_time, dt=pycbc_ts.delta_t)
    
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