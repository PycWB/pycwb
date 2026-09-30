"""Inspect a deterministic time veto without running a search."""
import numpy as np
from pycwb.types.time_series import TimeSeries
from pycwb.modules.conditioning_plugins.cwb_gating import gate_intervals
from pycwb.modules.conditioning_plugins.api import subtract_intervals

samples = np.zeros(64 * 128)
samples[30 * 128:30 * 128 + 16] = 2000
strain = TimeSeries(data=samples.copy(), t0=1000.0, dt=1 / 128)
excluded = gate_intervals(strain, edge=10)
accepted = subtract_intervals(None, excluded, 1010, 1054)
print("Excluded GPS intervals:", excluded)
print("Accepted GPS intervals:", accepted)
print("Accepted seconds:", sum(stop - start for start, stop in accepted))
print("Strain unchanged:", np.array_equal(strain.data, samples))
