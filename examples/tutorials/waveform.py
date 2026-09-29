"""An illustrative linearly polarized sine-Gaussian, normalized in source hrss."""
import numpy as np
from pycwb.types.time_series import TimeSeries


def get_td_waveform(delta_t, frequency=150.0, Q=9.0, hrss=1e-21, **parameters):
    # Use the same explicit support as the example configuration.
    start = float(parameters.get("t_start", -0.5))
    stop = float(parameters.get("t_end", 0.5))
    time = np.arange(start, stop, delta_t)
    tau = Q / (np.sqrt(2) * np.pi * frequency)
    plus = np.exp(-(time / tau) ** 2) * np.cos(2 * np.pi * frequency * time)
    plus *= hrss / np.sqrt(np.sum(plus ** 2) * delta_t)
    return {"type": "polarizations",
            "hp": TimeSeries(data=plus, t0=start, dt=delta_t),
            "hc": TimeSeries(data=np.zeros_like(plus), t0=start, dt=delta_t)}
