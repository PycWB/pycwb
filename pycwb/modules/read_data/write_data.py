"""Writers for gravitational-wave strain data."""

import os

from gwpy.timeseries import TimeSeries as GWpyTimeSeries

__all__ = ["save_to_gwf"]


def save_to_gwf(
    signals,
    detectors,
    channel_name,
    out_dir,
    start_time,
    duration,
    label,
):
    """Save one strain signal per detector to GWF files."""
    os.makedirs(out_dir, exist_ok=True)

    for signal, detector in zip(signals, detectors):
        strain = GWpyTimeSeries(data=signal.data, times=signal.sample_times)
        strain.channel = f"{detector}:{channel_name}"
        strain.name = strain.channel
        filename = (
            f"{detector}-{label}-{int(start_time)}-{int(duration)}.gwf"
        )
        strain.write(os.path.join(out_dir, filename))
