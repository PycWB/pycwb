from lal import REAL8TimeSeries
from pycbc.types import TimeSeries
from gwpy.timeseries import TimeSeries as GWpyTimeSeries
from pycwb.types.time_series import TimeSeries as PycWBTimeSeries

def convert_to_pycbc_timeseries(lal_ts):
    """
    Convert a LAL TimeSeries to a pycbc TimeSeries.
    """
    if type(lal_ts) == REAL8TimeSeries:
        # Convert to pycbc TimeSeries
        hp = TimeSeries(lal_ts.data.data, delta_t=lal_ts.deltaT, epoch=lal_ts.epoch)
    elif type(lal_ts) == TimeSeries:
        hp = lal_ts
    elif isinstance(lal_ts, GWpyTimeSeries):
        # Convert GWpy TimeSeries to pycbc TimeSeries
        hp = lal_ts.to_pycbc()
    elif isinstance(lal_ts, PycWBTimeSeries):
        # Convert PycWB TimeSeries to pycbc TimeSeries
        hp = lal_ts.to_pycbc()
    else:
        raise TypeError("Input must be a LAL or pycbc TimeSeries")

    return hp
