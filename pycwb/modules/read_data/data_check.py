import numpy as np
from gwpy.timeseries import TimeSeries


def data_check(data: TimeSeries, sample_rate: int):
    # check if data contains NaNs
    if data.value.any() == np.nan:
        raise ValueError('Data contains NaNs')
    # check if the sample rate is consitent with configuation
    if data.sample_rate.value != sample_rate:
        raise ValueError('Sample rate is not consistent with configuation')

    return True
