import numpy as np
from gwpy.timeseries import TimeSeries


def data_check(data: TimeSeries, sample_rate: int):
    """
    Check if data contains NaNs and if the sample rate is consistent with configuation

    :param data: time series data to be checked
    :type data: gwpy.timeseries.TimeSeries
    :param sample_rate: sample rate from configuration
    :type sample_rate: int
    :return: True if data is valid
    :rtype: bool
    :raises ValueError: if data contains NaNs or sample rate is not consistent with configuation
    """
    # check if data contains NaNs
    if data.value.any() == np.nan:
        raise ValueError('Data contains NaNs')
    # check if the sample rate is consitent with configuation
    if data.sample_rate.value != sample_rate:
        raise ValueError('Sample rate is not consistent with configuation')

    return True
