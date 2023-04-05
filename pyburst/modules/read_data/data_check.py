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


def check_and_resample(data, config, ifo_index):
    """
    Check data and resample it
    :param data:
    :type data: gwpy.timeseries.TimeSeries or pycbc.types.timeseries.TimeSeries
    :param config:
    :param ifo_index:
    :return:
    """
    if isinstance(data, TimeSeries):
        data = data.to_pycbc()

    # check if data contains NaNs
    if data.data.any() == np.nan:
        raise ValueError('Data contains NaNs')
    # check if the sample rate is consitent with configuation
    if data.sample_rate != config.inRate:
        raise ValueError('Sample rate is not consistent with configuation')

    # TODO: complete the following
    # data shift
    # SLAG
    # DC correction
    if config.dcCal[ifo_index] > 0 and config.dcCal[ifo_index] != 1.0:
        data.data *= config.dcCal[config.ifo.indexof(ifo_index)]

    # resampling
    if config.fResample > 0:
        data = data.resample(1.0 / config.fResample)

    new_sample_rate = data.sample_rate / (1 << config.levelR)
    data = data.resample(1.0 / new_sample_rate)

    # rescaling
    data.data *= (2 ** config.levelR) ** 0.5

    return data