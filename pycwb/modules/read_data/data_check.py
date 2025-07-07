import numpy as np
from gwpy.timeseries import TimeSeries
import logging

from pycwb.modules.cwb_conversions import convert_to_wavearray, convert_wavearray_to_timeseries, \
    convert_wavearray_to_pycbc_timeseries

logger = logging.getLogger(__name__)


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

    # DC correction
    if config.dcCal[ifo_index] > 0 and config.dcCal[ifo_index] != 1.0:
        data.data *= config.dcCal[config.ifo.indexof(ifo_index)]
        logger.info(f"DC correction: {config.dcCal[ifo_index]}")

    # resampling
    if config.fResample > 0:
        logger.info(f"Resampling data from {data.sample_rate} to {config.fResample}")
        # data = data.resample(1.0 / config.fResample)
        w = convert_to_wavearray(data)
        w.Resample(config.fResample)
        data = convert_wavearray_to_pycbc_timeseries(w)

    new_sample_rate = data.sample_rate / (1 << config.levelR)
    if new_sample_rate != config.inRate:
        logger.info(f"Resampling data from {data.sample_rate} to {new_sample_rate}")
        w = convert_to_wavearray(data)
        w.Resample(new_sample_rate)
        data = convert_wavearray_to_pycbc_timeseries(w)
    # data = data.resample(1.0 / new_sample_rate)

    # rescaling
    data.data *= (2 ** config.levelR) ** 0.5

    return data
