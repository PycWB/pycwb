import numpy as np
from gwpy.timeseries import TimeSeries
import logging
from pycwb.types.time_series import TimeSeries as PycwbTimeSeries
from pycwb.utils.conversions.timeseries import convert_to_pycbc_timeseries

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
    Legacy-compatible data check and resampling.

    This function keeps compatibility with the ROOT/legacy pipeline by returning
    a PyCBC TimeSeries (or compatible object accepted by legacy converters).

    :param data:
    :type data: pycwb.types.time_series.TimeSeries or pycbc.types.timeseries.TimeSeries or gwpy.timeseries.TimeSeries
    :param config:
    :param ifo_index:
    :return: pycbc.types.timeseries.TimeSeries
    """
    data = convert_to_pycbc_timeseries(data)

    # check if data contains NaNs
    if np.isnan(np.asarray(data.data, dtype=np.float64)).any():
        raise ValueError('Data contains NaNs')
    # check if the sample rate is consitent with configuation
    if float(data.sample_rate) != float(config.inRate):
        raise ValueError('Sample rate is not consistent with configuation')

    # DC correction
    if config.dcCal[ifo_index] > 0 and config.dcCal[ifo_index] != 1.0:
        data.data *= config.dcCal[ifo_index]
        logger.info(f"DC correction: {config.dcCal[ifo_index]}")

    # resampling
    if config.fResample > 0:
        logger.info(f"Resampling data from {float(data.sample_rate)} to {config.fResample}")
        data = data.resample(1.0 / float(config.fResample))

    new_sample_rate = float(data.sample_rate) / (1 << config.levelR)
    if new_sample_rate != config.inRate:
        logger.info(f"Resampling data from {float(data.sample_rate)} to {new_sample_rate}")
        data = data.resample(1.0 / float(new_sample_rate))

    # rescaling
    data.data *= (2 ** config.levelR) ** 0.5

    return data


def check_and_resample_py(data, config, ifo_index):
    """
    Python-native data check and resampling.

    Input is normalized to pycwb TimeSeries and output remains pycwb TimeSeries.

    :param data:
    :type data: pycwb.types.time_series.TimeSeries or pycbc.types.timeseries.TimeSeries or gwpy.timeseries.TimeSeries
    :param config:
    :param ifo_index:
    :return: pycwb TimeSeries
    """
    data = PycwbTimeSeries.from_input(data)

    # check if data contains NaNs
    if np.isnan(np.asarray(data.data)).any():
        raise ValueError('Data contains NaNs')
    # check if the sample rate is consitent with configuation
    sr = data.sample_rate
    # gwpy-derived TimeSeries may carry an astropy Quantity for sample_rate
    sr_val = float(sr.value) if hasattr(sr, "value") else float(sr)
    if sr_val != config.inRate:
        raise ValueError('Sample rate is not consistent with configuation')

    # DC correction
    if config.dcCal[ifo_index] > 0 and config.dcCal[ifo_index] != 1.0:
        data.data *= config.dcCal[ifo_index]
        logger.info(f"DC correction: {config.dcCal[ifo_index]}")

    # resampling
    if config.fResample > 0:
        logger.info(f"Resampling data from {data.sample_rate} to {config.fResample}")
        data = data.cwb_resampling(float(config.fResample))

    new_sample_rate = data.sample_rate / (1 << config.levelR)
    new_sr_val = float(new_sample_rate.value) if hasattr(new_sample_rate, "value") else float(new_sample_rate)
    if new_sr_val != config.rateANA:
        logger.info(f"Resampling data from {data.sample_rate} to {new_sample_rate}")
        data = data.cwb_resampling(new_sr_val)

    # rescaling
    data.data *= (2 ** config.levelR) ** 0.5

    return data
