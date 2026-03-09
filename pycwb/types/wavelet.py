import numpy as np
from dataclasses import dataclass


@dataclass
class Wavelet:
    """
    Data class for storing wavelet information.
    """

    type: str

    def forward(self, ts):
        """
        Perform the forward wavelet transform on a time series.

        :param ts: Time series to transform
        :type ts: TimeSeries
        """
        pass  # Placeholder for actual implementation

    def inverse(self, tf_series):
        """
        Perform the inverse wavelet transform on a time-frequency series.

        :param tf_series: Time-frequency series to transform back
        :type tf_series: TimeFrequencySeries
        """
        pass  # Placeholder for actual implementation

