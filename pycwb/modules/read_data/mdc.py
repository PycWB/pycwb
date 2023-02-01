import numpy as np
from gwpy.timeseries import TimeSeries
import pycbc.noise
import pycbc.psd
from .data_check import data_check
from ligo.segments import segment, segmentlist


def read_from_gwf(detector, sample_rate, filename, channel, start, end):
    # Read data from GWF file
    data = TimeSeries.read(filename, channel, start, end)

    # Check data
    data_check(data, sample_rate)

    # data shift
    # SLAG
    # DC correction
    # resampling
    # rescaling

    # calculate noise rms
    # compute snr
    # zero f<fLow to avoid whitening issues when psd noise is not well defined for f<fLow
    # compute mdc snr

    # scale snr?

    # return data
    return data


def generate_noise(psd: str, f_low: int = 30, delta_f: float = 1.0 / 16, duration: int = 32, delta_t: float = 1.0 / 4096, seed: int = 1234):
    # generate noise
    flen = int(2048 / delta_f) + 1
    if psd:
        psd = pycbc.psd.from_txt(psd, flen, delta_f, f_low)
    else:
        psd = pycbc.psd.aLIGOZeroDetHighPower(flen, delta_f, f_low)

    # Generate 32 seconds of noise at 4096 Hz
    t_samples = int(duration / delta_t)
    noise = pycbc.noise.noise_from_psd(t_samples, delta_t, psd, seed=seed)
    # return noise
    return noise
