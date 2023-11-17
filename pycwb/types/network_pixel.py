from dataclasses import dataclass


@dataclass(slots=True)
class PixelData:
    """
    Pixel data for single detector

    :param noise_rms: average noise rms
    :type noise_rms: float
    :param wave: vector of 00 pixel's wavelet amplitudes
    :type wave: float
    :param w_90: vector of 90 pixel's wavelet amplitudes
    :type w_90: float
    :param asnr: vector of 00 pixel's whitened amplitudes
    :type asnr: float
    :param a_90: vector of 90 pixel's whitened amplitudes
    :type a_90: float
    :param rank: vector of pixel's rank amplitudes
    :type rank: float
    :param index: index in wavearray
    :type index: int
    """
    noise_rms: float
    wave: float
    w_90: float
    asnr: float
    a_90: float
    rank: float
    index: int
    # __slots__ = ['noise_rms', 'wave', 'w_90', 'asnr', 'a_90', 'rank', 'index']
    #
    # def __init__(self, noise_rms, wave, w_90, asnr, a_90, rank, index):
    #     self.noise_rms = noise_rms  # average noise rms
    #     self.wave = wave  # vector of 00 pixel's wavelet amplitudes
    #     self.w_90 = w_90  # vector of 90 pixel's wavelet amplitudes
    #     self.asnr = asnr  # vector of 00 pixel's whitened amplitudes
    #     self.a_90 = a_90  # vector of 90 pixel's whitened amplitudes
    #     self.rank = rank  # vector of pixel's rank amplitudes
    #     self.index = index  # index in wavearray
    #
    # def __repr__(self):
    #     return self.to_dict().__repr__()
    #
    # def to_dict(self):
    #     return {key: getattr(self, key) for key in self.__slots__}


@dataclass(slots=True)
class Pixel:
    """
    Pixel data for network
    """
    time: int  # time index for master detector
    frequency: int  # frequency index (layer)
    layers: int  # number of frequency layers
    rate: float  # wavelet layer rate
    likelihood: float  # likelihood
    null: float  # null
    theta: float  # source angle theta index
    phi: float  # source angle phi index
    ellipticity: float  # waveform ellipticity
    polarisation: float  # waveform polarisation
    core: int  # pixel type: true - core , false - halo

    data: list[PixelData]  # list of PixelData
    td_amp: list[float]  # time domain amplitude
    neighbors: list[int]  # list of neighbors

    # __slots__ = ['time', 'frequency', 'layers', 'rate', 'likelihood', 'null', 'theta', 'phi',
    #              'ellipticity', 'polarisation', 'core', 'data', 'td_amp', 'neighbors']
    #
    # def __init__(self, time=None, frequency=None, layers=None, rate=None, likelihood=None, null=None,
    #              theta=None, phi=None, ellipticity=None, polarisation=None, core=None, data=None,
    #              td_amp=None, neighbors=None):
    #     self.time = time  # time index for master detector
    #     self.frequency = frequency  # frequency index (layer)
    #     self.layers = layers  # number of frequency layers
    #     self.rate = rate  # wavelet layer rate
    #     self.likelihood = likelihood  # likelihood
    #     self.null = null  # null
    #     self.theta = theta  # source angle theta index
    #     self.phi = phi  # source angle phi index
    #     self.ellipticity = ellipticity  # waveform ellipticity
    #     self.polarisation = polarisation  # waveform polarisation
    #     self.core = core  # pixel type: true - core , false - halo
    #
    #     self.data = data  # pixel data
    #     self.td_amp = td_amp  # time domain amplitude
    #     self.neighbors = neighbors  # list of neighbors
    #
    # def from_netpixel(self, netpixel):
    #     self.time = netpixel.time
    #     self.frequency = netpixel.frequency
    #     self.layers = netpixel.layers
    #     self.rate = netpixel.rate
    #     self.likelihood = netpixel.likelihood
    #     self.null = netpixel.null
    #     self.theta = netpixel.theta
    #     self.phi = netpixel.phi
    #     self.ellipticity = netpixel.ellipticity
    #     self.polarisation = netpixel.polarisation
    #     self.core = netpixel.core
    #     self.data = [PixelData(d.noiserms, d.wave, d.w_90, d.asnr, d.a_90, d.rank, d.index) for d in netpixel.data]
    #     self.td_amp = [t for t in netpixel.tdAmp]
    #     self.neighbors = list(netpixel.neighbors)
    #     return self
    #
    # def __repr__(self):
    #     return self.to_dict().__repr__()
    #
    # def to_dict(self):
    #     return {key: getattr(self, key) for key in self.__slots__}

    @property
    def time_in_seconds(self):
        dt = 1. / self.rate
        time = int(self.time / self.layers) * dt - dt / 2
        return time

    @property
    def frequency_in_hz(self):
        df = self.rate / 2
        return self.frequency * df
