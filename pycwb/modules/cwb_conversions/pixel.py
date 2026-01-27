import ROOT
import numpy as np

from pycwb.types.network_pixel import Pixel, PixelData
from .series import convert_to_wavearray, convert_wavearray_to_timeseries
def convert_pixel_to_netpixel(pixel, c_id):
    """
    Convert pixel to netpixel

    :param pixel: pixel
    :type pixel: Pixel
    :return: netpixel
    :rtype: ROOT.netpixel
    """
    netpixel = ROOT.netpixel()
    netpixel.clusterID = c_id
    netpixel.time = pixel.time
    netpixel.frequency = pixel.frequency
    netpixel.layers = pixel.layers
    netpixel.rate = pixel.rate
    netpixel.likelihood = pixel.likelihood
    netpixel.null = pixel.null
    netpixel.theta = pixel.theta
    netpixel.phi = pixel.phi
    netpixel.ellipticity = pixel.ellipticity
    netpixel.polarisation = pixel.polarisation
    netpixel.core = pixel.core
    netpixel.data = [convert_to_pixdata(d) for d in pixel.data]
    netpixel.tdAmp = convert_td_amp_to_cwb(pixel.td_amp)
    netpixel.neighbors = pixel.neighbors
    return netpixel


def convert_to_pixdata(pixeldata):
    pixdata = ROOT.pixdata()
    pixdata.noiserms = pixeldata.noise_rms
    pixdata.wave = pixeldata.wave
    pixdata.w_90 = pixeldata.w_90
    pixdata.asnr = pixeldata.asnr
    pixdata.a_90 = pixeldata.a_90
    pixdata.rank = pixeldata.rank
    pixdata.index = pixeldata.index
    return pixdata


def convert_netpixel_to_pixel(netpixel):
    pixel = Pixel(time=netpixel.time, frequency=netpixel.frequency, layers=netpixel.layers, rate=netpixel.rate,
                    likelihood=netpixel.likelihood, null=netpixel.null, theta=netpixel.theta, phi=netpixel.phi,
                    ellipticity=netpixel.ellipticity, polarisation=netpixel.polarisation, core=netpixel.core,
                    data=[PixelData(d.noiserms, d.wave, d.w_90, d.asnr, d.a_90, d.rank, d.index) for d in netpixel.data],
                    td_amp=np.array(netpixel.tdAmp), neighbors=list(netpixel.neighbors))

    return pixel


def convert_td_amp_to_cwb(td_amps):
    import ctypes
    c_double_p = ctypes.POINTER(ctypes.c_double)

    res = []
    if td_amps is not None and len(td_amps) > 0:
        for i, data in enumerate(td_amps):
            h = ROOT.wavearray(np.double)(len(data))
            ROOT.pycwb_copy_to_wavearray(data.ctypes.data_as(c_double_p), h, len(data))
            res.append(h)

    return res



