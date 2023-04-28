import ROOT


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
    netpixel.tdAmp = pixel.td_amp
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
