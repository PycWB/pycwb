from burst_waveform.models.white_noise_burst import WhiteNoiseBurstEllipticity
 
def get_td_waveform(frequency, bandwidth, duration, inj_length=1., xseed=0, pseed=0, mode=0, **kwargs):
    """
    Generate parameters for white noise burst injection.
    """
    params = {
        'frequency': frequency,
        'bandwidth': bandwidth,
        'duration': duration,
        'inj_length': inj_length,
        'pseed': pseed,
        'xseed': xseed,
        'mode': mode,
    }

    WNB = WhiteNoiseBurstEllipticity(params)
    wnb_burst_waveform_hplus, wnb_burst_waveform_hcross = WNB()

    return wnb_burst_waveform_hplus, wnb_burst_waveform_hcross