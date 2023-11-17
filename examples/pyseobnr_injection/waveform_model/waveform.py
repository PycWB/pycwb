from pyseobnr.generate_waveform import GenerateWaveform


def waveform_generator(mass1, mass2, spin1x, spin1y, spin1z, spin2x, spin2y, spin2z, distance, inclination, polarization, coa_phase,
                       f_lower, delta_t, **kwargs):
    parameters = {
        'mass1': mass1,
        'mass2': mass2,
        'spin1x': spin1x,
        'spin1y': spin1y,
        'spin1z': spin1z,
        'spin2x': spin2x,
        'spin2y': spin2y,
        'spin2z': spin2z,
        'distance': distance,
        'inclination': inclination,
        'polarization': polarization,
        'coa_phase': coa_phase,
        'f_ref': f_lower,
        'f22_start': f_lower,
        'deltaT': delta_t,
        "approximant": "SEOBNRv5HM",
    }
    wfm_gen = GenerateWaveform(parameters)
    hp, hc = wfm_gen.generate_td_polarizations_conditioned_2()

    return hp, hc

