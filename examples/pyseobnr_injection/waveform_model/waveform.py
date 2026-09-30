from pyseobnr.generate_waveform import GenerateWaveform
from pycwb.types.time_series import TimeSeries


def waveform_generator(mass1, mass2, spin1x, spin1y, spin1z, spin2x, spin2y, spin2z, distance, inclination, coa_phase,
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
        # Sky polarization is applied by PycWB's detector projection.
        'phi_ref': coa_phase,
        'f_ref': f_lower,
        'f22_start': f_lower,
        'deltaT': delta_t,
        "approximant": "SEOBNRv5HM",
    }
    wfm_gen = GenerateWaveform(parameters)
    hp, hc = wfm_gen.generate_td_polarizations_conditioned_2()

    # pySEOBNR returns LAL REAL8TimeSeries; the injection API takes native,
    # GWpy or PyCBC time series. Preserve the epoch and sampling explicitly.
    return {
        "type": "polarizations",
        "hp": TimeSeries(data=hp.data.data.copy(), t0=float(hp.epoch), dt=hp.deltaT),
        "hc": TimeSeries(data=hc.data.data.copy(), t0=float(hc.epoch), dt=hc.deltaT),
    }
