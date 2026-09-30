"""A small reproducible population of supported CBC waveforms."""

import numpy as np


def get_injection_parameters():
    return [dict(
        approximant="IMRPhenomXPHM", mass1=20.0, mass2=20.0,
        spin1z=float(spin), spin2z=0.0, distance=200.0,
        inclination=0.0, pol=0.0, coa_phase=0.0, f_lower=20.0,
        gps_time=1126259462.4, ra=0.0, dec=0.0,
        t_start=-10.0, t_end=1.0,
    ) for spin in np.linspace(-0.5, 0.5, 10)]
