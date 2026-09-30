"""Optional population for a user-supplied hyperbolic LALSuite approximant."""

import numpy as np


def get_injection_parameters(approximant):
    """Require an explicit model from the custom LAL build described in README."""
    if not approximant or approximant == "MODEL_NAME_FROM_YOUR_LAL_BUILD":
        raise ValueError("Supply the hyperbolic approximant installed in your custom LALSuite build")
    return [dict(
        approximant=approximant, mass1=20, mass2=20, spin1z=0, spin2z=0,
        hyp_eccentricity=1.15, b=float(impact), distance=200,
        inclination=0, pol=0, coa_phase=0, f_lower=20.0,
        gps_time=1126259462.4, ra=0, dec=0,
    ) for impact in np.linspace(50, 113, 10)]
