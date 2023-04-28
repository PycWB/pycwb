import numpy as np


def get_injection_parameters():
    # b range from 50 to 120, with 10 steps
    return [{
        'mass1': 20,
        'mass2': 20,
        'spin1z': 0,
        'spin2z': 0,
        'hyp_eccentricity': 1.15,
        'b': b,
        'distance': 200,
        'inclination': 0,
        'polarization': 0,
        'gps_time': 1126259462.4,
        'coa_phase': 0,
        'ra': 0,
        'dec': 0
    } for b in np.arange(50, 120, (120 - 50) / 10)]
