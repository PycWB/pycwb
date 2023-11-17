import numpy as np


def get_injection_parameters():
    return [{
        'mass1': 20,
        'mass2': 20,
        'spin1x': 0,
        'spin1y': 0,
        'spin1z': spin1z,
        'spin2x': 0,
        'spin2y': 0,
        'spin2z': 0,
        'distance': 200,
        'inclination': 0,
        'polarization': 0,
        'gps_time': 1126259462.4,
        'coa_phase': 0,
        'ra': 0,
        'dec': 0
    } for spin1z in np.arange(-0.5, 0.5, 1 / 10)]

