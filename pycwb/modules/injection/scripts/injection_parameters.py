import numpy as np
from pycwb.modules.injection.par_generator import inc_pol_replicator

def get_injection_parameters():
    # b range from 50 to 120, with 10 steps
    mass_list = [{
        'mass1': m1,
        'mass2': 20,
        'spin1z': 0,
        'spin2z': 0,
        'distance': d,
        'inclination': 0,
        'polarization': 0,
        'coa_phase': 0,
    } for m1 in np.arange(50, 120, (120 - 50) / 4) for d in np.arange(200, 400, (400 - 200) / 8)]

    inc_list = np.random.uniform(0, np.pi, 200)
    pol_list = np.random.uniform(0, 2*np.pi, 200)

    final_list = inc_pol_replicator(mass_list, inc_list, pol_list)

    return final_list
