import numpy as np
from pycwb.modules.injection.par_generator import inc_pol_replicator

def get_injection_parameters():
    # b range from 50 to 120, with 10 steps
    mass_list = [{
        'mass1': 30,
        'mass2': 20,
        'spin1z': 0,
        'spin2z': 0,
        'distance': 500,
        'inclination': 0,
        'polarization': 0,
        'coa_phase': 0,
        't_start': -2, # this is an conservative estimation
        't_end': 0.5, # this is an conservative estimation
    }]

    inc_list = np.random.uniform(0, np.pi, 6)
    pol_list = np.random.uniform(0, 2*np.pi, 6)

    final_list = inc_pol_replicator(mass_list, inc_list, pol_list)

    return final_list