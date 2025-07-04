import random
import numpy as np
from pycwb.modules.injection.par_generator import repeat

def get_injection_parameters():

    """
    Generate parameters for white noise burst injection.
    
    Returns:
        dict: A dictionary containing the parameters for the injection.
    """
    SGEQ3 = [{
        'frequency': f, 
        'Q': 3.0, 
        'name': f'SGE_Q3_{f}Hz'
    } for f in [36, 70, 235, 849, 1615]]

    SGEQ9a = [{
        'frequency': f, 
        'Q': 9.0, 
        'name': f'SGE_Q9_{f}Hz'
    } for f in [70, 100, 235, 361]]

    SGEQ9b = [{
        'frequency': f, 
        'Q': 9.0, 
        'name': f'SGE_Q9_{f}Hz'
    } for f in [36, 48, 153, 554, 849, 1304, 1615]]

    SGEQ100 = [{
        'frequency': f, 
        'Q': 100.0, 
        'name': f'SGE_Q100_{f}Hz'
    } for f in [48, 70, 235, 849, 1304, 1615]]

    SGE = SGEQ3 + SGEQ9a + SGEQ9b + SGEQ100

    # Add approximant type to each waveform
    for wf in SGE:
        wf['approximant'] = 'SGE'
    
    # Repeat the waveforms to create a larger set
    # CAUTION: DO NOT simply use `SGE * 10` as it will create a shallow copy!!!
    SGE = repeat(SGE, 10)

    iota_list = np.random.uniform(0, np.pi/2, len(SGE))

    for wf, iota in zip(SGE, iota_list):
        wf['ellipticity'] = iota

    # add estimation for the start and end time of the waveform
    for wf in SGE:
        wf['t_start'] = -1.0
        wf['t_end'] = 1.0
        wf['pol'] = 0.

    # add hrss to each waveform
    hrss_min = 5e-23
    hrss_list = [hrss_min * (2 ** i) for i in range(7)]
    for wf in SGE:
        wf['hrss'] = random.choice(hrss_list)

    return SGE
