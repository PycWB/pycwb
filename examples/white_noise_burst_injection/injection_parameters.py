import random
import numpy as np
from pycwb.modules.injection.par_generator import repeat


def get_injection_parameters_snr(seed=None):

    """
    Generate parameters for white noise burst injection.
    
    Returns:
        dict: A dictionary containing the parameters for the injection.
    """
    snr_min = 6.5
    snr_max = 300
    snr_dist = 2
    # snr^-snr_dist
    target_snr_list = []
    n = 0
    while True:
        snr = snr_min * snr_dist ** n
        if snr > snr_max:
            break
        target_snr_list.append(snr)
        n += 1

    RandomWNB17b = []
    for n in range(16):
        frequency = random.uniform(24, 1696)  # WNB17 -> (24,996)
        bandwidth = random.uniform(10, 300)
        duration = 10 ** random.uniform(np.log10(0.0001), np.log10(0.5))
        for m in range(30):
            RandomWNB17b.append({
                'name': f'WNB17b_{n}_{m}',
                'frequency': frequency,
                'bandwidth': bandwidth,
                'duration': duration,
                'inj_length': 1.0,  # Default value
                'pseed': seed + 100000 + n * 100 + m,
                'xseed': seed + 100001 + n * 100 + m,
                "t_start": -1.0, 
                "t_end": 1.0,
                'mode': 1  # symmetric
            })

    # randomly assign target SNR
    for i in range(len(RandomWNB17b)):
        RandomWNB17b[i]['target_snr'] = random.choice(target_snr_list)
    
    return RandomWNB17b
    
    
def get_injection_parameters(seed=None):
    RandomWNB17b = []
    for n in range(16):
        frequency = random.uniform(24, 1696)  # WNB17 -> (24,996)
        bandwidth = random.uniform(10, 300)
        duration = 10 ** random.uniform(np.log10(0.0001), np.log10(0.5))
        for m in range(30):
            RandomWNB17b.append({
                'name': f'WNB17b_{n}_{m}',
                'frequency': frequency,
                'bandwidth': bandwidth,
                'duration': duration,
                'inj_length': 1.0,  # Default value
                'pseed': seed + 100000 + n * 100 + m,
                'xseed': seed + 100001 + n * 100 + m,
                "t_start": -1.0, 
                "t_end": 1.0,
                'mode': 1,  # symmetric
                'pol': 0.0,  # Default polarization
            })

    
    # Repeat the waveforms to create a larger set
    # CAUTION: DO NOT simply use `RandomWNB17b * 10` as it will create a shallow copy!!!
    RandomWNB17b = repeat(RandomWNB17b, 10)

    iota_list = np.random.uniform(0, np.pi/2, len(RandomWNB17b))
    # add hrss to each waveform
    hrss_min = 5e-23
    hrss_list = [hrss_min * (2 ** i) for i in range(3)]

    for wf, iota in zip(RandomWNB17b, iota_list):
        wf['ellipticity'] = iota
        wf['hrss'] = random.choice(hrss_list)
        wf['approximant'] = 'WNB'

    print(f"Total number of waveforms: {len(RandomWNB17b)}")
    return RandomWNB17b


