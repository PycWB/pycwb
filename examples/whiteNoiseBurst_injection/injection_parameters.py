import random
import numpy as np


def get_injection_parameters():

    """
    Generate parameters for white noise burst injection.
    
    Returns:
        dict: A dictionary containing the parameters for the injection.
    """
    seed = random.randint(100000, 999999)  # Random seed for reproducibility
    
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
    
    