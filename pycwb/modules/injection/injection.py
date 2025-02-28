import numpy as np
from math import ceil
import logging

logger = logging.getLogger(__name__)


def generate_injection_list_from_config(injection_config):
    pass

def distribute_inj_in_gps_time(injections, rate, jitter, 
                               start_gps_time, end_gps_time, 
                               edge_buffer = 0,
                               allow_repeat=True):
    interval = 1 / rate
    if jitter > interval / 2:
        raise ValueError('Jitter is too large')
    
    total_available_time = (end_gps_time - edge_buffer) - (start_gps_time + edge_buffer)

    n_inj = len(injections)
    required_time = n_inj / rate

    if interval > total_available_time:
        raise ValueError(f'Rate is too large, required available time: {interval} s for rate {rate}, but only {total_available_time} s available')
    if required_time > total_available_time and not allow_repeat:
        raise ValueError(f'Not enough time to distribute injections, required time: {required_time} s, available time: {total_available_time} s')
    
    logger.info(f'Distributing {n_inj} injections in {total_available_time} s with rate {rate} and jitter {jitter}')
    
    n_inj_in_each_repeat = n_inj
    n_data_repeat = 1
    if required_time > total_available_time:
        n_inj_in_each_repeat = int(total_available_time * rate)
        n_data_repeat = ceil(n_inj / n_inj_in_each_repeat)
        logger.info(f'Using {n_data_repeat} data repeats to distribute {n_inj} injections')

    # shuffle injections
    np.random.shuffle(injections)

    # distribute injections
    gps_times = np.linspace(start_gps_time + interval/2, end_gps_time - interval/2, n_inj_in_each_repeat)
    if n_data_repeat > 1:
        gps_times = np.tile(gps_times, n_data_repeat)

        # remove the extra injections
        gps_times = gps_times[:n_inj]

    # add jitter
    gps_times += np.random.uniform(-jitter, jitter, n_inj)

    # add gps time and trail number to injections
    for i, inj in enumerate(injections):
        inj['gps_time'] = gps_times[i]
        inj['trail'] = i // n_inj_in_each_repeat

    return injections