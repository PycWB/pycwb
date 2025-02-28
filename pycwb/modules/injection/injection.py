import numpy as np
from math import ceil
import logging
from pycwb.modules.injection.par_generator import get_injection_list_from_parameters, repeat
from pycwb.modules.injection.sky_distribution import generate_sky_distribution, distribute_injections_on_sky

logger = logging.getLogger(__name__)


def generate_injection_list_from_config(injection_config, start_gps_time, end_gps_time):
    repeat_injection = injection_config['repeat_injection']
    rate = eval(injection_config['rate']) if type(injection_config['rate']) == str else injection_config['rate']
    jitter = injection_config['jitter']
    sky_distribution = injection_config['sky_distribution']

    injections = get_injection_list_from_parameters(injection_config)
    injections = repeat(injections, repeat_injection)
    sky_locations = generate_sky_distribution(sky_distribution, len(injections))
    distribute_injections_on_sky(injections, sky_locations)
    distribute_inj_in_gps_time(injections, rate, jitter, start_gps_time, end_gps_time, shuffle=False)

    return injections


def distribute_inj_in_gps_time(injections, rate, jitter, 
                               start_gps_time, end_gps_time, 
                               edge_buffer = 0,
                               shuffle=True,
                               allow_repeat=True):
    """
    Distribute injections in GPS time with a given rate and jitter.

    :param injections: The list of injections
    :param rate: The rate of injections
    :param jitter: The jitter of injections
    :param start_gps_time: The start GPS time
    :param end_gps_time: The end GPS time
    :param edge_buffer: The buffer time at the start and end of the data
    :param shuffle: Shuffle the injections before distributing in time
    :param allow_repeat: Allow repeating the injections if there is not enough time to distribute
    """
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
        logger.info(f'Using {n_data_repeat} data repeats to distribute {n_inj} injections, each trail contains {n_inj_in_each_repeat} injections')

    # shuffle injections
    if shuffle:
        np.random.shuffle(injections)
        logger.info('Shuffling injections before distributing in time')

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
        inj['trail_idx'] = i // n_inj_in_each_repeat

    return injections