import numpy as np
import random
from math import ceil
import logging
from pycwb.modules.injection.par_generator import get_injection_list_from_parameters, repeat
from pycwb.modules.injection.sky_distribution import generate_sky_distribution, distribute_injections_on_sky

logger = logging.getLogger(__name__)


def generate_injection_list_from_config(injection_config, start_gps_time, end_gps_time):
    seed = injection_config.get('seed', None)
    repeat_injection = injection_config.get('repeat_injection', None)
    sky_distribution = injection_config.get('sky_distribution', None)
    time_distribution = injection_config.get('time_distribution', None)

    # set random seed for reproducibility if specified
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        logger.info(f'Setting random seed to {seed} for injection generation reproducibility')

    injections = get_injection_list_from_parameters(injection_config)

    # repeat injections if specified
    if repeat_injection:
        injections = repeat(injections, repeat_injection)

    # distribute injections on sky if specified
    if sky_distribution:
        coordinate_system = sky_distribution.get('coordsys', 'icrs') 
        sky_locations = generate_sky_distribution(sky_distribution, len(injections))
        distribute_injections_on_sky(injections, sky_locations, coordsys = coordinate_system)
    else:
        # check if 'ra' and 'dec' are in injection or 'sky_loc' and 'coordsys' is specified
        for inj in injections:
            if not (('ra' in inj and 'dec' in inj) or ('sky_loc' in inj and 'coordsys' in inj)):
                raise ValueError("Either 'ra' and 'dec' or 'sky_loc' and 'coordsys' must be specified in the injections when no sky_distribution is provided")

    # distribute injections in GPS time
    if time_distribution:
        time_distribution_type = time_distribution.get('type', None)
        if time_distribution_type == 'rate':
            time_distribution_rate = time_distribution.get('rate', None)
            rate = eval(time_distribution_rate) if type(time_distribution_rate) == str else time_distribution_rate
            jitter = time_distribution.get('jitter', 0)
            injections, n_trails = distribute_inj_in_gps_time_by_rate(injections, rate, jitter, start_gps_time, end_gps_time, shuffle=False)
        elif time_distribution_type == 'poisson':
            time_distribution_rate = time_distribution.get('rate', None)
            rate = eval(time_distribution_rate) if type(time_distribution_rate) == str else time_distribution_rate
            max_trail = time_distribution.get('max_trail', None)
            injections, n_trails = distribute_inj_in_gps_time_by_poisson(injections, rate, start_gps_time, end_gps_time, max_trail=max_trail, shuffle=False)
        elif time_distribution_type == 'custom':
            raise ValueError('Custom time distribution is not supported anymore, if you want to use customized gps_time, simply remove the time_distribution field from the config')
        else:
            raise ValueError('Unknown time distribution, only support rate, poisson')
    else:
        # check if 'gps_time' is in injection
        n_trails = 1
        for inj in injections:
            if 'gps_time' not in inj:
                raise ValueError("'gps_time' must be specified in the injections when no time_distribution is provided")
            # find the maximum trail index if 'trail_idx' is specified
            if 'trail_idx' in inj:
                n_trails = max(n_trails, inj['trail_idx'] + 1)

    return injections, n_trails


def generate_auxiliary_injection_list_from_config(injection_config, start_gps_time, end_gps_time, n_trails):
    repeat_injection = injection_config['repeat_injection']
    sky_distribution = injection_config['sky_distribution']

    injections = get_injection_list_from_parameters(injection_config)
    injections = repeat(injections, repeat_injection)
    if sky_distribution:
        sky_locations = generate_sky_distribution(sky_distribution, len(injections))
        distribute_injections_on_sky(injections, sky_locations)

    if injection_config['time_distribution'] == 'rate':
        rate = eval(injection_config['rate']) if type(injection_config['rate']) == str else injection_config['rate']
        jitter = injection_config['jitter']
        injections, n_trails = distribute_inj_in_gps_time_by_rate(injections, rate, jitter, start_gps_time, end_gps_time, shuffle=False)
    elif injection_config['time_distribution'] == 'poisson':
        pass
    elif injection_config['time_distribution'] == 'custom':
        pass
    else:
        raise ValueError('Unknown time distribution, only support rate, poisson, custom')

    return injections, n_trails


def distribute_inj_in_gps_time_by_rate(injections, rate, jitter, 
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

    return injections, n_data_repeat


def distribute_inj_in_gps_time_by_poisson(injections, rate, start_gps_time, end_gps_time, edge_buffer = 0, max_trail = None, 
                                          shuffle=True):
    """
    Distribute injections in GPS time with a Poisson distribution. 

    :param injections: The list of injections
    :param rate: The rate of injections
    :param start_gps_time: The start GPS time
    :param end_gps_time: The end GPS time
    :param edge_buffer: The buffer time at the start and end of the data
    :param max_trails: The maximum number of trails for the case of auxiliary injections
    """
    
    # if incoherence and ifos is None:
    #     raise ValueError('ifos must be provided for incoherent injections')
    
    # shuffle injections
    if shuffle:
        np.random.shuffle(injections)
        logger.info('Shuffling injections before distributing in time')

    n_inj = len(injections)
    # # estimate if the number of injections is enough
    # est_avail_inj_time = n_inj / rate * len(ifos) if incoherence else n_inj / rate
    # total_available_time = ((end_gps_time - edge_buffer) - (start_gps_time + edge_buffer)) * n_trails
    # if est_avail_inj_time < total_available_time:
    #     logger.warning(f'Not enough injections to distribute in time, required time: {est_avail_inj_time} s, available time: {total_available_time} s')
    
    injection_idx = 0
    transient_signal = []

    trail_idx = 0

    while injection_idx < n_inj:
        t = start_gps_time + edge_buffer + np.random.exponential(1/rate)
        while t < end_gps_time - edge_buffer:
            par = {
                'gps_time': t + np.random.exponential(1/rate),
                'trail_idx': trail_idx,
            }
            transient_signal.append(injections[injection_idx] | par)
            injection_idx += 1
            t += np.random.exponential(1/rate)

            if injection_idx >= n_inj:
                break
                # raise ValueError('Not enough injections to distribute in time')
            
        trail_idx += 1
        if max_trail and trail_idx >= max_trail:
            break

    return transient_signal, trail_idx