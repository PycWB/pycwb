import warnings
import numpy as np
import random
from math import ceil
import logging
from pycwb.modules.injection.par_generator import get_injection_list_from_parameters, repeat
from pycwb.modules.injection.sky_distribution import generate_sky_distribution, distribute_injections_on_sky

logger = logging.getLogger(__name__)


def _normalize_injection_keys(injections: list) -> None:
    """Rename the deprecated ``'trail_idx'`` key to ``'trial_idx'`` in-place.

    Emits a single :class:`DeprecationWarning` if any injection dict still
    uses the old (misspelled) key, then renames it so all downstream code
    can assume ``'trial_idx'`` unconditionally.
    """
    found = False
    for inj in injections:
        if 'trail_idx' in inj:
            if not found:
                warnings.warn(
                    "Injection config uses the deprecated key 'trail_idx'; "
                    "rename it to 'trial_idx' in your injection configuration.",
                    DeprecationWarning,
                    stacklevel=3,
                )
                found = True
            inj['trial_idx'] = inj.pop('trail_idx')


def generate_injection_list_from_config_for_job_segments(injection_config, job_segments):
    """Generate and schedule injections into concrete job/tape intervals.

    Each job segment contributes available analysis livetime.  Every generated
    injection is assigned to exactly one job via top-level ``job_id`` and
    ``shift`` fields, so shifted jobs add capacity before a new ``trial_idx``
    is needed instead of receiving duplicate copies of nominal-window
    injections.
    """
    seed = injection_config.get('seed', None)
    repeat_injection = injection_config.get('repeat_injection', None)
    sky_distribution = injection_config.get('sky_distribution', None)
    time_distribution = injection_config.get('time_distribution', None)

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        logger.info(f'Setting random seed to {seed} for injection generation reproducibility')

    injections = get_injection_list_from_parameters(injection_config)
    _normalize_injection_keys(injections)

    if repeat_injection:
        injections = repeat(injections, repeat_injection)
    # sim_idx is assigned before scheduling so the same id follows the
    # injection through job ownership, simulation summary, and matching.
    for idx, inj in enumerate(injections):
        inj['sim_idx'] = idx

    if sky_distribution:
        coordinate_system = sky_distribution.get('coordsys', 'icrs')
        sky_locations = generate_sky_distribution(sky_distribution, len(injections))
        distribute_injections_on_sky(injections, sky_locations, coordsys=coordinate_system)
    else:
        for inj in injections:
            if not (('ra' in inj and 'dec' in inj) or ('sky_loc' in inj and 'coordsys' in inj)):
                raise ValueError("Either 'ra' and 'dec' or 'sky_loc' and 'coordsys' must be specified in the injections when no sky_distribution is provided")

    # Build a single ordered livetime axis from all usable job intervals once.
    # Shifted jobs add capacity before a new trial_idx is needed.
    intervals = _job_segment_intervals(job_segments)
    if not intervals:
        raise ValueError("No available job-segment livetime to schedule injections")

    if not time_distribution:
        # Explicit GPS times — assign each injection to the first containing interval.
        for inj in injections:
            if 'gps_time' not in inj:
                raise ValueError(
                    "'gps_time' must be specified in the injections "
                    "when no time_distribution is provided"
                )
            gps = float(inj['gps_time'])
            for interval in intervals:
                if interval['start'] <= gps < interval['end']:
                    _assign_to_interval(inj, interval, gps, inj.get('trial_idx', 0))
                    break
            else:
                raise ValueError(
                    f"Injection at gps_time={gps} does not fall within any "
                    f"job segment's analysis window"
                )
        return injections, max(
            (int(inj.get('trial_idx', 0)) for inj in injections), default=0
        ) + 1

    # Time-distribution scheduling (rate or Poisson) over the combined intervals.
    dist_type = time_distribution.get('type')
    rate_raw = time_distribution.get('rate')
    rate = eval(rate_raw) if isinstance(rate_raw, str) else rate_raw
    if dist_type == 'rate':
        jitter = time_distribution.get('jitter', 0)
        return distribute_inj_in_job_intervals_by_rate(
            injections, rate, jitter, intervals, shuffle=False,
        )
    if dist_type == 'poisson':
        max_trail = time_distribution.get('max_trail')
        return distribute_inj_in_job_intervals_by_poisson(
            injections, rate, intervals, max_trail=max_trail, shuffle=False,
        )
    if dist_type == 'custom':
        raise ValueError(
            'Custom time distribution is not supported anymore; '
            'remove the time_distribution field and provide explicit gps_time instead'
        )
    raise ValueError(f"Unknown time distribution type {dist_type!r}; only 'rate' and 'poisson' are supported")


def _job_segment_intervals(job_segments):
    """Return usable scheduling intervals from concrete job segments.

    Each interval keeps the nominal analysis-time bounds plus the owning
    ``job_id`` and superlag ``shift``.  CAT2 ``veto_windows`` are treated as
    keep windows, so they reduce the available livetime used for injection
    placement.
    """
    intervals = []
    for seg in sorted(job_segments, key=lambda s: (s.analyze_start, s.index)):
        shift = list(seg.shift) if seg.shift is not None else [0.0 for _ in seg.ifos]
        if seg.veto_windows:
            keep_windows = []
            for start, end in sorted(seg.veto_windows):
                lo = max(float(seg.analyze_start), float(start))
                hi = min(float(seg.analyze_end), float(end))
                if hi > lo:
                    keep_windows.append((lo, hi))
        else:
            keep_windows = [(float(seg.analyze_start), float(seg.analyze_end))]

        for start, end in keep_windows:
            if end <= start:
                continue
            intervals.append({
                'start': start,
                'end': end,
                'duration': end - start,
                'job_id': int(seg.index),
                'shift': shift,
            })
    return intervals


def _assign_to_interval(injection, interval, gps_time, trial_idx):
    """Attach scheduling ownership fields to one injection in-place."""
    injection['gps_time'] = float(gps_time)
    injection['trial_idx'] = int(trial_idx)
    injection['job_id'] = int(interval['job_id'])
    injection['shift'] = list(interval['shift'])


def _position_to_interval(intervals, position):
    """Map an offset on the combined livetime axis to ``(interval, gps_time)``.

    ``position`` is measured in seconds after concatenating all job intervals
    in scheduler order.  The returned GPS time is local to the owning interval.
    """
    pos = float(position)
    elapsed = 0.0
    for interval in intervals:
        next_elapsed = elapsed + interval['duration']
        if pos < next_elapsed or interval is intervals[-1]:
            return interval, interval['start'] + (pos - elapsed)
        elapsed = next_elapsed
    # Unreachable — loop always returns via `interval is intervals[-1]` guard.
    raise RuntimeError("position_to_interval: no interval found")


def distribute_inj_in_job_intervals_by_rate(injections, rate, jitter, intervals, shuffle=True):
    """Schedule injections at fixed rate over combined job livetime.

    If one pass over all intervals cannot hold every injection at the requested
    rate, the same interval set is reused with increasing ``trial_idx``.  Each
    injection still receives exactly one owning ``job_id``.
    """
    interval = 1 / rate
    if jitter > interval / 2:
        raise ValueError('Jitter is too large')

    total_available_time = sum(item['duration'] for item in intervals)
    n_inj = len(injections)
    required_time = n_inj / rate

    if interval > total_available_time:
        raise ValueError(f'Rate is too large, required available time: {interval} s for rate {rate}, but only {total_available_time} s available')

    if required_time > total_available_time:
        n_inj_in_each_trial = int(total_available_time * rate)
        if n_inj_in_each_trial <= 0:
            raise ValueError(f'Not enough livetime to schedule injections at rate {rate}')
        n_data_repeat = ceil(n_inj / n_inj_in_each_trial)
        logger.info(
            'Using %d trials to distribute %d injections across %.3f s of job livetime; each trial contains %d injections',
            n_data_repeat, n_inj, total_available_time, n_inj_in_each_trial,
        )
    else:
        n_inj_in_each_trial = n_inj
        n_data_repeat = 1

    if shuffle:
        np.random.shuffle(injections)
        logger.info('Shuffling injections before distributing in time')

    # Positions are laid out on the concatenated livetime axis, then mapped
    # back to the owning job interval below.
    positions = np.linspace(interval / 2, total_available_time - interval / 2, n_inj_in_each_trial)
    if n_data_repeat > 1:
        positions = np.tile(positions, n_data_repeat)[:n_inj]

    for i, inj in enumerate(injections):
        trial_idx = i // n_inj_in_each_trial
        interval_info, gps_time = _position_to_interval(intervals, positions[i])
        if jitter:
            gps_time += np.random.uniform(-jitter, jitter)
            gps_time = min(max(gps_time, interval_info['start']), interval_info['end'])
        _assign_to_interval(inj, interval_info, gps_time, trial_idx)

    return injections, n_data_repeat


def distribute_inj_in_job_intervals_by_poisson(injections, rate, intervals, max_trail=None, shuffle=True):
    """Schedule injections as a Poisson process over each trial's job intervals."""
    if shuffle:
        np.random.shuffle(injections)
        logger.info('Shuffling injections before distributing in time')

    injection_idx = 0
    trial_idx = 0
    scheduled = []
    n_inj = len(injections)

    while injection_idx < n_inj:
        for interval in intervals:
            t = interval['start'] + np.random.exponential(1 / rate)
            while t < interval['end']:
                inj = injections[injection_idx]
                _assign_to_interval(inj, interval, t, trial_idx)
                scheduled.append(inj)
                injection_idx += 1
                if injection_idx >= n_inj:
                    break
                t += np.random.exponential(1 / rate)
            # injection_idx >= n_inj already checked in inner while; outer
            # break is redundant when the inner break already fired.
            if injection_idx >= n_inj:
                break

        trial_idx += 1
        if max_trail and trial_idx >= max_trail:
            break

    return scheduled, trial_idx


def generate_auxiliary_injection_list_from_config(injection_config, start_gps_time, end_gps_time, n_trials):
    repeat_injection = injection_config['repeat_injection']
    sky_distribution = injection_config['sky_distribution']

    injections = get_injection_list_from_parameters(injection_config)
    injections = repeat(injections, repeat_injection)
    # Keep auxiliary injections on the same id convention as regular SIM rows.
    for idx, inj in enumerate(injections):
        inj['sim_idx'] = idx
    if sky_distribution:
        sky_locations = generate_sky_distribution(sky_distribution, len(injections))
        distribute_injections_on_sky(injections, sky_locations)

    if injection_config['time_distribution'] == 'rate':
        rate = eval(injection_config['rate']) if type(injection_config['rate']) == str else injection_config['rate']
        jitter = injection_config['jitter']
        injections, n_trials = distribute_inj_in_gps_time_by_rate(injections, rate, jitter, start_gps_time, end_gps_time, shuffle=False)
    elif injection_config['time_distribution'] == 'poisson':
        pass
    elif injection_config['time_distribution'] == 'custom':
        pass
    else:
        raise ValueError('Unknown time distribution, only support rate, poisson, custom')

    return injections, n_trials


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
        logger.info(f'Using {n_data_repeat} data repeats to distribute {n_inj} injections, each trial contains {n_inj_in_each_repeat} injections')

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

    # add gps time and trial number to injections
    for i, inj in enumerate(injections):
        inj['gps_time'] = gps_times[i]
        inj['trial_idx'] = i // n_inj_in_each_repeat

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
    
    if shuffle:
        np.random.shuffle(injections)
        logger.info('Shuffling injections before distributing in time')

    n_inj = len(injections)

    injection_idx = 0
    transient_signal = []

    trial_idx = 0

    while injection_idx < n_inj:
        t = start_gps_time + edge_buffer + np.random.exponential(1/rate)
        while t < end_gps_time - edge_buffer:
            par = {
                'gps_time': t + np.random.exponential(1/rate),
                'trial_idx': trial_idx,
            }
            transient_signal.append(injections[injection_idx] | par)
            injection_idx += 1
            t += np.random.exponential(1/rate)

            if injection_idx >= n_inj:
                break

        trial_idx += 1
        if max_trail and trial_idx >= max_trail:
            break

    return transient_signal, trial_idx
