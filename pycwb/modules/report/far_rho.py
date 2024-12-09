import numpy as np
import orjson


def far_rho(source, ranking_par, bin_size, livetime_key='livetime', save='far_rho.json', **kwargs):
    print(f"Calculating far and rho for source {source} with ranking parameter {ranking_par} with bin size {bin_size}")

    if source not in kwargs:
        print(f"Source {source} not found in the results")
        raise ValueError(f"Source {source} not found in the results")
    triggers = kwargs[source]

    if livetime_key not in kwargs:
        print(f"Live time key {livetime_key} not found in the results")
        raise ValueError(f"Live time key {livetime_key} not found in the results")
    livetime = kwargs[livetime_key]

    # extract the ranking parameter from the triggers
    if '[' not in ranking_par:
        values = [trigger[ranking_par] for trigger in triggers]
    else:
        base_key, index = ranking_par.split('[')
        index = int(index.rstrip(']'))
        values = [trigger[base_key][index] for trigger in triggers]

    # remove None in values and print warning with trigger job_id and id
    none_values = [triggers[i] for i, value in enumerate(values) if value is None]
    if none_values:
        print(f"Warning: {len(none_values)} triggers with None value in {ranking_par}")
        for none_value in none_values:
            print(none_value)

    values = [value for value in values if value is not None]

    # calculate the n_events vs rho
    hist, bins = np.histogram(values, bins=np.arange(min(values), max(values) + bin_size, bin_size))
    accumulate_hist = np.cumsum(hist[::-1])[::-1]
    bins = bins[:-1]

    # calculate the far for each bin
    livetime_in_years = livetime / 86400 / 365.25
    far = [accumulate_hist[i] / livetime_in_years for i in range(len(accumulate_hist))]
    data = {'bins': bins, 'far': far, 'n_events': hist,
            'ranking_par': ranking_par, 'source': source, 'livetime': livetime}

    if save:
        with open(save, 'wb') as f:
            f.write(orjson.dumps(data, option=orjson.OPT_SERIALIZE_NUMPY))

    return data