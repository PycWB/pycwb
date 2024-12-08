import matplotlib.pyplot as plt
import numpy as np


def report(far_rho_source=None, **kwargs):

    print(f"Reporting the results")
    if far_rho_source:
        if far_rho_source not in kwargs:
            print(f"Source {far_rho_source} not found in the results")
            raise ValueError(f"Source {far_rho_source} not found in the results")
        data = kwargs[far_rho_source]

        # plot the far vs ranking parameter
        plt.plot(data['bins'], data['far'], drawstyle='steps-post')
        plt.xlabel(data['ranking_par'])
        plt.ylabel('far')
        plt.yscale('log')
        plt.savefig(f"{far_rho_source}.png")
        plt.close()

        # plot the number of events vs ranking parameter
        plt.plot(data['bins'], data['n_events'], drawstyle='steps-post')
        plt.xlabel(data['ranking_par'])
        plt.ylabel('number of events')
        plt.yscale('log')
        plt.savefig(f"{far_rho_source}_n_events.png")
        plt.close()


def report_zero_lag(source, livetime_key='livetime_zerolag', far_rho_source='far_rho', **kwargs):
    print(f"Reporting the zero lag results")
    if source not in kwargs:
        print(f"Source {source} not found in the results")
        raise ValueError(f"Source {source} not found in the results")
    triggers = kwargs[source]

    if livetime_key not in kwargs:
        print(f"Live time key {livetime_key} not found in the results")
        raise ValueError(f"Live time key {livetime_key} not found in the results")
    livetime = kwargs[livetime_key]

    if far_rho_source not in kwargs:
        print(f"Source {far_rho_source} not found in the results")
        raise ValueError(f"Source {far_rho_source} not found in the results")
    far_rho_data = kwargs[far_rho_source]

    # attach far to the triggers if the trigger's ranking_par is within the range of bin
    ranking_par = far_rho_data['ranking_par']
    bins = far_rho_data['bins']
    far = far_rho_data['far']

    if '[' not in ranking_par:
        values = [trigger[ranking_par] for trigger in triggers]
    else:
        base_key, index = ranking_par.split('[')
        index = int(index.rstrip(']'))
        values = [trigger[base_key][index] for trigger in triggers]

    # align the values to the bin
    bin_indices = np.digitize(values, bins) - 1
    far_values = [far[i] for i in bin_indices]
    for i, trigger in enumerate(triggers):
        trigger['far'] = far_values[i]

    # plot the triggers far vs ranking parameter
    print(f"Plotting far vs {ranking_par}")
    plt.scatter(values, far_values)
    plt.xlabel(ranking_par)
    plt.ylabel('far')
    plt.yscale('log')
    plt.savefig(f"{source}_far.png")
    plt.close()

    print(f"Plotting far vs n_events accumulative distribution")
    # plot accumulated triggers vs trigger['far']
    ranked_triggers = sorted(triggers, key=lambda x: 1/x['far'], reverse=True)
    ifar = np.array([1/trigger['far'] for trigger in ranked_triggers])
    plot_y = np.arange(1, len(ranked_triggers) + 1)
    plt.plot(ifar, plot_y, drawstyle='steps-post')

    livetime_in_years = livetime / 86400 / 365.25
    ifar_min = min(ifar)
    n_events_at_ifar_min = livetime_in_years / ifar_min
    ifar_max = max(ifar)
    n_events_at_ifar_max = livetime_in_years / ifar_max
    plt.plot([ifar_min, ifar_max], [n_events_at_ifar_min, n_events_at_ifar_max], color='black')
    print(f"ifar_min: {ifar_min}, n_events_at_ifar_min: {n_events_at_ifar_min}")
    print(f"ifar_max: {ifar_max}, n_events_at_ifar_max: {n_events_at_ifar_max}")

    plt.xlabel('ifar')
    plt.ylabel('n_events')
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(f"{source}_far_n_events.png")
    plt.close()
