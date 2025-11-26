import matplotlib.pyplot as plt  # pyright: ignore[reportMissingModuleSource]
import numpy as np  # pyright: ignore[reportMissingImports]
from pycwb.modules.post_production.waveform_reconstruction import * 


#Put in the same module as PostProcess Functions  

plot_kwargs = { 'CL_color': 'lightgray',
                'CL_alpha': 0.5,
                'injected_color': 'black',
                'injected_linestyle': '-',
                'injected_alpha': .3,
                'median_color': 'blue',
                'median_linestyle': '--',
                'mean_color': 'red',
                'mean_linestyle': '-.',
                'figsize': (10, 6),
                'fontsize': 12, 
}       



def plot_waveform_reconstruction(reconstructed, injected, domain, confidence_level = .95, percentile_method = 'percentiles', plot_median = False):
    """Plot the time domain waveform reconstruction with confidence intervals.
    """
    #initialise image 
    fig, ax = plt.subplots(figsize=plot_kwargs['figsize'])  

    if domain == 'frequency':
        reconstructed = np.abs(reconstructed)
        injected = np.abs(injected) 
        x_values = np.fft.fftfreq(len(injected.data), d=injected._delta_t)
        x_label = 'Frequency [Hz]'

    elif domain == 'time': 
        x_values = injected.sample_times
        x_label = 'GPS Time [s]'

    else: 
        raise ValueError("Domain must be 'time' or 'frequency'")

    #Compute relevant statistics 
    lower_bound, upper_bound = compute_confidence_intervals(reconstructed, confidence_level, method = percentile_method)  
    mean = np.mean(reconstructed, axis=0)

    #plot the confidence intervals, mean and median 
    ax.fill_between(x_values, lower_bound, upper_bound, color=plot_kwargs['CL_color'], alpha=plot_kwargs['CL_alpha'], label=f'{confidence_level}% CI') 
    ax.plot(x_values, mean, label='Mean', color=plot_kwargs['mean_color'], linestyle=plot_kwargs['mean_linestyle'])
    ax.plot(x_values, injected, label='Injected', color=plot_kwargs['injected_color'], linestyle=plot_kwargs['injected_linestyle'], alpha=plot_kwargs['injected_alpha']) 
    if plot_median: 
        median = np.median(reconstructed, axis=0) 
        ax.plot(x_values, median, label='Median', color=plot_kwargs['median_color'], linestyle=plot_kwargs['median_linestyle'])

    #Visualization 
    ax.grid(True)
    ax.set_ylabel('Strain', fontsize=plot_kwargs['fontsize'])
    ax.legend(fontsize=plot_kwargs['fontsize'])
    ax.tick_params(labelsize=plot_kwargs['fontsize']) 
    ax.set_xlabel(x_label, fontsize=plot_kwargs['fontsize'])
    if domain == 'frequency':
        plt.xscale('log'), plt.yscale('log')




    #Define the "to save" dictionary
    to_save = {'time': injected.sample_times,
        'injected': injected,
        'mean': mean,
        'median': median,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'CL': confidence_level
    }
    plt.close() 



    return fig, to_save
def plot_time_waveform_reconstruction(reconstructed, injected, confidence_level = .95, percentile_method = 'percentiles', plot_median = False):
    """Plot the time-domain waveform reconstruction with confidence intervals."""
    fig, ax = plt.subplots(figsize=plot_kwargs['figsize'])
    reco_arr = np.asarray(reconstructed)
    lower_bound, upper_bound = compute_confidence_intervals(reco_arr, confidence_level, method = percentile_method)
    mean = np.mean(reco_arr, axis=0)
    ax.fill_between(injected.sample_times, lower_bound, upper_bound, color=plot_kwargs['CL_color'], alpha=plot_kwargs['CL_alpha'], label=f'{confidence_level}% CI')
    ax.plot(injected.sample_times, mean, label='Mean', color=plot_kwargs['mean_color'], linestyle=plot_kwargs['mean_linestyle'])
    ax.plot(injected.sample_times, injected, label='Injected', color=plot_kwargs['injected_color'], linestyle=plot_kwargs['injected_linestyle'], alpha=plot_kwargs['injected_alpha'])
    median = None
    if plot_median:
        median = np.median(reco_arr, axis=0)
        ax.plot(injected.sample_times, median, label='Median', color=plot_kwargs['median_color'], linestyle=plot_kwargs['median_linestyle'])
    ax.grid(True)
    ax.set_ylabel('Strain', fontsize=plot_kwargs['fontsize'])
    ax.legend(fontsize=plot_kwargs['fontsize'])
    ax.tick_params(labelsize=plot_kwargs['fontsize'])
    ax.set_xlabel('GPS Time [s]', fontsize=plot_kwargs['fontsize'])
    to_save = {'time': injected.sample_times,
        'injected': injected,
        'mean': mean,
        'median': median,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'CL': confidence_level
    }
    plt.close()
    return fig, to_save

def plot_frequency_waveform_reconstruction(reconstructed, injected_fft, confidence_level = .95, percentile_method = 'percentiles', plot_median = False):
    """Plot the frequency-domain waveform reconstruction (positive frequencies only) with confidence intervals."""
    fig, ax = plt.subplots(figsize=plot_kwargs['figsize'])
    reco_arr = np.abs(np.asarray(reconstructed))
    inj_arr = np.abs(np.asarray(injected_fft))
    n = len(getattr(injected_fft, 'data', inj_arr))
    dt = getattr(injected_fft, '_delta_t', None)
    x_values = np.fft.fftfreq(n, d=dt)
    pos_mask = x_values >= 0
    x_pos = x_values[pos_mask]
    if reco_arr.ndim == 1:
        reco_pos = reco_arr[pos_mask]
    else:
        reco_pos = reco_arr[:, pos_mask]
    if inj_arr.ndim == 1:
        inj_pos = inj_arr[pos_mask]
    else:
        inj_pos = inj_arr[..., pos_mask]
    lower_bound, upper_bound = compute_confidence_intervals(reco_pos, confidence_level, method = percentile_method)
    mean = np.mean(reco_pos, axis=0)
    ax.fill_between(x_pos, lower_bound, upper_bound, color=plot_kwargs['CL_color'], alpha=plot_kwargs['CL_alpha'], label=f'{confidence_level}% CI')
    ax.plot(x_pos, mean, label='Mean', color=plot_kwargs['mean_color'], linestyle=plot_kwargs['mean_linestyle'])
    ax.plot(x_pos, inj_pos, label='Injected', color=plot_kwargs['injected_color'], linestyle=plot_kwargs['injected_linestyle'], alpha=plot_kwargs['injected_alpha'])
    median = None
    if plot_median:
        median = np.median(reco_pos, axis=0)
        ax.plot(x_pos, median, label='Median', color=plot_kwargs['median_color'], linestyle=plot_kwargs['median_linestyle'])
    ax.grid(True)
    ax.set_ylabel('Strain (magnitude)', fontsize=plot_kwargs['fontsize'])
    ax.legend(fontsize=plot_kwargs['fontsize'])
    ax.tick_params(labelsize=plot_kwargs['fontsize'])
    ax.set_xlabel('Frequency [Hz]', fontsize=plot_kwargs['fontsize'])
    plt.xscale('log'); plt.yscale('log')
    to_save = {'frequency': x_pos,
        'injected': inj_pos,
        'mean': mean,
        'median': median,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'CL': confidence_level
    }
    plt.close()
    return fig, to_save


def plot_time_bias(reconstructed, injected, confidence_level = .95, percentile_method='percentiles', normalize = True):
    """Plot the bias in the time domain between reconstructed and injected waveforms."""
    x_values = injected.sample_times
    x_label = 'GPS Time [s]'

    bias = np.asarray(reconstructed) - np.asarray(injected)
    lower_bound, upper_bound = compute_confidence_intervals(bias, confidence_level, method=percentile_method)
    mean_bias = np.mean(bias, axis=0)

    fig, ax = plt.subplots(figsize=plot_kwargs['figsize'])
    ax.fill_between(x_values, lower_bound, upper_bound, color=plot_kwargs['CL_color'], alpha=plot_kwargs['CL_alpha'], label=f'{confidence_level}% CI')
    ax.plot(x_values, mean_bias, label='Mean Bias', color=plot_kwargs['mean_color'], linestyle=plot_kwargs['mean_linestyle'])
    ax.legend(fontsize=plot_kwargs['fontsize'])
    ax.grid(True)
    ax.tick_params(labelsize=plot_kwargs['fontsize'])
    ax.set_ylabel('Bias', fontsize=plot_kwargs['fontsize'])
    ax.set_xlabel(x_label, fontsize=plot_kwargs['fontsize'])

    to_save = {'time': injected.sample_times,
        'mean_bias': mean_bias,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'CL': confidence_level
    }
    plt.close()
    return fig, to_save


def plot_frequency_bias(reconstructed, injected_fft, confidence_level = .95, percentile_method='percentiles', normalize = True):
    """Plot the bias in the frequency domain (positive frequencies only)."""
    # Work with magnitudes
    reco_arr = np.abs(np.asarray(reconstructed))
    inj_arr = np.abs(np.asarray(injected_fft))

    n = len(getattr(injected_fft, 'data', inj_arr))
    dt = getattr(injected_fft, '_delta_t', None)
    x_values = np.fft.fftfreq(n, d=dt)
    pos_mask = x_values >= 0
    x_pos = x_values[pos_mask]


    bias = reco_arr - inj_arr
    lower_bound, upper_bound = compute_confidence_intervals(bias, confidence_level, method=percentile_method)
    mean_bias = np.mean(bias, axis=0)

    fig, ax = plt.subplots(figsize=plot_kwargs['figsize'])
    ax.fill_between(x_pos, lower_bound, upper_bound, color=plot_kwargs['CL_color'], alpha=plot_kwargs['CL_alpha'], label=f'{confidence_level}% CI')
    ax.plot(x_pos, mean_bias, label='Mean Bias', color=plot_kwargs['mean_color'], linestyle=plot_kwargs['mean_linestyle'])
    ax.legend(fontsize=plot_kwargs['fontsize'])
    ax.grid(True)
    ax.tick_params(labelsize=plot_kwargs['fontsize'])
    ax.set_ylabel('Bias', fontsize=plot_kwargs['fontsize'])
    ax.set_xlabel('Frequency [Hz]', fontsize=plot_kwargs['fontsize'])
    plt.xscale('log'); plt.yscale('log')

    to_save = {'frequency': x_pos,
        'mean_bias': mean_bias,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'CL': confidence_level
    }
    plt.close()
    return fig, to_save




def plot_overlap(reconstructed, injected, plot_median = True): 
    """
    Plot the overlap between the reconstructed waveforms and the injected waveform.
    """
    #Compute the overlap distribution of all waveforms 
    overlaps = compute_overlap(reconstructed, injected) 

    #Plot the histogram of the overlaps
    fig, ax = plt.subplots(figsize=plot_kwargs['figsize'])
    bins = int(np.sqrt(len(overlaps)))
    ax.hist(overlaps, bins=bins, density=True, histtype='step', label='Reconstructed Overlap', color=plot_kwargs['injected_color']) 

    #If plot mean or median, compute the overlap of the mean (median) waveform and plot it 
    mean_reconstructed = np.mean(reconstructed, axis=0)
    mean_overlap = compute_overlap(mean_reconstructed, injected)
    ax.axvline(mean_overlap, color=plot_kwargs['mean_color'], linestyle=plot_kwargs['mean_linestyle'], label='Overlap of the mean') 

    if plot_median:
        median_reconstructed = np.median(reconstructed, axis=0) 
        median_overlap = compute_overlap(median_reconstructed, injected)
        ax.axvline(median_overlap, color=plot_kwargs['median_color'], linestyle=plot_kwargs['median_linestyle'], label='Overlap of the median')

    #Set the labels and legend 
    ax.set_xlabel('Overlap', fontsize=plot_kwargs['fontsize'])
    ax.set_ylabel('Density', fontsize=plot_kwargs['fontsize'])
    ax.legend(fontsize=plot_kwargs['fontsize'])
    ax.grid(True)
    ax.tick_params(labelsize=plot_kwargs['fontsize'])

    to_save = {'overlaps': overlaps}
    to_save['mean_overlap'] = mean_overlap
    to_save['median_overlap'] = median_overlap if plot_median else None
    return fig, to_save


def plot_auto_overlap(reconstructed, injected):
    """
    Plot the overlap between the reconstructed waveform and its correspondent injected waveform. 
    It differs from plot_overlap as it computes the overlap between each reconstructed waveform and its correspondent injected waveform. 
    """
    #Compute the overlap distribution of all waveforms 
    overlaps = compute_overlap(reconstructed, injected)
    mean_overlap = np.mean(overlaps) 

    #Plot the histogram of the overlaps
    fig, ax = plt.subplots(figsize=plot_kwargs['figsize'])
    bins = int(np.sqrt(len(overlaps)))
    ax.hist(overlaps, bins=bins, density=True, histtype='step', color=plot_kwargs['injected_color'])
    ax.axvline(mean_overlap, color=plot_kwargs['mean_color'], linestyle=plot_kwargs['mean_linestyle'], label='Mean Overlap') 
    ax.set_xlabel('Overlap', fontsize=plot_kwargs['fontsize'])
    ax.set_ylabel('Density', fontsize=plot_kwargs['fontsize'])
    ax.grid(True)
    ax.tick_params(labelsize=plot_kwargs['fontsize'])

    to_save = {'overlaps': overlaps}  
    return fig, to_save 


def plot_time_cumulative_hrss(reconstructed, injected, confidence_level, percentile_method, plot_median=True):
    """Plot the cumulative distribution of HRSS in the time domain."""
    # Compute relevant statistics
    injected_hrss = compute_cumulative_hrss(injected, injected._delta_t, axis=0)
    reconstructed_hrss = compute_cumulative_hrss(reconstructed, injected._delta_t, axis=1) / injected_hrss[-1]
    mean_hrss = reconstructed_hrss.mean(axis=0)
    lower_bound, upper_bound = compute_confidence_intervals(reconstructed_hrss, confidence_level=confidence_level, method=percentile_method)

    x_values = injected.sample_times
    x_label = 'GPS Time [s]'

    fig, ax = plt.subplots(figsize=plot_kwargs['figsize'])
    ax.plot(x_values, injected_hrss / injected_hrss[-1], label='Injected HRSS', color=plot_kwargs['injected_color'], linestyle=plot_kwargs['injected_linestyle'])
    ax.plot(x_values, mean_hrss, label='Mean Reconstructed HRSS', color=plot_kwargs['mean_color'], linestyle=plot_kwargs['mean_linestyle'])
    ax.fill_between(x_values, lower_bound, upper_bound, color=plot_kwargs['CL_color'], alpha=plot_kwargs['CL_alpha'], label=f'{confidence_level}% CI')

    median_hrss = None
    if plot_median:
        median_hrss = np.median(reconstructed_hrss, axis=0)
        ax.plot(x_values, median_hrss, label='Median Reconstructed HRSS', color=plot_kwargs['median_color'], linestyle=plot_kwargs['median_linestyle'])

    # Visualization
    ax.grid(True)
    ax.legend(fontsize=plot_kwargs['fontsize'])
    ax.tick_params(labelsize=plot_kwargs['fontsize'])
    ax.set_ylabel('Cumulative HRSS (normalized)', fontsize=plot_kwargs['fontsize'])
    ax.set_xlabel(x_label, fontsize=plot_kwargs['fontsize'])

    to_save = {'time': injected.sample_times,
      'injected_hrss': injected_hrss / injected_hrss[-1],
      'mean_hrss': mean_hrss,
      'median_hrss': median_hrss,
      'lower_bound': lower_bound,
      'upper_bound': upper_bound,
      'CL': confidence_level
    }

    plt.close()
    return fig, to_save


def plot_frequency_cumulative_hrss(reconstructed, injected_fft, confidence_level, percentile_method, plot_median=True):
    """Plot the cumulative distribution of HRSS in the frequency domain (positive frequencies only)."""
    # Work with magnitudes for frequency-domain HRSS
    inj_arr = np.abs(np.asarray(injected_fft))

    n = len(getattr(injected_fft, 'data', inj_arr))
    dt = getattr(injected_fft, '_delta_t', None)
    x_values = np.fft.fftfreq(n, d=dt)
    pos_mask = x_values >= 0
    x_pos = x_values[pos_mask]

    injected_hrss = compute_cumulative_hrss(injected_fft, dt, axis=0)
    reconstructed_hrss = compute_cumulative_hrss(reconstructed, dt, axis=1) / injected_hrss[-1]

    # select positive frequencies
    injected_hrss_pos = injected_hrss[pos_mask]
    reconstructed_hrss_pos = reconstructed_hrss[..., pos_mask]

    mean_hrss = reconstructed_hrss_pos.mean(axis=0)
    lower_bound, upper_bound = compute_confidence_intervals(reconstructed_hrss_pos, confidence_level=confidence_level, method=percentile_method)

    fig, ax = plt.subplots(figsize=plot_kwargs['figsize'])
    ax.plot(x_pos, injected_hrss_pos / injected_hrss_pos[-1], label='Injected HRSS', color=plot_kwargs['injected_color'], linestyle=plot_kwargs['injected_linestyle'])
    ax.plot(x_pos, mean_hrss, label='Mean Reconstructed HRSS', color=plot_kwargs['mean_color'], linestyle=plot_kwargs['mean_linestyle'])
    ax.fill_between(x_pos, lower_bound, upper_bound, color=plot_kwargs['CL_color'], alpha=plot_kwargs['CL_alpha'], label=f'{confidence_level}% CI')

    median_hrss = None
    if plot_median:
        median_hrss = np.median(reconstructed_hrss_pos, axis=0)
        ax.plot(x_pos, median_hrss, label='Median Reconstructed HRSS', color=plot_kwargs['median_color'], linestyle=plot_kwargs['median_linestyle'])

    # Visualization
    ax.grid(True)
    ax.legend(fontsize=plot_kwargs['fontsize'])
    ax.tick_params(labelsize=plot_kwargs['fontsize'])
    ax.set_ylabel('Cumulative HRSS (normalized)', fontsize=plot_kwargs['fontsize'])
    ax.set_xlabel('Frequency [Hz]', fontsize=plot_kwargs['fontsize'])
    plt.xscale('log'); plt.yscale('log')

    to_save = {'frequency': x_pos,
      'injected_hrss': injected_hrss_pos / injected_hrss_pos[-1],
      'mean_hrss': mean_hrss,
      'median_hrss': median_hrss,
      'lower_bound': lower_bound,
      'upper_bound': upper_bound,
      'CL': confidence_level
    }

    plt.close()
    return fig, to_save


def plot_hrss_histogram(reconstructed, injected): 
    """Plot the scatter of the reconstructed HRSS compared to the injected HRSS.
    """
    injected_hrss, reconstructed_hrss = [], []
    #Compute the hrss of all waveforms 
    for i, reconstructed_waveform in enumerate(reconstructed): 
        reconstructed_hrss.append(compute_hrss(reconstructed_waveform.data, injected[i]._delta_t))
    injected_hrss.append(compute_hrss(injected.data, injected._delta_t))

    #Plot hrss histogram 
    fig, ax = plt.subplots(figsize=plot_kwargs['figsize'])
    bins = int(np.sqrt(len(reconstructed_hrss)))
    ax.hist(reconstructed_hrss, bins=30, density=True, histtype='step', label='Reconstructed HRSS', color=plot_kwargs['injected_color'])
    ax.axvline(injected_hrss[0], color=plot_kwargs['mean_color'], linestyle=plot_kwargs['mean_linestyle'], label='Injected HRSS') 
    #Set the labels and legend 
    ax.set_xlabel('Injected signal energy', fontsize=plot_kwargs['fontsize'])
    ax.set_ylabel('Reconstructed signal energy', fontsize=plot_kwargs['fontsize'])
    ax.legend(fontsize=plot_kwargs['fontsize'])
    ax.grid(True)
    ax.tick_params(labelsize=plot_kwargs['fontsize'])


    to_save = {'injected_hrss': injected_hrss,
        'reconstructed_hrss': reconstructed_hrss
    }
    return fig, to_save


def plot_leakage(reconstructed, injected):
    """
    Plot the leakage of the reconstructed waveforms compared to the injected waveform.
    """

    #compute leakage over a specified     
    t = np.arange(0, 20, 1) / 20 
    leakage_mean, leakage_std = compute_leakage(reconstructed, injected, t)

    #Plot Reconstructed waveform over the 1st axis and Leakage(time) over the 2nd axis  
    fig, ax = plt.subplots(figsize=plot_kwargs['figsize'])
    ax.errorbar(t, leakage_mean, yerr=leakage_std, fmt='o', color=plot_kwargs['injected_color'], label='Leakage') 

    #Set the labels and legend 
    ax.set_xlabel('Time [s] - Injection end time', fontsize=plot_kwargs['fontsize'])
    ax.set_ylabel('Leakage', fontsize=plot_kwargs['fontsize'])
    ax.legend(fontsize=plot_kwargs['fontsize'])
    ax.grid(True)
    ax.tick_params(labelsize=plot_kwargs['fontsize'])

    to_save = {'leakage_mean': leakage_mean,
        'leakage_std': leakage_std,
        'time': t
    }
    return fig, to_save

def plot_hrss(reconstruced, injected): 
    """
    Plot the HRSS of the reconstructed waveforms compared to the injected waveform. 
    """ 
    reconstructed_hrss = [] 
    for waveform in reconstruced: 
        reconstructed_hrss.append(compute_hrss(waveform.data, injected._delta_t))
    injected_hrss = compute_hrss(injected.data, injected._delta_t) 

    fig, ax = plt.subplots(figsize=plot_kwargs['figsize'])
    ax.hist(reconstructed_hrss, bins=30, density=True, histtype='step', label='Reconstructed HRSS', color=plot_kwargs['injected_color']) 
    ax.axvline(injected_hrss, color=plot_kwargs['mean_color'], linestyle=plot_kwargs['mean_linestyle'], label='Injected HRSS') 

    #Set the labels and legend 
    ax.set_xlabel('hrss', fontsize=plot_kwargs['fontsize'])
    ax.set_ylabel('Density', fontsize=plot_kwargs['fontsize'])
    ax.legend(fontsize=plot_kwargs['fontsize'])
    ax.grid(True)
    ax.tick_params(labelsize=plot_kwargs['fontsize'])

    to_save = {'reconstructed_hrss': reconstructed_hrss,
        'injected_hrss': injected_hrss
    }
    return fig, to_save