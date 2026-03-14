import matplotlib.pyplot as plt  # pyright: ignore[reportMissingModuleSource]
import numpy as np  # pyright: ignore[reportMissingImports]
from pycwb.modules.post_production.waveform_reconstruction import * 


#Put in the same module as PostProcess Functions  

plot_kwargs = { 'ref_color': 'darkgray', 
                'ref_ls': '-', 
                'inj_color': 'k', 
                'inj_ls': '-',
                'inj_alpha': .3,
                'plot_mean': True,
                'plot_median': True,
                'mean_linestyle': '-.',
                'CL_color': 'lightgray',
                'CL_alpha': 0.5, 
                'injected_alpha': .3,
                'median_color': 'blue',
                'median_linestyle': '--',
                'mean_color': 'red',
                'mean_linestyle': '-.',
                'figsize': (10, 6),
                'fontsize': 12, 
}       



def plot_time_waveform_reconstruction(reconstructed, reference = None, injected = None, confidence_level = .95, **kwargs): 
    """
    Plots the time-domain waveform reconstruction with confidence intervals.
    
    :param reconstructed: List of the reconstructed waveforms
    :param reference: The reference waveform (on-source)
    :param injected: The injected waveform
    :param confidence_level: The confidence level for the intervals
    :param kwargs: Additional keyword arguments for plot customization
    :return: A tuple containing the figure and a dictionary of data to save
    """
    plot_kwargs.update(kwargs) 
    to_save = {'CL': confidence_level, 'reference': reference,'injected': injected} 
    
    lower_bound, upper_bound = compute_confidence_intervals(reconstructed, confidence_level, method= kwargs['percentile_method'], reference_waveform=reference)
    time = reconstructed[0].sample_times.data 
    to_save.update({'time': time, 'lower_bound': lower_bound, 'upper_bound': upper_bound, 'CL': confidence_level})

    fig, ax = plt.subplots(figsize=plot_kwargs['figsize'])
    ax.fill_between(time, lower_bound, upper_bound, color=plot_kwargs['CL_color'], alpha=plot_kwargs['CL_alpha'], label=f'{confidence_level * 100}% CI')

    if reference: 
        ax.plot(time, reference, label='On Source', color=plot_kwargs['ref_color'], linestyle=plot_kwargs['ref_ls']) 
    
    if injected: 
        ax.plot(time, injected, label='Injected', color=plot_kwargs['inj_color'], linestyle=plot_kwargs['inj_ls'], alpha=plot_kwargs['inj_alpha'])

    if kwargs.get('plot_mean', True):
        mean = np.nanmean(reconstructed, axis=0)
        ax.plot(time, mean, label='Mean', color=plot_kwargs['mean_color'], linestyle=plot_kwargs['mean_linestyle']) 
        to_save.update({'mean': mean}) 

    if kwargs.get('plot_median', True): 
        median = np.nanmedian(reconstructed, axis=0) 
        ax.plot(time, median, label='Median', color=plot_kwargs['median_color'], linestyle=plot_kwargs['median_linestyle'])
        to_save.update({'median': median}) 

    ax.set_xlabel('GPS Time [s]', fontsize=plot_kwargs['fontsize'])
    ax.set_ylabel('Strain', fontsize=plot_kwargs['fontsize'])
    ax.legend(fontsize=plot_kwargs['fontsize'])
    ax.tick_params(labelsize=plot_kwargs['fontsize'])
    ax.grid(True)
    plt.close() 
    return fig, to_save 


def plot_frequency_waveform_reconstruction(reconstructed, reference = None, injected = None, confidence_level = .95, **kwargs):
    """
    Plots the frequency-domain waveform reconstruction with confidence intervals.
    
    :param reconstructed: List of the reconstructed waveforms
    :param reference: The reference waveform (on-source)
    :param injected: The injected waveform
    :param confidence_level: The confidence level for the intervals
    :param kwargs: Additional keyword arguments for plot customization
    :return: A tuple containing the figure and a dictionary of data to save
    """

    plot_kwargs.update(kwargs)
    to_save = {'CL': confidence_level, 'reference': reference,'injected': injected} 

    delta_t = getattr(reconstructed[0], '_delta_t', None)
    reconstructed = np.abs(reconstructed)
    reference = np.abs(reference) if reference is not None else None
    injected = np.abs(injected) if injected is not None else None

    lower_bound, upper_bound = compute_confidence_intervals(reconstructed, confidence_level, method=kwargs['percentile_method'],
                                                            reference_waveform=reference)
    frequency = np.fft.fftfreq(len(reference.data), d= delta_t)

    fig, ax = plt.subplots(figsize=plot_kwargs['figsize'])
    ax.fill_between(frequency, lower_bound, upper_bound, color=plot_kwargs['CL_color'], alpha=plot_kwargs['CL_alpha'], label=f'{confidence_level * 100}% CI')

    if reference: 
        ax.plot(frequency, reference, label='On Source', color=plot_kwargs['ref_color'], linestyle=plot_kwargs['ref_ls'])

    if injected:
        ax.plot(frequency, injected, label='Injected', color=plot_kwargs['inj_color'], linestyle=plot_kwargs['inj_ls'], alpha=plot_kwargs['inj_alpha'])
    
    if kwargs.get('plot_mean', True):
        mean = np.nanmean(reconstructed, axis=0)
        ax.plot(frequency, mean, label='Mean', color=plot_kwargs['mean_color'], linestyle=plot_kwargs['mean_linestyle'])
        to_save.update({'mean': mean})
    
    if kwargs.get('plot_median', True):
        median = np.nanmedian(reconstructed, axis=0)
        ax.plot(frequency, median, label='Median', color=plot_kwargs['median_color'], linestyle=plot_kwargs['median_linestyle'])
        to_save.update({'median': median})
    
    ax.set_xlabel('Frequency [Hz]', fontsize=plot_kwargs['fontsize'])
    ax.set_ylabel('Strain (magnitude)', fontsize=plot_kwargs['fontsize'])
    ax.legend(fontsize=plot_kwargs['fontsize'])
    ax.tick_params(labelsize=plot_kwargs['fontsize'])
    plt.xscale('log'); plt.yscale('log')
    ax.grid(True)
    to_save.update({'frequency': frequency, 'lower_bound': lower_bound, 'upper_bound': upper_bound, 'CL': confidence_level})
    plt.close()
    return fig, to_save




def plot_time_bias(reconstructed, reference, confidence_level = .95, **kwargs):

    """
    Plots the time-domain bias of the waveform reconstruction with confidence intervals.
    
    :param reconstructed: List of the reconstructed waveforms
    :param reference: The reference waveform (on-source)
    :param confidence_level: The confidence level for the intervals
    :param percentile_method: Method to compute percentiles
    :param normalize: Whether to normalize the bias
    :return: A tuple containing the figure and a dictionary of data to save
    """
    plot_kwargs.update(kwargs) 
    time = reference.sample_times
    to_save = {} 

    bias = np.asarray(reconstructed) - np.asarray(reference)

    if kwargs.get('normalize', False): 
        bias = bias / np.abs(np.asarray(reference))

    lower_bound, upper_bound = compute_confidence_intervals(bias, confidence_level, method=kwargs['percentile_method'], reference_waveform=reference)
    to_save.update({'time': time, 'lower_bound': lower_bound, 'upper_bound': upper_bound, 'CL': confidence_level}) 

    fig, ax = plt.subplots(figsize=plot_kwargs['figsize'])
    
    ax.fill_between(time, lower_bound, upper_bound, color=plot_kwargs['CL_color'], alpha=plot_kwargs['CL_alpha'], label=f'{confidence_level}% CI')

    mean_bias = np.nanmean(bias, axis=0)
    ax.plot(time, mean_bias, label='Mean Bias', color=plot_kwargs['mean_color'], linestyle=plot_kwargs['mean_linestyle'])
    to_save.update({'mean_bias': mean_bias})

    if kwargs.get('plot_median', True): 
        median_bias = np.nanmedian(bias, axis=0)
        ax.plot(time, median_bias, label='Median Bias', color=plot_kwargs['median_color'], linestyle=plot_kwargs['median_linestyle'])
        to_save.update({'median_bias': median_bias}) 

    ax.legend(fontsize=plot_kwargs['fontsize'])
    ax.grid(True)
    ax.tick_params(labelsize=plot_kwargs['fontsize'])
    ax.set_ylabel('Bias', fontsize=plot_kwargs['fontsize'])
    ax.set_xlabel('Time [s]', fontsize=plot_kwargs['fontsize'])


    plt.close()
    return fig, to_save


def plot_frequency_bias(reconstructed, reference, confidence_level = .95, **kwargs):
    """
    Plot the bias in the frequency domain
    :param reconstructed: List of the reconstructed waveforms
    :param reference: The reference waveform (on-source or injected)
    :param confidence_level: The confidence level for the intervals
    :param normalize: Whether to normalize the bias
    """ 
    plot_kwargs.update(kwargs) 
    # Work with magnitudes
    delta_t = getattr(reconstructed[0], '_delta_t', None)

    reconstructed = np.abs(np.asarray(reconstructed))
    reference = np.abs(np.asarray(reference))
    to_save = {} 

    #Compute frequency array and selected positive frequencies only 
    N = len(reconstructed[0].data)
    frequencies = np.fft.fftfreq(N, d=delta_t)
    mask = frequencies >= 0
    frequencies = frequencies[mask]

    #Compute the bias 
    bias = reconstructed - reference
    if kwargs.get('normalize', False): 
        bias = bias / np.abs(reference) 

    #Compute Confidence Intervals and store them 
    lower_bound, upper_bound = compute_confidence_intervals(bias, confidence_level, method=kwargs['percentile_method'], reference_waveform=reference)
    mean_bias = np.nanmean(bias, axis=0)
    to_save.update({'frequency': frequencies, 'lower_bound': lower_bound[mask], 'upper_bound': upper_bound[mask], 'CL': confidence_level, 'mean_bias': mean_bias[mask]})

    fig, ax = plt.subplots(figsize=plot_kwargs['figsize'])
    ax.fill_between(frequencies, lower_bound[mask], upper_bound[mask], color=plot_kwargs['CL_color'], alpha=plot_kwargs['CL_alpha'], label=f'{confidence_level}% CI')
    ax.plot(frequencies, mean_bias[mask], label='Mean Bias', color=plot_kwargs['mean_color'], linestyle=plot_kwargs['mean_linestyle'])
    
    if plot_kwargs.get('plot_median', True): 
        median_bias = np.nanmedian(bias, axis=0)[mask]
        ax.plot(frequencies, median_bias, label='Median Bias', color=plot_kwargs['median_color'], linestyle=plot_kwargs['median_linestyle'])
        to_save.update({'median_bias': median_bias})
    
    ax.legend(fontsize=plot_kwargs['fontsize'])
    ax.grid(True)
    ax.tick_params(labelsize=plot_kwargs['fontsize'])

    ax.set_ylabel('Bias', fontsize=plot_kwargs['fontsize'])
    ax.set_xlabel('Frequency [Hz]', fontsize=plot_kwargs['fontsize'])
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.close() 
    return fig, to_save




def plot_overlap(reconstructed, reference, **kwargs): 
    """
    Plot the overlap between the reconstructed waveforms and the injected waveform.
    """
    #Compute the overlap distribution of all waveforms 
    plot_kwargs.update(kwargs)
    to_save = {} 
    overlaps = compute_overlap(reconstructed, reference) 

    to_save.update({'overlaps': overlaps})

    #Plot the histogram of the overlaps
    fig, ax = plt.subplots(figsize=plot_kwargs['figsize'])
    bins = int(np.sqrt(len(overlaps)))
    ax.hist(overlaps, bins=bins, density=True, histtype='step', label='Reconstructed Overlap', color=plot_kwargs['inj_color']) 

    if kwargs.get('plot_mean', True):
        mean_reconstructed = np.nanmean(reconstructed, axis=0)
        mean_overlap = compute_overlap(mean_reconstructed, reference)
        ax.axvline(mean_overlap, color=plot_kwargs['mean_color'], linestyle=plot_kwargs['mean_linestyle'], label='Overlap of the mean') 
        to_save.update({'mean_overlap': mean_overlap})

    if kwargs.get('plot_median', True):
        median_reconstructed = np.nanmedian(reconstructed, axis=0) 
        median_overlap = compute_overlap(median_reconstructed, reference)
        ax.axvline(median_overlap, color=plot_kwargs['median_color'], linestyle=plot_kwargs['median_linestyle'], label='Overlap of the median')
        to_save.update({'median_overlap': median_overlap})

    #Set the labels and legend 
    ax.set_xlabel('Overlap', fontsize=plot_kwargs['fontsize'])
    ax.set_ylabel('Density', fontsize=plot_kwargs['fontsize'])
    ax.legend(fontsize=plot_kwargs['fontsize'])
    ax.grid(True)
    ax.tick_params(labelsize=plot_kwargs['fontsize'])
    plt.close() 
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
    ax.hist(overlaps, bins=bins, density=True, histtype='step', color=plot_kwargs['inj_color'])
    ax.axvline(mean_overlap, color=plot_kwargs['mean_color'], linestyle=plot_kwargs['mean_linestyle'], label='Mean Overlap') 
    ax.set_xlabel('Overlap', fontsize=plot_kwargs['fontsize'])
    ax.set_ylabel('Density', fontsize=plot_kwargs['fontsize'])
    ax.grid(True)
    ax.tick_params(labelsize=plot_kwargs['fontsize'])

    to_save = {'overlaps': overlaps}  
    plt.close() 
    return fig, to_save 


def plot_time_cumulative_hrss(reconstructed, reference, injected = None, confidence_level = 0.95, **kwargs):
    """Plot the cumulative distribution of HRSS in the time domain."""
    # Compute relevant statistics

    plot_kwargs.update(kwargs) 
    # Work with magnitudes
    reference_hrss = compute_cumulative_hrss(reference, reference._delta_t, axis=0)
    reconstructed_hrss = compute_cumulative_hrss(reconstructed, reference._delta_t, axis=1) / reference_hrss[-1]
    lower_bound, upper_bound = compute_confidence_intervals(reconstructed_hrss, confidence_level=confidence_level, 
                                                            method=kwargs.get('percentile_method'), reference_waveform=reference)
    mean_hrss = reconstructed_hrss.mean(axis=0)
    time = reference.sample_times.data 

    to_save = {'CL': confidence_level, 'reference_hrss': reference_hrss / reference_hrss[-1], 'lower_bound': lower_bound, 'upper_bound': upper_bound, 'mean_hrss': mean_hrss, 'time': time}

    #Plot sequence 
    fig, ax = plt.subplots(figsize=plot_kwargs['figsize'])


    ax.plot(time, reference_hrss / reference_hrss[-1], label='On-Source HRSS', color=plot_kwargs['inj_color'], linestyle=plot_kwargs['inj_ls'])
    ax.plot(time, mean_hrss, label='Mean Reconstructed HRSS', color=plot_kwargs['mean_color'], linestyle=plot_kwargs['mean_linestyle'])
    ax.fill_between(time, lower_bound, upper_bound, color=plot_kwargs['CL_color'], alpha=plot_kwargs['CL_alpha'], label=f'{confidence_level * 100}% CI')

    if injected is not None:
        injected_hrss = compute_cumulative_hrss(injected, injected._delta_t, axis=0)
        ax.plot(time, injected_hrss / reference_hrss[-1], label='Injected HRSS', color=plot_kwargs['inj_color'], linestyle=plot_kwargs['inj_ls'])
        to_save['injected_hrss'] = injected_hrss 


    if kwargs.get('plot_median', True):
        median_hrss = np.median(reconstructed_hrss, axis=0) 
        ax.plot(time, median_hrss, label='Median Reconstructed HRSS', color=plot_kwargs['median_color'], linestyle=plot_kwargs['median_linestyle'])
        to_save['median_hrss'] = median_hrss 

    # Visualization
    ax.grid(True)
    ax.legend(fontsize=plot_kwargs['fontsize'])
    ax.tick_params(labelsize=plot_kwargs['fontsize'])
    ax.set_ylabel('Cumulative HRSS (normalized)', fontsize=plot_kwargs['fontsize'])
    ax.set_xlabel('GPS Time [s]', fontsize=plot_kwargs['fontsize'])

    plt.close()
    return fig, to_save


def plot_frequency_cumulative_hrss(reconstructed, reference, injected = None, confidence_level = 0.95, plot_median=True, **kwargs):
    """Plot the cumulative distribution of HRSS in the frequency domain (positive frequencies only)."""
        
    plot_kwargs.update(kwargs) 


    reference_hrss = compute_cumulative_hrss(reference, reference._delta_t, axis=0)
    reconstructed_hrss = compute_cumulative_hrss(reconstructed, reference._delta_t, axis=1) / reference_hrss[-1]
    lower_bound, upper_bound = compute_confidence_intervals(reconstructed_hrss, confidence_level=confidence_level, method=plot_kwargs.get('percentile_method'),
                                                            reference_waveform=reference) 
    mean_hrss = reconstructed_hrss.mean(axis=0)

    #Frequencies to be used in the plot 
    delta_t = reconstructed[0]._delta_t 
    N = len(reference.data)
    frequency = np.fft.fftfreq(N, d=delta_t)
    mask = frequency >= 0
    frequency = frequency[mask] 

    to_save = {'CL': confidence_level, 'reference_hrss': reference_hrss[mask], 'lower_bound': lower_bound[mask], 'upper_bound': upper_bound[mask], 'mean_hrss': mean_hrss[mask],\
                'frequency': frequency}


    fig, ax = plt.subplots(figsize=plot_kwargs['figsize'])
    ax.plot(frequency, reference_hrss[mask] / reference_hrss[-1], label='Injected HRSS', color=plot_kwargs['ref_color'], linestyle=plot_kwargs['ref_ls'])
    ax.plot(frequency, mean_hrss[mask], label='Mean Reconstructed HRSS', color=plot_kwargs['mean_color'], linestyle=plot_kwargs['mean_linestyle'])
    ax.fill_between(frequency, lower_bound[mask], upper_bound[mask], color=plot_kwargs['CL_color'], alpha=plot_kwargs['CL_alpha'], label=f'{confidence_level}% CI')

    #Plot injected if passed 
    if injected is not None:
        injected_hrss = compute_cumulative_hrss(injected, injected._delta_t, axis=0)
        ax.plot(frequency, injected_hrss[mask] / reference_hrss[-1], label='Injected HRSS', color=plot_kwargs['inj_color'], linestyle=plot_kwargs['inj_ls'])
        to_save['injected_hrss'] = injected_hrss[mask]

    #Plot median if requested
    if plot_kwargs.get('plot_median', True):
        median_hrss = np.median(reconstructed_hrss, axis=0)[mask]
        ax.plot(frequency, median_hrss, label='Median Reconstructed HRSS', color=plot_kwargs['median_color'], linestyle=plot_kwargs['median_linestyle'])
        to_save['median_hrss'] = median_hrss

    # Visualization
    ax.grid(True)
    ax.legend(fontsize=plot_kwargs['fontsize'])
    ax.tick_params(labelsize=plot_kwargs['fontsize'])
    ax.set_ylabel('Cumulative HRSS (normalized)', fontsize=plot_kwargs['fontsize'])
    ax.set_xlabel('Frequency [Hz]', fontsize=plot_kwargs['fontsize'])
    plt.xscale('log'); plt.yscale('log')


    plt.close()
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
    ax.errorbar(t, leakage_mean, yerr=leakage_std, fmt='o', color=plot_kwargs['inj_color'], label='Leakage') 

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
    plt.close()
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
    ax.hist(reconstructed_hrss, bins=30, density=True, histtype='step', label='Reconstructed HRSS', color=plot_kwargs['inj_color']) 
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
    plt.close() 
    return fig, to_save


