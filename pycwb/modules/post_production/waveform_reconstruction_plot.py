import matplotlib.pyplot as plt 
import numpy as np 
from Postprocess_Functions import * 
#from plotting import plot_styles, default_plot_config


#Put in the same module as PostProcess Functions  

plot_kwargs = {'CL': 0.95,
                'method': 'shortest',
                'CL_color': 'lightgray',
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

def plot_time_waveform_reconstruction(reconstructed, injected):
    """Plot the time domain waveform reconstruction with confidence intervals.
    """
    #initialise image 
    fig, ax = plt.subplots(figsize=plot_kwargs['figsize'])  

    #Compute relevant statistics 
    lower_bound, upper_bound = compute_confidence_intervals(reconstructed, plot_kwargs['CL'], method=plot_kwargs["method"]) 
    mean = np.mean(reconstructed, axis=0)
    median = np.median(reconstructed, axis=0) 

    #plot the confidence intervals
    ax.fill_between(injected.sample_times, lower_bound, upper_bound, color=plot_kwargs['CL_color'], alpha=plot_kwargs['CL_alpha'], label=f'{plot_kwargs["CL"]}% CI') 
    #plot the median and the mean 
    ax.plot(injected.sample_times, mean, label='Mean', color=plot_kwargs['mean_color'], linestyle=plot_kwargs['mean_linestyle'])
    ax.plot(injected.sample_times, median, label='Median', color=plot_kwargs['median_color'], linestyle=plot_kwargs['median_linestyle'])
    ax.plot(injected.sample_times, injected, label='Injected', color=plot_kwargs['injected_color'], linestyle=plot_kwargs['injected_linestyle'], alpha=plot_kwargs['injected_alpha']) 

    #Visualization 
    ax.grid(True)
    ax.set_xlabel('GPS Time [s]', fontsize=plot_kwargs['fontsize'])
    ax.set_ylabel('Strain', fontsize=plot_kwargs['fontsize'])
    ax.legend(fontsize=plot_kwargs['fontsize'])
    ax.tick_params(labelsize=plot_kwargs['fontsize']) 

    #Define the "to save" dictionary
    to_save = {'time': injected.sample_times,
        'injected': injected,
        'mean': mean,
        'median': median,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'CL': plot_kwargs['CL']
    }
    return fig, to_save


def plot_frequency_waveform_reconstruction(reconstructed, injected): 
    """Plot the frequency domain waveform reconstruction with confidence intervals.
    """
    #initialise image 
    reconstructed_data = np.abs(reconstructed) 
    injected_data = np.abs(injected) 

    fig, ax = plt.subplots(figsize=plot_kwargs['figsize']) 

    #Compute relevant statistics 
    lower_bound, upper_bound = compute_confidence_intervals(reconstructed_data, plot_kwargs['CL'], method=plot_kwargs["method"]) 
    mean = np.mean(reconstructed_data, axis=0)
    median = np.median(reconstructed_data, axis=0) 

    #plot the confidence intervals
    ax.fill_between(injected.frequencies, lower_bound, upper_bound, color=plot_kwargs['CL_color'], alpha=plot_kwargs['CL_alpha'], label=f'{plot_kwargs["CL"]}% CI') 
    #plot the median and the mean 
    ax.plot(injected.frequencies, mean, label='Mean', color=plot_kwargs['mean_color'], linestyle=plot_kwargs['mean_linestyle'])
    ax.plot(injected.frequencies, median, label='Median', color=plot_kwargs['median_color'], linestyle=plot_kwargs['median_linestyle'])
    ax.plot(injected.frequencies, injected_data, label='Injected', color=plot_kwargs['injected_color'], linestyle=plot_kwargs['injected_linestyle']) 

    #Visualization 
    ax.grid(True)
    ax.set_xlabel('Frequency [Hz]', fontsize=plot_kwargs['fontsize'])
    ax.set_ylabel('Strain', fontsize=plot_kwargs['fontsize'])
    ax.legend(fontsize=plot_kwargs['fontsize'])
    ax.tick_params(labelsize=plot_kwargs['fontsize']) 

    plt.xscale('log')
    plt.yscale('log')

    to_save = {'frequency': injected.frequencies,
        'injected': injected_data,
        'mean': mean,
        'median': median,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'CL': plot_kwargs['CL']
    }

    return fig, to_save


def plot_time_bias(reconstructed, injected, normalize = True): 
    """ 
    Plot the bias of the reconstructed waveforms compared to the injected waveform.
    """
    #Compute 
    bias = np.asarray(reconstructed) - np.asarray(injected)
    if normalize: 
        bias /= injected  
    lower_bound, upper_bound = compute_confidence_intervals(bias, plot_kwargs['CL'], method="shortest") 
    mean_bias = np.mean(bias, axis=0) 

    #Normalize wrt injected waveform if requested

    fig, ax = plt.subplots(figsize=plot_kwargs['figsize']) 
    ax.fill_between(injected.sample_times, lower_bound, upper_bound, color=plot_kwargs['CL_color'], alpha=plot_kwargs['CL_alpha'], label=f'{plot_kwargs["CL"]}% CI')
    ax.plot(injected.sample_times, mean_bias, label='Mean Bias', color=plot_kwargs['mean_color'], linestyle=plot_kwargs['mean_linestyle'])  

    ax.set_xlabel('GPS Time [s]', fontsize=plot_kwargs['fontsize'])
    ax.set_ylabel('Bias', fontsize=plot_kwargs['fontsize'])
    ax.legend(fontsize=plot_kwargs['fontsize'])
    ax.grid(True)
    ax.tick_params(labelsize=plot_kwargs['fontsize'])
    
    if normalize: 
        ax.set_ylim(-4,4) 

    to_save = {'time': injected.sample_times,
        'mean_bias': mean_bias,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'CL': plot_kwargs['CL']
    }
    return fig, to_save 


def plot_frequency_bias(reconstructed, injected, normalize = True):
    """ 
    Plot the bias of the reconstructed waveforms compared to the injected waveform in frequency domain.
    """
    #Compute 
    reconstructed_data = np.abs(reconstructed)
    injected_data= np.abs(injected)

    bias = reconstructed_data - injected_data
    mean_bias = np.mean(bias, axis=0) 
    lower_bound, upper_bound = compute_confidence_intervals(bias, plot_kwargs['CL'], method="shortest") 

    if normalize: 
        mean_bias /= injected_data
        lower_bound /= injected_data
        upper_bound /= injected_data
    fig, ax = plt.subplots(figsize=plot_kwargs['figsize']) 
    ax.fill_between(injected.frequencies, lower_bound, upper_bound, color=plot_kwargs['CL_color'], alpha=plot_kwargs['CL_alpha'], label=f'{plot_kwargs["CL"]}% CI')
    ax.plot(injected.frequencies, mean_bias, label='Mean Bias', color=plot_kwargs['mean_color'], linestyle=plot_kwargs['mean_linestyle'])  

    ax.set_xlabel('Frequency [Hz]', fontsize=plot_kwargs['fontsize'])
    ax.set_ylabel('Bias', fontsize=plot_kwargs['fontsize'])
    ax.legend(fontsize=plot_kwargs['fontsize'])
    ax.grid(True)
    ax.tick_params(labelsize=plot_kwargs['fontsize'])

    ax.set_xscale('log')

    to_save = {'frequency': injected.frequencies,
        'mean_bias': mean_bias,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'CL': plot_kwargs['CL']
    }
    return fig, to_save 


def plot_overlap(reconstructed, injected, plot_mean = True, plot_median = True): 
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
    if plot_mean:
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
    to_save['mean_overlap'] = mean_overlap if plot_mean else None
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


def plot_cumulative_hrss(reconstructed, injected, plot_mean=True, plot_median=True):
    """Plot the cumulative distribution of the reconstructed HRSS compared to the injected HRSS.
    """
    mean_reconstructed = np.mean(reconstructed, axis=0)
    median_reconstructed = np.median(reconstructed, axis=0)

    fig, ax = plt.subplots(figsize=plot_kwargs['figsize'])
    injected_hrss = compute_cumulative_hrss(injected, injected._delta_t, axis=0) 
    reconstructed_hrss = compute_cumulative_hrss(reconstructed, injected._delta_t, axis=1) / injected_hrss[-1]
    mean_hrss = reconstructed_hrss.mean(axis=0) 
    median_hrss = np.median(reconstructed_hrss, axis=0) 
    #mean_hrss = compute_cumulative_hrss(mean_reconstructed, injected._delta_t, axis=0) / injected_hrss[-1]
    #median_hrss = compute_cumulative_hrss(median_reconstructed, injected._delta_t, axis=0) / injected_hrss[-1]

    lower_bound, upper_bound = compute_confidence_intervals(reconstructed_hrss, plot_kwargs['CL'], method="shortest") 

    fig, ax = plt.subplots(figsize=plot_kwargs['figsize'])
    ax.plot(injected.sample_times, injected_hrss / injected_hrss[-1], label='Injected HRSS', color=plot_kwargs['injected_color'], linestyle=plot_kwargs['injected_linestyle'])
    ax.plot(injected.sample_times, mean_hrss, label='Mean Reconstructed HRSS', color=plot_kwargs['mean_color'], linestyle=plot_kwargs['mean_linestyle'])
    ax.plot(injected.sample_times, median_hrss, label='Median Reconstructed HRSS', color=plot_kwargs['median_color'], linestyle=plot_kwargs['median_linestyle'])
    ax.fill_between(injected.sample_times, lower_bound, upper_bound, color=plot_kwargs['CL_color'], alpha=plot_kwargs['CL_alpha'], label=f'{plot_kwargs["CL"]}% CI')

    ax.grid(True)
    ax.set_xlabel('GPS Time [s]', fontsize=plot_kwargs['fontsize'])
    ax.set_ylabel('Cumulative HRSS (normalized)', fontsize=plot_kwargs['fontsize'])
    ax.legend(fontsize=plot_kwargs['fontsize'])
    ax.tick_params(labelsize=plot_kwargs['fontsize'])

    to_save = {'time': injected.sample_times,
      'injected_hrss': injected_hrss / injected_hrss[-1],
      'mean_hrss': mean_hrss,
      'median_hrss': median_hrss,
      'lower_bound': lower_bound,
      'upper_bound': upper_bound,
      'CL': plot_kwargs['CL']
    }

    return fig, to_save 

def plot_frequency_cumulative_hrss(reconstructed, injected, plot_mean=True, plot_median=True):
    """Plot the cumulative distribution of the reconstructed HRSS compared to the injected HRSS in frequency domain.
    """
    reconstructed_data = np.abs(reconstructed)
    injected_data = np.abs(injected)

    df = injected.frequencies[1] - injected.frequencies[0] 
    mean_reconstructed = np.mean(reconstructed_data, axis=0)
    median_reconstructed = np.median(reconstructed_data, axis=0)

    fig, ax = plt.subplots(figsize=plot_kwargs['figsize'])
    reconstructed_hrss = compute_cumulative_hrss(reconstructed_data, df, axis=1)
    injected_hrss = compute_cumulative_hrss(injected_data, df, axis=0) 
    mean_hrss = reconstructed_hrss.mean(axis=0) 
    median_hrss = np.median(reconstructed_hrss, axis=0) 
    #mean_hrss = compute_cumulative_hrss(mean_reconstructed, df, axis=0) / injected_hrss[-1]
    #median_hrss = compute_cumulative_hrss(median_reconstructed, df, axis=0) / injected_hrss[-1]

    lower_bound, upper_bound = compute_confidence_intervals(reconstructed_hrss / injected_hrss[-1], plot_kwargs['CL'], method="shortest")
    injected_hrss /= injected_hrss[-1]     # Normalize injected HRSS to 1 After normalizing the others 

    fig, ax = plt.subplots(figsize=plot_kwargs['figsize'])
    ax.semilogx(injected.frequencies, injected_hrss, label='Injected HRSS', color=plot_kwargs['injected_color'], linestyle=plot_kwargs['injected_linestyle'])
    ax.semilogx(injected.frequencies, mean_hrss, label='Mean Reconstructed HRSS', color=plot_kwargs['mean_color'], linestyle=plot_kwargs['mean_linestyle'])
    ax.semilogx(injected.frequencies, median_hrss, label='Median Reconstructed HRSS', color=plot_kwargs['median_color'], linestyle=plot_kwargs['median_linestyle'])
    ax.fill_between(injected.frequencies, lower_bound, upper_bound, color=plot_kwargs['CL_color'], alpha=plot_kwargs['CL_alpha'], label=f'{plot_kwargs["CL"]}% CI')
    ax.grid(True) 
    ax.set_xlabel('Frequency [Hz]', fontsize=plot_kwargs['fontsize'])
    ax.set_ylabel('Cumulative HRSS (normalized)', fontsize=plot_kwargs['fontsize'])
    ax.legend(fontsize=plot_kwargs['fontsize'])
    ax.tick_params(labelsize=plot_kwargs['fontsize'])

    to_save = {'frequency': injected.frequencies,
      'injected_hrss': injected_hrss,
      'mean_hrss': mean_hrss,
      'median_hrss': median_hrss,
      'lower_bound': lower_bound,
      'upper_bound': upper_bound,
      'CL': plot_kwargs['CL']
    }
    
    return fig, to_save 


def plot_hrss_scatter(reconstructed, injected): 
    """Plot the scatter of the reconstructed HRSS compared to the injected HRSS.
    """
    injected_hrss, reconstructed_hrss = [], []
    #Compute the hrss of all waveforms 
    for i, reconstructed_waveform in enumerate(reconstructed): 
        reconstructed_hrss.append(compute_hrss(reconstructed_waveform.data, injected[i]._delta_t))
        injected_hrss.append(compute_hrss(injected[i].data, injected[i]._delta_t))

    max_val = max(max(injected_hrss), max(reconstructed_hrss)) * 1.01
    #Plot the scatter plot of the hrss
    fig, ax = plt.subplots(figsize=plot_kwargs['figsize'])
    ax.scatter(injected_hrss, reconstructed_hrss, alpha=0.5, color=plot_kwargs['injected_color'], label='Reconstructed HRSS') 
    ax.plot([0, max_val], [0, max_val], color='gray', linestyle='--', label='y=x') 

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

    tstart, tend = injected.tstart, injected.tend 
    
    ax.plot(injected.sample_times, injected, label='Injected', color=plot_kwargs['injected_color'], linestyle=plot_kwargs['injected_linestyle'], alpha=plot_kwargs['injected_alpha'])
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