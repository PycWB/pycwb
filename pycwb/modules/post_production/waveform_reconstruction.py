from concurrent.futures import ProcessPoolExecutor
from pycwb.types.waveform import Waveform, load_waveform  
from pycwb.types.time_series import TimeSeries
from tqdm import tqdm  # type: ignore
import numpy as np  # type: ignore
import logging 
import os


logger = logging.getLogger(__name__) 

def save(figure, results_dictionary, directory, filename, extension = 'pdf'): 
    """Save the figure and results dictionary to the specified directory.
    """
    plots_dir = os.path.join(directory, 'reports', 'waveform_reconstruction_plots')
    results_dir = os.path.join(directory, 'reports', 'waveform_reconstruction_results')

    if not os.path.exists(plots_dir) or not os.path.exists(results_dir, exist_ok=True): 
        create_save_directories(directory)

    #Save figure
    figure_path = os.path.join(directory,  f'{filename}.{extension}', bbox_inches='tight') 
    figure.savefig(figure_path)

    #Save results dictionary as a npz file
    results_path = os.path.join(directory, f"{filename}.npz")
    np.savez(results_path, **results_dictionary)



def create_save_directories(analysis_directory): 
    """Create a directory for saving plots and results.

    Parameters
    ----------
    analysis_directory : str
        The base directory for analysis outputs. 
    """
    if not os.path.exists(analysis_directory):
        raise FileNotFoundError(f"The provided analysis directory {analysis_directory} does not exist.")
    
    plot_directory = os.path.join(analysis_directory, 'reports', 'waveform_reconstruction_plots')
    results_directory = os.path.join(analysis_directory, 'reports', 'waveform_reconstruction_results')     

    os.makedirs(plot_directory, exist_ok=True)
    os.makedirs(results_directory, exist_ok=True)



def load_and_slice(args):

    file, reference, time_shift, phase_shift, early_start, scale = args 

    if time_shift is None and phase_shift is None and reference is None: 
        raise ValueError(f"Either one among reference, time_shift or phase_shift is needed, while ({reference}, {time_shift}, {phase_shift})")
    
    waveform = load_waveform(file, skip_nans = True) 
    if waveform is None: 
        return None 
    reference = load_waveform(reference)#, resample = waveform.sample_rate)  

    #rescale the data. If scale < 0, the waveform is downscaled, if > 0, the waveform is upscaled. The scale factor is applied as sqrt(2) ** scale
    if scale != 0:
        waveform.data *= np.sqrt(2) ** scale  
          
    try: 
        waveform, reference = sync_waveforms(waveform, reference, time_shift, phase_shift) 
        #reference = waveform.syncWaveform(reference, sync_phase = True)
        if len(waveform) != len(reference):
            waveform, reference = pad_waveforms(waveform, reference)
        waveform, reference = slice_waveforms(waveform, reference, early_start) 
    
    except Exception as e: 
        return None 

    return waveform, reference 

def sync_waveforms(waveform, reference, time_shift = None, phase_shift = None):
    """
    Synchronize a single waveform to the reference waveform.
    :returns: synchronized waveform and reference waveform
    """
    w_copy, r_copy = waveform.copy(), reference.copy() 

    if time_shift is None and phase_shift is None: 
        w_copy, r_copy = w_copy.syncWaveform(r_copy, sync_phase=True)

    if time_shift: 
        w_copy.timeShift(time_shift) 

    if phase_shift: 
        w_copy.phaseShift(phase_shift)

    return w_copy, r_copy



def pad_waveforms(waveform, reference):
    """
    Pad two waveforms with zeros so they share the same start and end times.
    Returns both padded waveforms.
    """
    w, r = waveform.copy(), reference.copy()   
    dt = w.delta_t

    # Determine combined start and end times
    t_start = min(w.sample_times[0], r.sample_times[0])
    t_end   = max(w.sample_times[-1], r.sample_times[-1])

    # Pad or truncate w_pad
    n_pre_w  = int(round((w.sample_times[0] - t_start) / dt))
    n_post_w = int(round((t_end - w.sample_times[-1]) / dt))

    if n_pre_w > 0:
        w.prepend_zeros(n_pre_w)
    if n_post_w > 0:
        w.append_zeros(n_post_w)

    # Pad or truncate ref_pad
    n_pre_r  = int(round((r.sample_times[0] - t_start) / dt))
    n_post_r = int(round((t_end - r.sample_times[-1]) / dt))
    if n_pre_r > 0:
        r.prepend_zeros(n_pre_r)
    if n_post_r > 0:
        r.append_zeros(n_post_r) 

    
    w._findStartEnd() 
    r._findStartEnd()

    return w, r


def slice_waveforms(waveform, reference, early_start = 0): 
    """
    Slice a single waveform to the time range of the reference waveform.
    :returns: sliced waveform and reference waveform
    """
    
    w, r = waveform.copy(), reference.copy() 
    
    #Define start and stop to slice waveforms 
    start, stop =  r.istart, int(r.iend) 
    start = int(max(start - early_start * w.sample_rate, 0)) 
    
    w = Waveform(TimeSeries(w[start:stop], dt = 1 / w.sample_rate, t0 = r.tstart), folder=w.folder)
    r = Waveform(TimeSeries(r[start:stop], dt = 1 / r.sample_rate, t0 = r.tstart), folder=r.folder)

    w._total_time_shift = waveform._total_time_shift
    w._total_phase_shift = waveform._total_phase_shift

    return w, r 


def compute_confidence_intervals(waveforms, confidence_level=0.95, method = "percentiles", reference_waveform = None):
    """
    Compute the confidence intervals for a list of waveforms.
    """
    #Assuming waveforms are sliced to the same time range
    if method == "percentiles": 
        lower_bound, upper_bound = np.nanpercentile(waveforms, [(1-confidence_level)*100/2, (1+confidence_level)*100/2], axis = 0)
    
    #Return lower datas and upper bound based on LC
    elif method == "upper": 
        lower_bound, upper_bound = np.nanpercentile(waveforms, [0, confidence_level * 100], axis = 0)

    #Return lower datas and upper bound based on CL
    elif method == "lower": 
        lower_bound, upper_bound = np.nanpercentile(waveforms, [(1-confidence_level) * 100, 100], axis = 0)
    
    elif method == "BCa": 
        if reference_waveform is None: 
            raise ValueError("Reference waveform must be provided for BCa confidence intervals.")
        lower_bound, upper_bound = BCa_confidence_intervals(waveforms, reference_waveform, confidence_level=confidence_level)
    
    elif method == "studentized_bootstrap": 
        if reference_waveform is None: 
            raise ValueError("Reference waveform must be provided for studentized bootstrap confidence intervals.")
        lower_bound, upper_bound = studentized_bootstrap_confidence_intervals(waveforms, reference_waveform, confidence_level)
    
    else: 
        raise ValueError("Unsupported ordering method.") 
    
    return lower_bound, upper_bound 


def studentized_bootstrap_confidence_intervals(waveforms,reference_waveform,confidence_level): 
    """
    Compute studentized bootstrap confidence intervals for a list of waveforms.
    """
    waveforms = np.asarray(waveforms)
    n_bootstrap, n_points = waveforms.shape

    lower_bound = np.zeros(n_points)
    upper_bound = np.zeros(n_points)
    alpha = (1 - confidence_level) / 2

    se = np.nanstd(waveforms, axis=0, ddof=1) #define standard error (se)
    studentized_residuals = (waveforms - reference_waveform) / se 
    print(se) 
    print(studentized_residuals) 
    t_up, t_low = np.nanpercentile(studentized_residuals, [100 * alpha, 100 * (1 - alpha)], axis=0) 
    
    print(t_up, t_low)
    lower_bound = reference_waveform - t_low * se
    upper_bound = reference_waveform - t_up * se 

    return lower_bound, upper_bound

def BCa_confidence_intervals(waveforms, reference_waveform, confidence_level=0.95):
    """
    Compute the bias-corrected and accelerated (BCa) confidence intervals for a list of waveforms.
    """
    from scipy.stats import norm  # type: ignore

    alpha = (1 - confidence_level) / 2
    lower_percentile = alpha * 100
    upper_percentile = (1 - alpha) * 100

    waveforms = np.array(waveforms)
    n_waveforms, n_points = waveforms.shape

    lower_bound = np.zeros(n_points)
    upper_bound = np.zeros(n_points)

    for i in range(n_points):
        point_values = waveforms[:, i]
        point_values = point_values[~np.isnan(point_values)]

        if len(point_values) == 0:
            lower_bound[i] = np.nan
            upper_bound[i] = np.nan
            continue

        point_mean = reference_waveform[i] 

        z0 = norm.ppf(np.sum(point_values < point_mean) / len(point_values))

        jackknife_means = []
        for j in range(len(point_values)):
            jackknife_sample = np.delete(point_values, j)
            jackknife_means.append(np.mean(jackknife_sample))
        jackknife_means = np.array(jackknife_means)

        mean_jackknife = np.mean(jackknife_means)
        acc_numerator = np.sum((mean_jackknife - jackknife_means) ** 3)
        acc_denominator = 6 * (np.sum((mean_jackknife - jackknife_means) ** 2)) ** 1.5
        a = acc_numerator / acc_denominator if acc_denominator != 0 else 0

        z_lower = norm.ppf(lower_percentile / 100)
        z_upper = norm.ppf(upper_percentile / 100)

        adj_lower = norm.cdf(2 * z0 + z_lower)#/ (1 - a * z_lower)) * 100
        adj_upper = norm.cdf(2 * z0 + z_upper)#/ (1 - a * z_upper)) * 100 

        lower_bound[i] = np.percentile(point_values, adj_lower * 100)
        upper_bound[i] = np.percentile(point_values, adj_upper * 100)
    return lower_bound, upper_bound
    

def compute_overlap(reconstructed, reference_waveform):  
    """
    Compute the overlap between the reconstructed waveforms and the injected waveform."""
    
    #Compute overlap of N waveforms against a single, reference  waveform 

    if type(reference_waveform) == Waveform: 
        reconstructed = np.atleast_2d(reconstructed) 
        norm1 = np.linalg.norm(reconstructed, axis = 1)
        norm2 = np.linalg.norm(reference_waveform)
        overlaps = np.dot(reconstructed, reference_waveform) / (norm1 * norm2)

    #Compute eoverlap of N waveforms against their respective injected waveform 
    elif type(reference_waveform) == list and len(reference_waveform) == len(reconstructed): 
        overlaps = [] 
        for reconstructed_waveform, ref_waveform in zip(reconstructed, reference_waveform): 
            overlap = compute_overlap(reconstructed_waveform, ref_waveform)
            overlaps.append(overlap)

    else: 
        raise ValueError("Reference waveform list length must be 1 or equal to the number of reconstructed waveforms.") 
    
    overlaps = np.array(overlaps)

    if overlaps.size == 1:
        return overlaps.item() 

    return overlaps 

def compute_cumulative_hrss(waveform, delta_t, axis = 1): 
    """
    Compute the cumulative hrss of a waveform.
    """
    hrss = np.sqrt(np.cumsum(np.abs(waveform) ** 2, axis=axis) * delta_t)
    return hrss


def compute_leakage(reconstructed, reference_waveform, time): 
    """
    Compute the time leakage as a function of time after the injected signal's end time 
    """
    #Compute the hrss of the injected waveform and initialize leaked hrss array
    injected_hrss = np.sqrt(np.sum(np.square(reference_waveform.data)))
    leaked_hrss = np.zeros(shape=(len(reconstructed), len(time)))

    #Compute end time of reference waveform and time step 
    end_time_idx = np.max(np.where(reference_waveform > reference_wavefrom.max() * 1e-3))
    end_time = reference_waveform.sample_times.data[end_time_idx + 1]
    dt = time[1] - time[0]

    #Compute leakage over time for each recontructed waveform 
    for i, waveform in enumerate(reconstructed): 
        for j in range(20): 
            try:
                leaked_hrss[i,j] = np.sqrt(np.nansum(waveform.time_slice(end_time + j*dt, end_time + (j+1)*dt).data ** 2)) / injected_hrss 
            except (IndexError, ValueError): 
                pass 
        
    #Return mean and std of leakage over all reconstructed waveforms
    mean_leakage = np.nanmean(leaked_hrss, axis=0)
    std_leakage = np.nanstd(leaked_hrss, axis=0) / np.sqrt(len(reconstructed))
    return mean_leakage, std_leakage 


def compute_hrss(waveform, delta_t):
    """
    Compute the hrss for a list of waveforms.
    """
    return np.sqrt(np.sum(np.abs(waveform) ** 2) * delta_t) 





def load_one_waveform_OLD(args):
    """
    Load a single waveform from the specified folder and subfolder.
    :returns: Waveform object or None if loading fails 
    """
    folder, ifo, type_, whitened, format, resample, skip_trigger = args

    try:     
        file_name = f"{ifo}_wf_{type_}.{format}" if not whitened else f"{ifo}_wf_{type_}_whiten.{format}"
        waveform = load_waveform(os.path.join(folder, file_name), resample = resample)
    
        if np.any(np.isnan(waveform.data)): 
            return None 

        if skip_trigger and f"{ifo}_wf_INJ.{format}" not in folder: 
            return None

        else: 
            return waveform

    except Exception as e:
        return None


def sync_waveforms_OLD(waveforms, reference, sync_phase=True, time_shift = None, phase_shift = None, max_workers=None):
    """
    Synchronize all waveforms to the reference waveform(s). If len(reference) == 1, synchronize all waveforms to the same reference, 
    otherwise synchronize each waveform to its respective reference. 
    :param waveforms: list of Waveform to synchronize
    :param reference: single Waveform or list of Waveform (same length as waveforms)
    :param sync_phase: bool, whether to synchronize phase
    :return: sync_waveforms_, reference_waveforms_ 
    """
    sync_waveforms_ = []
    reference_waveforms_ = []
    discarded_waveforms = 0


    if isinstance(reference, Waveform):
        args = [(w, reference, sync_phase, time_shift, phase_shift) for w in waveforms]

    elif isinstance(reference, list) and len(reference) == len(waveforms):
        args = [(w, r, sync_phase, time_shift, phase_shift) for w, r in zip(waveforms, reference)]

    else:
        raise ValueError(
            "Reference must be a single Waveform or a list with the same length as waveforms."
        )

    with ProcessPoolExecutor(max_workers=max_workers or os.cpu_count()) as exe:
        results = list(tqdm( exe.map(_sync_one, args), total=len(pairs),desc="Synchronizing waveforms (parallel)"))

    for r in results:
        if r is None:
            discarded_waveforms += 1
            continue

        w_sync, ref = r
        sync_waveforms_.append(w_sync)
        reference_waveforms_.append(ref)

    if discarded_waveforms > 0:
        logger.warning(f"{discarded_waveforms} over {len(waveforms)} waveforms were discarded during synchronization.")

    return sync_waveforms_, reference_waveforms_