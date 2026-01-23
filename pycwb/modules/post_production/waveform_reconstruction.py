#Postprocessing for waveform reconstruction 

from concurrent.futures import ProcessPoolExecutor
from pycwb.types.waveform import Waveform, load_waveform 
from tqdm import tqdm  # type: ignore
import numpy as np  # type: ignore
import logging 
import os

#TODO: Implement whitened options 

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


def load_waveforms(folder, ifo, resample = None, load_injected=False, skip_trigger = False, whitened=False,  format="hdf", max_workers=None):
    """
    Load waveforms from the specified folder 
    Parameters: 
    :param folder: str, path to the folder containing waveform subfolders
    :param ifo: str, interferometer name (e.g., 'H1', 'L1', 'V1')
    :param resample: float or None, if specified, resample waveforms to this sampling rate
    :param load_injected: bool, if True, load injected waveforms instead of reconstructed ones
    :param skip_trigger: bool, if True, skip waveforms without injected trigger
    :param whitened: bool, if True, load whitened waveforms
    :param format: str, file format of the waveforms ('hdf', 'txt' or 'npy')
    :param max_workers: int or None, maximum number of parallel workers 
    :return: list of loaded Waveform objects
    """
    trigger_folders = os.listdir(folder)
    loaded_waveforms = []
    discarded = 0
    
    args = [(folder, f, ifo, load_injected, whitened, format, resample, skip_trigger) for f in trigger_folders]

    with ProcessPoolExecutor(max_workers=max_workers or os.cpu_count()) as exe:
        results = list(tqdm(exe.map(_load_one_waveform, args), total=len(args), desc="Loading waveforms"))

    loaded_waveforms = [r for r in results if r is not None]
    discarded = len(trigger_folders) - len(loaded_waveforms) 
    if discarded > 0:
        logger.warning(f"{discarded} over {len(trigger_folders)} waveforms were discarded during loading.")

    return loaded_waveforms 


def _load_one_waveform(args):
    """
    Load a single waveform from the specified folder and subfolder.
    :returns: Waveform object or None if loading fails 
    """
    folder, subfolder, ifo, load_injected, whitened, format, resample, skip_trigger = args

    try:
        if not load_injected: 
            file_name = "reconstructed_waveform_{ifo}_whitened.{format}" if whitened else f"reconstructed_waveform_{ifo}.{format}"
        else: 
            file_name = f"injected_strain_{ifo}.{format}"
      
        waveform = load_waveform(os.path.join(folder, subfolder, file_name), resample = resample)

        if np.any(np.isna(waveform.data)): 
            return None 

        if skip_trigger and f"injected_strain_{ifo}.{format}" not in os.listdir(os.path.join(folder, subfolder)): 
            return None

        else: 
            return waveform

    except Exception as e:
        return None


def sync_waveforms(waveforms, reference, sync_phase=True, max_workers=None):
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
        pairs = [(w, reference, sync_phase) for w in waveforms]

    elif isinstance(reference, list) and len(reference) == len(waveforms):
        pairs = [(w, r, sync_phase) for w, r in zip(waveforms, reference)]

    else:
        raise ValueError(
            "Reference must be a single Waveform or a list with the same length as waveforms."
        )

    with ProcessPoolExecutor(max_workers=max_workers or os.cpu_count()) as exe:
        results = list(tqdm( exe.map(_sync_one, pairs), total=len(pairs),desc="Synchronizing waveforms (parallel)"))

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

def _sync_one(args):
    """
    Synchronize a single waveform to the reference waveform.
    :returns: synchronized waveform and reference waveform
    """
    waveform, reference, sync_phase = args
    waveform.syncWaveform(reference, sync_phase=sync_phase)
    return waveform, reference


def pad_waveforms(waveforms, reference, max_workers=None):
    """
    Pad all waveforms to have the same time support as reference waveform(s),
    ensuring every output pair has exactly the same length.
    
    :param waveforms: list of TimeSeries to pad
    :param reference: single TimeSeries or list of TimeSeries (same length)
    :return: padded_waveforms, padded_references
    """
    if isinstance(reference, Waveform):
        pairs = [(w, reference) for w in waveforms]

    elif isinstance(reference, list) and len(reference) == len(waveforms):
        pairs = list(zip(waveforms, reference))
    else:
        raise ValueError("Reference must be a single TimeSeries or a list of same length as waveforms.")

    padded_waveforms = []
    padded_refs = []
    discarded = 0 
    with ProcessPoolExecutor(max_workers=max_workers or os.cpu_count()) as exe:
        results = list(tqdm(exe.map(_pad_pair_to_same_length, pairs), total=len(pairs), desc="Padding waveforms (parallel)")) 

    # Collect results
    for r in results:
        if r is not None:
            padded_waveforms.append(r[0])
            padded_refs.append(r[1])
        else:
            discarded += 1

    if discarded > 0:
        print(f"{discarded} waveforms could not be padded and were discarded.")

    return padded_waveforms, padded_refs

def _pad_pair_to_same_length(args):
    """
    Pad two waveforms with zeros so they share the same start and end times.
    Returns both padded waveforms.
    """
    try: 
        w, ref_w = args
        w_pad = w.copy()
        ref_pad = ref_w.copy()
        dt = w.delta_t

        # Determine combined start and end times
        t_start = min(w_pad.sample_times[0], ref_pad.sample_times[0])
        t_end   = max(w_pad.sample_times[-1],   ref_pad.sample_times[-1])

        # Pad or truncate w_pad
        n_pre_w  = int(round((w_pad.sample_times[0] - t_start) / dt))
        n_post_w = int(round((t_end - w_pad.sample_times[-1]) / dt))
        if n_pre_w > 0:
            w_pad.prepend_zeros(n_pre_w)
        if n_post_w > 0:
            w_pad.append_zeros(n_post_w)

        # Pad or truncate ref_pad
        n_pre_r  = int(round((ref_pad.sample_times[0] - t_start) / dt))
        n_post_r = int(round((t_end - ref_pad.sample_times[-1]) / dt))
        if n_pre_r > 0:
            ref_pad.prepend_zeros(n_pre_r)
        if n_post_r > 0:
            ref_pad.append_zeros(n_post_r)

        #Now both have exactly the same length
        assert len(w_pad) == len(ref_pad)

    except Exception:
        return None 
    
    return w_pad, ref_pad




def slice_waveforms(waveforms, reference, max_workers=None):
    """
    Slice all waveforms to the same time range as the reference waveforms. If len(reference_waveform) == 1, slice all waveforms to the same time range, 
    otherwise slice each waveform to its respective reference.
    """

    sliced_list = []
    reference_waveforms_ = []
    discarded_waveforms = 0

    # ------------------------------------------------------------
    # Build (waveform, reference) pairs
    # ------------------------------------------------------------
    if isinstance(reference, Waveform):
        pairs = [(w, reference) for w in waveforms]

    elif isinstance(reference, list) and len(reference) == len(waveforms):
        pairs = list(zip(waveforms, reference))

    else:
        raise ValueError("Reference must be a single Waveform or a list with the same length as waveforms."
        )
    #Parallel execution
    with ProcessPoolExecutor(max_workers=max_workers or os.cpu_count()) as exe:
        results = list(tqdm(exe.map(_slice_one, pairs), total=len(pairs), desc="Slicing waveforms (parallel)"))

    #collect results 
    for r in results:
        if r is None:
            discarded_waveforms += 1
            continue

        slice_w, slice_r = r
        sliced_list.append(slice_w)
        reference_waveforms_.append(slice_r)

    if discarded_waveforms > 0:
        logger.warning(f"{discarded_waveforms} over {len(waveforms)} waveforms were discarded.") 

    return sliced_list, reference_waveforms_



def _slice_one(args): 
    """
    Slice a single waveform to the time range of the reference waveform.
    :returns: sliced waveform and reference waveform
    """
    w, ref_w = args
    start, stop = w.istart, w.iend 

    w_slice = Waveform(w[start:stop]) 
    ref_slice = Waveform(ref_w[start:stop]) 

    if len(w_slice) != len(ref_slice):
        return None
    
    return w_slice, ref_slice 


 


def _slice_one_OLD(args):
    """
    Slice a single waveform to the time range of the reference waveform.
    :returns: sliced waveform and reference waveform
    """
    w, ref_w = args


    t_start, t_end = ref_w.tstart, ref_w.tend
    ref_slice = ref_w.time_slice(t_start, t_end)
    N = len(ref_slice.data)

    start_idx = int(round((t_start - ref_w.sample_times[0]) / ref_w.delta_t))
    end_idx = start_idx + N

    slice_w = Waveform(w[start_idx:end_idx])
    slice_r = Waveform(ref_slice)

    # Hard guarante
    if len(slice_w) != len(slice_r):
        return None

    return slice_w, slice_r




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
    end_time = reference_waveform.tend
    dt = time[1] - time[0]

    #Compute leakage over time for each recontructed waveform 
    for i, waveform in enumerate(reconstructed): 
        for j in range(20): 
            try:
                leaked_hrss[i,j] = np.sqrt(np.nansum(waveform.time_slice(end_time + j*dt, end_time + (j+1)*dt).data ** 2)) / injected_hrss 
            except IndexError: 
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



