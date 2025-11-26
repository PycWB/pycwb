#Postprocessing for waveform reconstruction 

from pycwb.types.waveform import Waveform, load_waveform 
from tqdm import tqdm  # type: ignore
import numpy as np  # type: ignore
import logging 
import os 

#Put in a module ()

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


def load_waveforms(folder, ifo, load_injected = True, whitened = False, format = 'hdf'): 
    """
    Load reconstructed waveforms from a given folder.
    """
    trigger_folders = os.listdir(folder)
    reconstructed_waveforms, injected = [], [] 
    for f in tqdm(trigger_folders, desc="Loading waveforms"):
        try: 
            reconstructed_file_name = f"reconstructed_waveform_{ifo}_whitened.{format}" if whitened else f"reconstructed_waveform_{ifo}.{format}"
            injected_file_name = f"injected_strain_{ifo}.{format}" 
           
            #load reconstructed waveforms (r) for the selected trigger folder and ifo 
            r_waveform = load_waveform(os.path.join(folder,f, reconstructed_file_name))
            if load_injected: 
                i_waveform = load_waveform(os.path.join(folder,f, injected_file_name), resample=r_waveform._delta_t)
                injected.append(i_waveform) 
            reconstructed_waveforms.append(r_waveform)

        except (ValueError, FileNotFoundError): 
            pass 

    #TODO: IMPLEMENT WHITNENING OPTIONS AND POSTPROD 
    return reconstructed_waveforms, injected


def sync_waveforms(waveforms, reference, sync_phase = True):
    """
    Synchronize all waveforms to a reference waveform.
    """
    sync_waveforms = [] 
    discarded_waveforms = 0
    
    #If 1 reference waveform is given, sync all waveforms to it
    if type(reference) == Waveform:
        for waveform in tqdm(waveforms, desc="Synchronizing waveforms"):
            try: 
                waveform.syncWaveform(reference, sync_phase = sync_phase)
                sync_waveforms.append(waveform)
            except ValueError:
                discarded_waveforms += 1
                pass 

    #If the same number of waveforms and references is given, sync each waveform to its respective reference 
    elif type(reference) == list and len(reference) == len(waveforms):
        for i, waveform in tqdm(enumerate(waveforms), desc="Synchronizing waveforms"):
            try: 
                waveform.syncWaveform(reference[i], sync_phase = sync_phase)
                sync_waveforms.append(waveform)
            except ValueError:
                discarded_waveforms += 1
                pass
    
    else: 
        raise ValueError("Reference waveform must either be a single Waveform or a list of Waveforms with the same length as the waveforms to be synchronized.") 
    if discarded_waveforms > 0:
        logger.warning(f"{discarded_waveforms} over {len(waveforms)} waveforms were discarded during synchronization due to errors .")

    return sync_waveforms 


def slice_waveforms(waveforms, reference_waveforms): 
    """
    Slice all waveforms to the same time range as the reference waveforms. If len(reference_waveform) == 1, slice all waveforms to the same time range, 
    otherwise slice each waveform to its respective reference.
    """ 
    sliced_list, reference_waveforms_ = [], [] 
    
    discarded_waveforms = 0
    #Raise Value Error if the number of reference waveforms is not 1 or equal to the number of waveforms 
    if type(reference_waveforms) == Waveform:
        t_start = reference_waveforms.tstart 
        t_end = reference_waveforms.tend 
        N = np.size(reference_waveforms.time_slice(t_start, t_end)) 
        reference_waveforms_ = Waveform(reference_waveforms.time_slice(t_start, t_end))
        for w in tqdm(waveforms, desc = "Slicing waveforms"): 
            #Find the start index in the waveform sample time
            start_idx = np.argmin(np.abs(w.sample_times - t_start))
            end_idx = start_idx + N

            #Append the sliced waveform to the list
            try: 
                sliced_list.append(Waveform(w[start_idx:end_idx]))

            except ValueError:
                discarded_waveforms += 1

    #If N reference waveforms are given, slice each waveform to its respective reference
    elif type(reference_waveforms) == list and len(reference_waveforms) == len(waveforms):
        for i, w in tqdm(enumerate(waveforms), desc = "Slicing waveforms"): 
            #Find start and end time from the respective reference waveform
            ref_w = reference_waveforms[i] 
            t_start, t_end = ref_w.tstart, ref_w.tend
            N = np.size(ref_w.time_slice(t_start, t_end))
            #Find the start index in the waveform sample time
            start_idx = np.argmin(np.abs(w.sample_times - t_start))
            end_idx = start_idx + N

            #Append the sliced waveform to the list
            try: 
                slice_w = Waveform(w[start_idx:end_idx])
                slice_r = Waveform(ref_w.time_slice(t_start, t_end))
                sliced_list.append(slice_w)
                reference_waveforms_.append(slice_r)


            except ValueError:
                discarded_waveforms += 1
                
    else: 
        raise ValueError("Reference waveform must either be a single Waveform or a list of Waveforms with the same length as the waveforms to be synchronized.")
    if discarded_waveforms > 0:
        logger.warning(f"{discarded_waveforms} over {len(waveforms)} waveforms were discarded during synchronization due to errors .")    

    return sliced_list, reference_waveforms_


def compute_confidence_intervals(waveforms, confidence_level=0.95, method = "percentiles"):
    """
    Compute the confidence intervals for a list of waveforms.
    """
    #Assuming waveforms are sliced to the same time range
    if method == "percentiles": 
        lower_bound, upper_bound = np.percentile(waveforms, [(1-confidence_level)*100/2, (1+confidence_level)*100/2], axis = 0)
    
    #Return lower datas and upper bound based on LC
    elif method == "upper": 
        lower_bound, upper_bound = np.percentile(waveforms, [0, confidence_level * 100], axis = 0)

    #Return lower datas and upper bound based on CL
    elif method == "lower": 
        lower_bound, upper_bound = np.percentile(waveforms, [(1-confidence_level) * 100, 100], axis = 0)
        
    else: 
        raise ValueError("Unsupported ordering method.") 
    
    return lower_bound, upper_bound 

def compute_overlap(reconstructed, injected):  
    """
    Compute the overlap between the reconstructed waveforms and the injected waveform."""
    
    #Compute overlap of N waveforms against a single, reference  waveform 
    if type(injected) == Waveform: 
        reconstructed = np.atleast_2d(reconstructed) 
        norm1 = np.linalg.norm(reconstructed, axis = 1)
        norm2 = np.linalg.norm(injected)
        overlaps = np.dot(reconstructed, injected) / (norm1 * norm2)

    #Compute eoverlap of N waveforms against their respective injected waveform 
    elif type(injected) == list and len(injected) == len(reconstructed): 
        overlaps = [] 
        for reconstructed_waveform, injected_waveform in zip(reconstructed, injected): 
            overlap = compute_overlap(reconstructed_waveform, injected_waveform)
            overlaps.append(overlap)
    #Raise error if lengths do not match
    else: 
        raise ValueError("Injected waveform list length must be 1 or equal to the number of reconstructed waveforms.") 
    
    overlaps = np.array(overlaps)

    if overlaps.size == 1:
        return overlaps.item() 

    return overlaps 

def compute_cumulative_hrss(waveform, delta_t, axis = 1): 
    """
    Compute the cumulative hrss of a waveform.
    """
    hrss = np.sqrt(np.cumsum(np.array(waveform) * np.conj(waveform), axis=axis) * delta_t)
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
                leaked_hrss[i,j] = np.sqrt(np.sum(np.square(waveform.time_slice(end_time + j*dt, end_time + (j+1)*dt).data))) / injected_hrss 
            except IndexError: 
                pass 
        
    #Return mean and std of leakage over all reconstructed waveforms
    mean_leakage = np.mean(leaked_hrss, axis=0)
    std_leakage = np.std(leaked_hrss, axis=0) / np.sqrt(len(reconstructed))
    return mean_leakage, std_leakage 

def compute_hrss(waveform, delta_t):
    """
    Compute the hrss for a list of waveforms.
    """
    return np.sqrt(np.sum(np.array(waveform) * np.conj(waveform)) * delta_t) 



