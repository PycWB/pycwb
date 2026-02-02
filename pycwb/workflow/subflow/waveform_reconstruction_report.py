from pycwb.types.waveform import load_waveform
import sys 
from pycwb.modules.post_production.waveform_reconstruction_plot import *
from pycwb.modules.post_production.waveform_reconstruction import load_waveforms, sync_waveforms, slice_waveforms, pad_waveforms
from numpy.linalg import norm 
import os 
import numpy as np  # pyright: ignore[reportMissingImports]
import logging

#this in the subflow and is call via CLI 

#TODO: Implement Single Folder Analysis (almost done)

logger = logging.getLogger(__name__)


def process_strain(folder, ifo, reference_folder = None, whitened = False, confidence_level = .95, **kwargs):  
    """
    Process the folder containing the waveforms and perform analysis.
    Parameters: 
        folder (str): Path to the folder containing the analysis.
        ifo (str): Interferometer to process.
        reference_folder (str, None): Path to the folder containing the reference injected waveforms. If None, the first folder in the list is used.
        args (argparse.Namespace): Command line arguments.
    """ 
    #Define parameters from kwargs 
    ordering = kwargs.get('ordering', 'percentiles')
    plot_median = kwargs.get('plot_median', False)
    max_workers = kwargs.get('max_workers', 8)
    waveform_format = kwargs.get('waveform_format', 'hdf') 

    logger.info(f"Processing waveform reconstruction analysis for {ifo} in folder: {folder}")

    #Define the folder for loadingthe data and saving the results 
    plots_folder = os.path.join(folder, "reports/plots")
    results_folder = os.path.join(folder, "reports/results")
    triggers_folder = os.path.join(folder, "trigger")

    os.makedirs(plots_folder, exist_ok=True)
    os.makedirs(results_folder, exist_ok=True) 

    #Create reference folder. If None, choose a random folder. If starts with 'trigger_' consider it as a full path inside current folder. 
    if reference_folder is None:
        logger.warning("Reference folder not provided. Using the first folder in triggers as reference.")
        reference_folder =  next(os.listdir(triggers_folder)) 
        reference_folder = os.path.join(triggers_folder, reference_folder)

    logger.info(f"Using absolute reference from folder: {reference_folder}")

    if whitened: 
        reference_name = f"reconstructed_waveform_{ifo}_whitened.{waveform_format}"
    else:
        reference_name = f"reconstructed_waveform_{ifo}.{waveform_format}" 



    #Load all the reconstructed waveforms if not already loaded in the previous step 
    reconstructed_waveforms = load_waveforms(triggers_folder, ifo, whitened = whitened, skip_trigger=True, load_injected = False, format = waveform_format, max_workers=max_workers) 
    reference_waveform = load_waveform(os.path.join(reference_folder, reference_name), resample=reconstructed_waveforms[0]._delta_t)        
    reconstructed_waveforms, reference_waveforms = sync_waveforms(reconstructed_waveforms, reference_waveform, sync_phase = True, max_workers=max_workers)


    #Plot time Leakage 
    logger.info("Plotting Leakage")
    leakage_fig, leakage_data = plot_leakage(reconstructed_waveforms, reference_waveform)
    leakage_fig.savefig(os.path.join(plots_folder, f"leakage_{ifo}.png"), bbox_inches='tight')
    filename = f"leakage_{ifo}_wth" if whitened else f"leakage_{ifo}"
    leakage_fig.savefig(os.path.join(plots_folder, f"{filename}.png"), bbox_inches='tight') 
    np.savez(os.path.join(results_folder, f"{filename}.npz"), **leakage_data) 

    #Slice the waveforms for further comparison. Only store the reference waveform as a single object (they're N copies for the same waveform)
    
    
    reconstructed_waveforms, reference_waveforms = pad_waveforms(reconstructed_waveforms, reference_waveforms, max_workers=max_workers)
    reconstructed_waveforms, reference_waveforms = slice_waveforms(reconstructed_waveforms, reference_waveforms, max_workers=max_workers) 
    
    reference_waveform = reference_waveforms[0]

    #Plot the time domain waveforms with CI 
    logger.info("Plotting time domain waveforms")
    twaveform_fig, twaveform_data = plot_time_waveform_reconstruction(reconstructed_waveforms, reference_waveform, 
                                          confidence_level = confidence_level, percentile_method = ordering, plot_median = plot_median)  
    
    filename = f"time_waveform_reconstruction_{ifo}_wth" if whitened else f"time_waveform_reconstruction_{ifo}"
    twaveform_fig.savefig(os.path.join(plots_folder, f"{filename}.png"), bbox_inches='tight')
    np.savez(os.path.join(results_folder, f"{filename}.npz"), **twaveform_data)

    #Plot the time domain bias with CI 

    tbias_fig, tbias_data = plot_time_bias(reconstructed_waveforms, reference_waveform, confidence_level = confidence_level, percentile_method = ordering, normalize = False)
    
    filename = f"time_bias_{ifo}_wth" if whitened else f"time_bias_{ifo}"
    tbias_fig.savefig(os.path.join(plots_folder, f"{filename}.png"), bbox_inches='tight') 
    np.savez(os.path.join(results_folder, f"{filename}.npz"), **tbias_data) 

    #Plot the overlap 
    overlap_fig, overlap_data = plot_overlap(reconstructed_waveforms, reference_waveform)
    filename = f"overlap_{ifo}_wth" if whitened else f"overlap_{ifo}"
    overlap_fig.savefig(os.path.join(plots_folder, f"{filename}.png"),  bbox_inches='tight')
    np.savez(os.path.join(results_folder, f"{filename}.npz"), **overlap_data)

    #Plot the time domain cumulative hrss 
    chrss_fig, chrss_data = plot_time_cumulative_hrss(reconstructed_waveforms, reference_waveform, confidence_level = confidence_level, percentile_method = ordering,\
                                plot_median = plot_median)
    chrss_fig.savefig(os.path.join(plots_folder, f"cumulative_hrss_{ifo}.png"), bbox_inches='tight') 
    filename = f"cumulative_hrss_{ifo}_wth" if whitened else f"cumulative_hrss_{ifo}"
    chrss_fig.savefig(os.path.join(plots_folder, f"{filename}.png"), bbox_inches='tight')
    np.savez(os.path.join(results_folder, f"{filename}.npz"), **chrss_data)

    #Plot the HRSS histogram 
    hrss_fig, hrss_data = plot_hrss(reconstructed_waveforms, reference_waveform) 
    hrss_fig.savefig(os.path.join(plots_folder, f"hrss_{ifo}.png"), bbox_inches='tight') 
    filename = f"hrss_{ifo}_wth" if whitened else f"hrss_{ifo}"
    hrss_fig.savefig(os.path.join(plots_folder, f"{filename}.png"), bbox_inches='tight')
    np.savez(os.path.join(results_folder, f"{filename}.npz"), **hrss_data)

    #Switch to frequency domain for further analysis 
    [reconstructed.fft() for reconstructed in reconstructed_waveforms]

    #Plot the frequency domain waveforms with CI
    logger.info('Plotting frequency domain waveforms')
    fwaveform_fig, fwaveform_data = plot_frequency_waveform_reconstruction(reconstructed_waveforms, reference_waveform.fft(), 
                                            confidence_level = confidence_level, percentile_method = ordering, plot_median = plot_median)
    
    filename = f"frequency_waveform_reconstruction_{ifo}_wth" if whitened else f"frequency_waveform_reconstruction_{ifo}" 
    fwaveform_fig.savefig(os.path.join(plots_folder, f"{filename}.png"), bbox_inches='tight') 
    np.savez(os.path.join(results_folder, f"{filename}.npz"), **fwaveform_data)

    #Plot the frequency domain bias with CI
    fbias_fig, fbias_data = plot_frequency_bias(reconstructed_waveforms, reference_waveform, confidence_level = confidence_level, percentile_method = ordering)
    filename = f"frequency_bias_{ifo}_wth" if whitened else f"frequency_bias_{ifo}"
    fbias_fig.savefig(os.path.join(plots_folder, f"{filename}.png"), bbox_inches='tight') 
    np.savez(os.path.join(results_folder, f"{filename}.npz"), **fbias_data) 

    #Plot the frequency domain cumulative hrss
    fchrss_fig, fchrss_data = plot_frequency_cumulative_hrss(reconstructed_waveforms, reference_waveform, confidence_level = confidence_level, percentile_method = ordering,\
                                plot_median = plot_median)

    filename = f"frequency_cumulative_hrss_{ifo}_wth" if whitened else f"frequency_cumulative_hrss_{ifo}"
    fchrss_fig.savefig(os.path.join(plots_folder, f"{filename}.png"), bbox_inches='tight')
    np.savez(os.path.join(results_folder, f"{filename}.npz"), **fchrss_data)
    
    plt.close('all') 


