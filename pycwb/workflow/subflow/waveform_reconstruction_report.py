from pycwb.types.waveform import load_waveform
from pycwb.modules.post_production.waveform_reconstruction import load_waveforms, sync_waveforms, slice_waveforms
from pycwb.modules.post_production.waveform_reconstruction_plot import *
from numpy.linalg import norm 
import os 
import numpy as np  # pyright: ignore[reportMissingImports]
import logging

#this in the subflow and is call via CLI 

#TODO: Implement Single Folder Analysis (almost done)

logger = logging.getLogger(__name__)


def process_strain(folder, ifo, reference_folder, confidence_level, use_absolute_reference, use_relative_reference, waveform_format, ordering, plot_median):  
    """
    Process the folder containing the waveforms and perform analysis.
    Parameters: 
        folder (str): Path to the folder containing the analysis.
        ifo (str): Interferometer to process.
        reference_folder (str, None): Path to the folder containing the reference injected waveforms. If None, the first folder in the list is used.
        args (argparse.Namespace): Command line arguments.
    """ 
    print(f"Processing waveform reconstruction analysis for {ifo} in folder: {folder}")

    #Define the folder for loadingthe data and saving the results 
    triggers_folder = os.path.join(folder, "trigger")
    plots_folder = os.path.join(folder, "reports/plots")
    results_folder = os.path.join(folder, "reports/results")

    #Create folders if thye do not exist 
    os.makedirs(plots_folder, exist_ok=True)
    os.makedirs(results_folder, exist_ok=True) 

    #If True, plots statistics of waveforms with respect to relative reference (Significantly increases computational times)
    if use_relative_reference:
        logger.info("Using relative reference for waveform reconstruction analysis.")
        #Load all the reconstructed and injected waveforms, synchronize and slice them
        reconstructed_waveforms, injected_waveforms = load_waveforms(triggers_folder, ifo, load_injected = True, format = waveform_format) 
        reconstructed_waveforms = sync_waveforms(reconstructed_waveforms, injected_waveforms, sync_phase = True)
        reconstructed_waveforms_sliced, injected_waveforms = slice_waveforms(reconstructed_waveforms, injected_waveforms) 

        #Plot the overlap wrt to the relative injected waveform
        logger.info("Plotting overlap against their relative reference")
        auto_ovlp_fig, auto_ovlp_data = plot_auto_overlap(reconstructed_waveforms_sliced, injected_waveforms)
        auto_ovlp_fig.savefig(os.path.join(plots_folder, f"auto_overlap_{ifo}.png"),  bbox_inches='tight') 
        np.savez(os.path.join(results_folder, f"auto_overlap_{ifo}.npz"), **auto_ovlp_data) 
    
        #Delete unused variables to free memory
        del(reconstructed_waveforms_sliced)
        del(injected_waveforms)

    if use_absolute_reference: 
    
        #Load all the reconstructed waveforms if not already loaded in the previous step
        if "reconstructed_waveforms" not in locals():
            reconstructed_waveforms, reference_waveform = load_waveforms(triggers_folder, ifo, load_injected = False, format = waveform_format) 
            logger.info(f"Loaded {len(reconstructed_waveforms)} reconstructed waveforms for {ifo}.")
        #Load the reference injected waveform from a common trigger folder. If no folder is given, take the first one in the list 
        if reference_folder is None:
            reference_folder =  os.path.join(triggers_folder, os.listdir(triggers_folder)[0]) 
        else: 
            reference_folder = os.path.join(triggers_folder, reference_folder)
    
        logger.info(f"Using absolute reference from folder: {reference_folder}")
        reference_waveform = load_waveform(os.path.join(reference_folder, f"injected_strain_{ifo}.{waveform_format}"), resample=reconstructed_waveforms[0]._delta_t)
        reconstructed_waveforms = sync_waveforms(reconstructed_waveforms, reference_waveform, sync_phase = True)

        #Plot time Leakage 
        logger.info("Plotting Leakage")
        leakage_fig, leakage_data = plot_leakage(reconstructed_waveforms, reference_waveform)
        leakage_fig.savefig(os.path.join(plots_folder, f"leakage_{ifo}.png"), bbox_inches='tight')
        np.savez(os.path.join(results_folder, f"leakage_{ifo}.npz"), **leakage_data) 

        #Slice the waveforms for further comparison 
        reconstructed_waveforms, injected_waveform = slice_waveforms(reconstructed_waveforms, reference_waveform) 
    
        #Plot the time domain waveforms with CI 
        logger.info("Plotting time domain waveforms")
        twaveform_fig, twaveform_data = plot_time_waveform_reconstruction(reconstructed_waveforms, injected_waveform, 
                                          confidence_level = confidence_level, percentile_method = ordering, plot_median = plot_median) 
        twaveform_fig.savefig(os.path.join(plots_folder, f"time_waveform_reconstruction_{ifo}.png"), bbox_inches='tight')
        np.savez(os.path.join(results_folder, f"time_waveform_reconstruction_{ifo}.npz"), **twaveform_data)

        #Plot the time domain bias with CI 
        tbias_fig, tbias_data = plot_time_bias(reconstructed_waveforms, injected_waveform, confidence_level = confidence_level, percentile_method = ordering, normalize = False)
        tbias_fig.savefig(os.path.join(plots_folder, f"time_bias_{ifo}.png")) 
        np.savez(os.path.join(results_folder, f"time_bias_{ifo}.npz"), **tbias_data) 

        #Plot the overlap 
        overlap_fig, overlap_data = plot_overlap(reconstructed_waveforms, injected_waveform)
        overlap_fig.savefig(os.path.join(plots_folder, f"overlap_{ifo}.png"),  bbox_inches='tight')
        np.savez(os.path.join(results_folder, f"overlap_{ifo}.npz"), **overlap_data)

        #Plot the time domain cumulative hrss 
        chrss_fig, chrss_data = plot_time_cumulative_hrss(reconstructed_waveforms, injected_waveform, confidence_level = confidence_level, percentile_method = ordering,\
                                 plot_median = plot_median)
        chrss_fig.savefig(os.path.join(plots_folder, f"cumulative_hrss_{ifo}.png"), bbox_inches='tight') 
        np.savez(os.path.join(results_folder, f"cumulative_hrss_{ifo}.npz"), **chrss_data)

        #Plot the HRSS histogram 
        hrss_fig, hrss_data = plot_hrss(reconstructed_waveforms, injected_waveform) 
        hrss_fig.savefig(os.path.join(plots_folder, f"hrss_{ifo}.png"), bbox_inches='tight') 
        np.savez(os.path.join(results_folder, f"hrss_{ifo}.npz"), **hrss_data)

        #Switch to frequency domain for further analysis 
        [reconstructed.fft() for reconstructed in reconstructed_waveforms]

        #Plot the frequency domain waveforms with CI
        logger.info('Plotting frequency domain waveforms')
        fwaveform_fig, fwaveform_data = plot_frequency_waveform_reconstruction(reconstructed_waveforms, injected_waveform.fft(), 
                                              confidence_level = confidence_level, percentile_method = ordering, plot_median = plot_median)
        fwaveform_fig.savefig(os.path.join(plots_folder, f"frequency_waveform_reconstruction_{ifo}.png"), bbox_inches='tight') 
        np.savez(os.path.join(results_folder, f"frequency_waveform_reconstruction_{ifo}.npz"), **fwaveform_data)

        #Plot the frequency domain bias with CI
        fbias_fig, fbias_data = plot_frequency_bias(reconstructed_waveforms, injected_waveform, confidence_level = confidence_level, percentile_method = ordering, normalize = False)
        fbias_fig.savefig(os.path.join(plots_folder, f"frequency_bias_{ifo}.png"), bbox_inches='tight') 
        np.savez(os.path.join(results_folder, f"frequency_bias_{ifo}.npz"), **fbias_data) 

        #Plot the frequency domain cumulative hrss
        fchrss_fig, fchrss_data = plot_frequency_cumulative_hrss(reconstructed_waveforms, injected_waveform, confidence_level = confidence_level, percentile_method = ordering,\
                                 plot_median = plot_median)
        fchrss_fig.savefig(os.path.join(plots_folder, f"frequency_cumulative_hrss_{ifo}.png"), bbox_inches='tight')
        np.savez(os.path.join(results_folder, f"frequency_cumulative_hrss_{ifo}.npz"), **fchrss_data)
        
        plt.close('all') 
        return 0 


