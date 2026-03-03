from pycwb.types.waveform import load_waveform
from concurrent.futures import ProcessPoolExecutor
import sys 
from pycwb.modules.post_production.waveform_reconstruction_plot import *
#from pycwb.modules.post_production.waveform_reconstruction import load_and_slice#, sync_waveforms, slice_waveforms, pad_waveforms
sys.path.insert(0,'/home/alessandro.martini/pycwb/pycwb/modules/post_production')
from waveform_reconstruction import load_and_slice
from numpy.linalg import norm 
import os 
from pathlib import Path
from scipy.stats import mode 
import numpy as np  # pyright: ignore[reportMissingImports]
import logging

#this in the subflow and is call via CLI 

#TODO: Implement Single Folder Analysis (almost done)

logger = logging.getLogger(__name__)


def process_strain(folder, ifo, reference = None, whitened = False, confidence_level = .95, **kwargs):  
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
    max_workers = kwargs.get('max_workers', 8)
    waveform_format = kwargs.get('waveform_format', 'hdf') 

    logger.info(f"Processing waveform reconstruction analysis for {ifo} in folder: {folder}")

    #Define the folder for loadingthe data and saving the results 
    plots_folder = os.path.join(folder, "reports/plots")
    results_folder = os.path.join(folder, "reports/results")
    os.makedirs(plots_folder, exist_ok=True)
    os.makedirs(results_folder, exist_ok=True) 
    triggers_folder = Path(folder) / "trigger"

    filename = f'{ifo}_wf_REC.{waveform_format}' if not whitened else f'{ifo}_wf_REC_whiten.{waveform_format}'
    triggers_directories = [str(d.resolve() / filename) for d in triggers_folder.iterdir() if d.is_dir() and f'{ifo}_wf_INJ.{waveform_format}' in [f.name for f in d.iterdir()]]

    #Load and slice all wavefroms to be used for computations
    early_start = kwargs.get('early_start', 0)
    scale = kwargs.get('scale', 0)
    print(scale)
    args = [(t_dir, reference, None, None, early_start, scale) for t_dir in triggers_directories]

    with ProcessPoolExecutor(max_workers=8 or os.cpu_count()) as exe:
        reconstructed_waveforms = list(tqdm(exe.map(load_and_slice, args), total= len(args)))
    reconstructed_waveforms, reference_waveform = map(list,zip(*[wf for wf in reconstructed_waveforms if wf is not None])) 


    tot_waveforms = len(reconstructed_waveforms) 
    lengths = [len(wf) for wf in reconstructed_waveforms] 
    mode_length = mode(lengths).mode
    if not all(length == mode_length for length in lengths):
        reconstructed_waveforms = [wf for wf in reconstructed_waveforms if len(wf) == mode_length] 
        print(f"Removed waveforms with lengths different from the mode length. Number of waveforms removed: {tot_waveforms - len(reconstructed_waveforms)}. Mode length: {mode_length}")

    reference_waveform = next((r for r in reference_waveform if len(r) == mode_length), None)  

    #Plot the time domain waveforms with CI 
    logger.info("Plotting time domain waveforms")
    twaveform_fig, twaveform_data = plot_time_waveform_reconstruction(reconstructed_waveforms, reference_waveform, 
                                          confidence_level = confidence_level, percentile_method = ordering, **kwargs)  
    
    filename = f"time_waveform_reconstruction_{ifo}_wth" if whitened else f"time_waveform_reconstruction_{ifo}"
    twaveform_fig.savefig(os.path.join(plots_folder, f"{filename}.png"), bbox_inches='tight')
    np.savez(os.path.join(results_folder, f"{filename}.npz"), **twaveform_data)

    #Plot the time domain bias with CI 

    tbias_fig, tbias_data = plot_time_bias(reconstructed_waveforms, reference_waveform, confidence_level = confidence_level, percentile_method = ordering, normalize = False,
    **kwargs)
    
    filename = f"time_bias_{ifo}_wth" if whitened else f"time_bias_{ifo}"
    tbias_fig.savefig(os.path.join(plots_folder, f"{filename}.png"), bbox_inches='tight') 
    np.savez(os.path.join(results_folder, f"{filename}.npz"), **tbias_data) 

    #Plot the overlap 
    overlap_fig, overlap_data = plot_overlap(reconstructed_waveforms, reference_waveform, **kwargs)
    filename = f"overlap_{ifo}_wth" if whitened else f"overlap_{ifo}"
    overlap_fig.savefig(os.path.join(plots_folder, f"{filename}.png"),  bbox_inches='tight')
    np.savez(os.path.join(results_folder, f"{filename}.npz"), **overlap_data)

    #Plot the time domain cumulative hrss 
    chrss_fig, chrss_data = plot_time_cumulative_hrss(reconstructed_waveforms, reference_waveform, confidence_level = confidence_level, percentile_method = ordering,\
                                **kwargs)
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
                                            confidence_level = confidence_level, percentile_method = ordering, **kwargs)
    
    filename = f"frequency_waveform_reconstruction_{ifo}_wth" if whitened else f"frequency_waveform_reconstruction_{ifo}" 
    fwaveform_fig.savefig(os.path.join(plots_folder, f"{filename}.png"), bbox_inches='tight') 
    np.savez(os.path.join(results_folder, f"{filename}.npz"), **fwaveform_data)

    #Plot the frequency domain bias with CI
    fbias_fig, fbias_data = plot_frequency_bias(reconstructed_waveforms, reference_waveform, confidence_level = confidence_level, percentile_method = ordering, **kwargs)
    filename = f"frequency_bias_{ifo}_wth" if whitened else f"frequency_bias_{ifo}"
    fbias_fig.savefig(os.path.join(plots_folder, f"{filename}.png"), bbox_inches='tight') 
    np.savez(os.path.join(results_folder, f"{filename}.npz"), **fbias_data) 

    #Plot the frequency domain cumulative hrss
    fchrss_fig, fchrss_data = plot_frequency_cumulative_hrss(reconstructed_waveforms, reference_waveform, confidence_level = confidence_level, percentile_method = ordering,\
                                **kwargs)

    filename = f"frequency_cumulative_hrss_{ifo}_wth" if whitened else f"frequency_cumulative_hrss_{ifo}"
    fchrss_fig.savefig(os.path.join(plots_folder, f"{filename}.png"), bbox_inches='tight')
    np.savez(os.path.join(results_folder, f"{filename}.npz"), **fchrss_data)
    

    null_name = f"{ifo}_wf_NUL.{waveform_format}" if not whitened else f"{ifo}_wf_NUL_whiten.{waveform_format}"
    args = [(Path(wf.folder) / null_name, reference, wf._total_time_shift, None, early_start, scale) for wf in reconstructed_waveforms]

    del(reconstructed_waveforms)
    with ProcessPoolExecutor(max_workers = max_workers) as exe: 
        null_waveforms = list(tqdm(exe.map(load_and_slice, args), total = len(args)))

    null_waveforms, reference_waveform = zip(*[r for r in null_waveforms if r is not None]) 
    reference_waveform = reference_waveform[0]
    tnull_fig, tnull_results = plot_time_waveform_reconstruction(null_waveforms, confidence_level = confidence_level, 
                                                percentile_method = ordering, **kwargs)  
    filename=f"null_reconstruction_{ifo}_wth" if whitened else f"null_reconstruction_{ifo}" 
    tnull_fig.savefig(os.path.join(plots_folder, f"{filename}.png"), bbox_inches='tight')
    np.savez(os.path.join(results_folder, f"{filename}.npz"), **tnull_results)
