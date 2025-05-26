import logging
import os
import psutil
from pycwb.config import Config
from pycwb.modules.catalog import add_events_to_catalog
from pycwb.modules.super_cluster.super_cluster import supercluster_wrapper
from pycwb.modules.super_cluster.supercluster import supercluster
from pycwb.modules.xtalk.monster import load_catalog
from pycwb.modules.coherence.coherence import coherence
from pycwb.modules.read_data import generate_injection, generate_noise_for_job_seg, read_from_job_segment, check_and_resample
from pycwb.modules.data_conditioning import data_conditioning, regression, whitening_mesa, whitening_cwb

from pycwb.modules.likelihood import likelihood
from pycwb.types.job import WaveSegment
from pycwb.types.network import Network
from pycwb.modules.workflow_utils.job_setup import print_job_info
from pycwb.utils.dataclass_object_io import save_dataclass_to_json
from pycwb.workflow.subflow.postprocess_and_plots import plot_trigger_flow, reconstruct_waveforms_flow, plot_skymap_flow
from scipy.stats import anderson 
import numpy as np 

logger = logging.getLogger(__name__)


def job_segment_conditioning(working_dir: str, config: Config, job_seg: WaveSegment, compress_json):
    print_job_info(job_seg)

    if not job_seg.frames and not job_seg.noise and not job_seg.injections:
        raise ValueError("No data to process")

    data = None
    pvalues = {}
    means = {} 
    stds = {} 

    if job_seg.frames:
        data = read_from_job_segment(config, job_seg)
    if job_seg.noise:
        data = generate_noise_for_job_seg(job_seg, config.inRate, data=data)
    if job_seg.injections:
        data = generate_injection(config, job_seg, data)

    logger.info("Memory usage: %f.2 MB", psutil.Process().memory_info().rss / 1024 / 1024)

    #check and resample the data 
    data = [check_and_resample(data[i], config, i) for i in range(len(job_seg.ifos))]

    
    data_regression = [regression(config, h) for h in data] 
     

    if config.whiteMethod == 'wavelet': 
        tf_maps, nrms = zip(*[whitening_cwb(config, h) for h in data_regression])

    elif config.whiteMethod == 'mesa':
        tf_maps, nrms = zip(*[whitening_mesa(config, h) for h in data_regression])
    
    sample_rate = config.inRate / 2 ** (config.levelR) 
    print(sample_rate) 
    logger.info("Applying data conditionind and testing with Anderson")
    #Compute p-value with anderson test every 20 seconds 
    step = int(sample_rate * config.whiteStride) 
    for i, ifo in enumerate(job_seg.ifos): 
        pvalues[ifo] = [] 
        means[ifo] = [] 
        stds[ifo] = [] 
        size = len(tf_maps[i].data.data) 
        for j in range(1,size // step - 1): 
            p, m, s = compute_statistics(tf_maps[i].data.data[j * step : (j+1) * step])
            pvalues[ifo].append(p)
            means[ifo].append(m)
            stds[ifo].append(s) 

    logger.info("Memory usage: %f.2 MB", psutil.Process().memory_info().rss / 1024 / 1024)

    save_conditioning(config, nrms, pvalues, means, stds, working_dir, job_seg) 

def save_conditioning(config, nrms, pvalues, means, stds, working_dir, job_seg):
    nrms_folder = f'{working_dir}/nrms'
    anderson_folder = f'{working_dir}/anderson' 
    mean_folder = f'{working_dir}/mean'
    std_folder = f'{working_dir}/std' 

    for folder in [nrms_folder, anderson_folder, mean_folder, std_folder]: 
        os.makedirs(folder, exist_ok = True) 
    
    for i, ifo in enumerate(config.ifo): 
        np.save(f'{nrms_folder}/nrms_{job_seg.index}_{ifo}', nrms[i].data.data)
        np.save(f'{anderson_folder}/pvalues_{job_seg.index}_{ifo}', pvalues[ifo])    
        np.save(f'{mean_folder}/mean_{job_seg.index}_{ifo}', means[ifo])
        np.save(f'{std_folder}/std_{job_seg.index}_{ifo}', stds[ifo])

def compute_statistics(data): 
    pvalue = anderson_pval(data) 
    mean = data.mean() 
    std = data.std() 
    if pvalue > 1: 
        pvalue, mean, std = (np.nan,) * 3 
    if pvalue < 1e-5: 
        pvalue, mean, std = (np.nan,) * 3
    return pvalue, mean, std


def anderson_pval(data): 
    """performs an Anderson Darling test on input data and returns its pvalue """ 
    
    statistics = anderson(data)[0]
    if statistics <= 0.2: 
        pvalue = 1 - np.exp(-13.436 + 101.14 * (statistics)- 223.73 * statistics ** 2) 
    
    elif statistics > 0.2 and statistics <= 0.34: 
        pvalue = 1 - np.exp(-8.318 + 42.796 * statistics- 59.938* statistics ** 2)
    
    elif statistics > 0.34 and statistics < .6: 
        pvalue = np.exp(0.9177 - 4.279 * statistics - 1.38 * statistics ** 2)
    
    elif statistics >= .6: 
        pvalue = np.exp(1.2937 - 5.709 * statistics + 0.0186 * statistics ** 2)
        
    return pvalue




