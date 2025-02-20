import logging
import os
import psutil
from pycwb.config import Config
from pycwb.modules.catalog import add_events_to_catalog
from pycwb.modules.super_cluster.super_cluster import supercluster_wrapper
from pycwb.modules.super_cluster.supercluster import supercluster
from pycwb.modules.xtalk.monster import load_catalog
from pycwb.modules.coherence.coherence import coherence
from pycwb.modules.read_data import generate_injection, generate_noise_for_job_seg, read_from_job_segment
from pycwb.modules.data_conditioning import data_conditioning
from pycwb.modules.likelihood import likelihood
from pycwb.types.job import WaveSegment
from pycwb.types.network import Network
from pycwb.modules.workflow_utils.job_setup import print_job_info
from pycwb.utils.dataclass_object_io import save_dataclass_to_json
from pycwb.workflow.subflow.postprocess_and_plots import plot_trigger_flow, reconstruct_waveforms_flow, plot_skymap_flow
from scipy.stats import anderson 


logger = logging.getLogger(__name__)


def job_segment_conditioning(working_dir: str, config: Config, job_seg: WaveSegment):
    print_job_info(job_seg)

    if not job_seg.frames and not job_seg.noise and not job_seg.injections:
        raise ValueError("No data to process")

    data = None

    if job_seg.frames:
        data = read_from_job_segment(config, job_seg)
    if job_seg.noise:
        data = generate_noise_for_job_seg(job_seg, config.inRate, data=data)
    if job_seg.injections:
        data = generate_injection(config, job_seg, data)

    logger.info("Memory usage: %f.2 MB", psutil.Process().memory_info().rss / 1024 / 1024)

    tf_maps, nRMS_list = data_conditioning(config, data)
    logger.info("Memory usage: %f.2 MB", psutil.Process().memory_info().rss / 1024 / 1024)

    pvalues = [anderson_test(tf_map.data) for tf_map in tf_maps]
    
    save_pvalue(pvalues, config)
    

def anderson_test(data): 
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

def save_pvalue(values, config, working_dir: str, sub_dir: str):
    test_folder = '/'.join([working_dir, sub_dir]) 
    if not os.path.exists(test_folder): 
        os.makedirs(test_folder)
    else: 
        print(f"Test folder {test_folder} already exists, skip") 
    filename = '/'.join([test_folder,f'anderson_{config.whiteMethod}.txt']) 
    
    print(f'Saving Anderson pvalue') 
     try:
        with open(filename, 'a') as f:  # Open file in append mode
            f.write(' '.join(map(str, values)) + '\n')  # Append value with newline
    except FileNotFoundError:
        with open(filename, 'w') as f:  
            f.write(' '.join(config.ifos) + \n)
            f.write(' '.join(map(str, values))+ '\n')
            
    return filename 



