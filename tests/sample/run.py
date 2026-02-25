import sys
import time

sys.path.insert(0, "../..")

from pycwb.config import Config
from pycwb.modules.logger import logger_init
from pycwb.modules.read_data.data_check import check_and_resample

logger_init()

config = Config()
config.load_from_yaml('./user_parameters_injection.yaml')
config.nproc = 1

from pycwb.modules.read_data import generate_injection, generate_noise_for_job_seg
from pycwb.modules.job_segment import create_job_segment_from_config

job_segments = create_job_segment_from_config(config)

data = generate_noise_for_job_seg(job_segments[0], config.inRate, f_low=config.fLow)
data = generate_injection(config, job_segments[0], data)

from pycwb.modules.data_conditioning.data_conditioning_python import data_conditioning

data = [check_and_resample(data[i], config, i) for i in range(len(job_segments[0].ifos))]

strains, nRMS = data_conditioning(config, data)

###### coherence

from pycwb.modules.cwb_coherence import coherence

fragment_clusters = coherence(config, strains)

###### super cluster

from pycwb.modules.super_cluster.super_cluster import supercluster_wrapper
from pycwb.modules.xtalk.monster import load_catalog

xtalk_coeff, xtalk_lookup_table, layers, _ = load_catalog(config.MRAcatalog)

super_fragment_clusters = supercluster_wrapper(config, None, fragment_clusters, strains,
                                            xtalk_coeff, xtalk_lookup_table, layers)
## ignore the release error below
# Exception ignored in: <function TimeFrequencySeries.__del__ at 0x1c932ee80>
# Traceback (most recent call last):
#   File "/Users/yumengxu/miniforge3/envs/pycwb_x64/lib/python3.11/site-packages/pycwb/types/time_frequency_series.py", line 63, in __del__
#   File "/Users/yumengxu/miniforge3/envs/pycwb_x64/lib/python3.11/site-packages/pycwb/types/wdm.py", line 119, in release
# AttributeError: 'CPyCppyy_NoneType' object has no attribute 'release'