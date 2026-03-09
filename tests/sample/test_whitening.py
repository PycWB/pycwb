import sys
import time

sys.path.insert(0, "../..")

from pycwb.modules.data_conditioning.whitening_cwb import whitening_cwb
from pycwb.modules.read_data.data_check import check_and_resample
from pycwb.config import Config
from pycwb.modules.logger import logger_init
from pycwb.modules.read_data import generate_injection, generate_noise_for_job_seg
from pycwb.modules.job_segment import create_job_segment_from_config
from pycwb.modules.data_conditioning.regression import regression
from pycwb.modules.data_conditioning.data_conditioning_python import whitening_python
import numpy as np

logger_init()

config = Config()
config.load_from_yaml('./user_parameters_injection.yaml')
config.nproc = 1



job_segments = create_job_segment_from_config(config)


data = generate_noise_for_job_seg(job_segments[0], config.inRate, f_low=config.fLow)
data = generate_injection(config, job_segments[0], data)

data = [check_and_resample(data[i], config, i) for i in range(len(job_segments[0].ifos))]

data_regressions = [regression(config, h) for h in data]

res = [whitening_cwb(config, h) for h in data_regressions]
res_python = [whitening_python(config, h) for h in data_regressions]

conditioned_strains, nRMS_list = zip(*res)
conditioned_strains_python, nRMS_list_python = zip(*res_python)

conditioned_strains_relative_diff = np.abs(conditioned_strains[0].data.data - conditioned_strains_python[0].data) / np.max((np.abs(conditioned_strains[0].data.data)))
print(f"Max relative difference in conditioned strains: {(conditioned_strains_relative_diff.max()):.6e}")

from pycwb.modules.cwb_conversions import convert_time_frequency_series_to_wseries, WSeries_to_matrix

nRMS = WSeries_to_matrix(convert_time_frequency_series_to_wseries(nRMS_list[0]))
nRMS_py = nRMS_list_python[0].data

print(f"Check if nRMS shapes match: {nRMS_py.shape == nRMS.shape}, nRMS_py.shape = {nRMS_py.shape}, nRMS.shape = {nRMS.shape}")

if nRMS_py.shape == nRMS.shape:
	nRMS_relative_diff = np.abs(nRMS - nRMS_py) / np.maximum(np.abs(nRMS).max(), 1e-12)
	print(f"Max relative difference in nRMS TF maps: {nRMS_relative_diff.max():.6e}")
	print(f"C++ nRMS range: [{nRMS.min():.6e}, {nRMS.max():.6e}], mean={nRMS.mean():.6e}")
	print(f"PY  nRMS range: [{nRMS_py.min():.6e}, {nRMS_py.max():.6e}], mean={nRMS_py.mean():.6e}")