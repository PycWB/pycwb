import sys
import time

sys.path.insert(0, "../..")

from pycwb.modules.read_data.data_check import check_and_resample
from pycwb.config import Config
from pycwb.modules.logger import logger_init
import numpy as np

logger_init()

config = Config()
config.load_from_yaml('./user_parameters_injection.yaml')
config.nproc = 1


from pycwb.modules.read_data import generate_injection, generate_noise_for_job_seg
from pycwb.modules.job_segment import create_job_segment_from_config

job_segments = create_job_segment_from_config(config)

data = generate_noise_for_job_seg(job_segments[0], config.inRate, f_low=config.fLow)
data = generate_injection(config, job_segments[0], data)

data = [check_and_resample(data[i], config, i) for i in range(len(job_segments[0].ifos))]


from pycwb.modules.data_conditioning.regression import regression
from pycwb.modules.data_conditioning.data_conditioning_python import regression_python


data_regressions = [regression(config, h) for h in data]
data_regressions_python = [regression_python(config, h) for h in data]

relative_diff = np.abs(data_regressions[0].data - data_regressions_python[0]) / np.max((np.abs(data_regressions[0].data)))
print(f"Max relative difference: {(relative_diff.max()):.6e}")