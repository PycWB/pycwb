import os

from pycwb.config import Config
from pycwb.modules.logger import logger_init

if not os.environ.get('HOME_WAT_FILTERS'):
    print('Please set HOME_WAT_FILTERS to the directory of WAT filters')
    exit(1)

logger_init()

config = Config('./user_parameters_injection.yaml')

#%% md
## generate injected data for each detector with given parameters in config
#%%
from pycwb.modules.read_data import generate_injection
from pycwb.modules.job_segment import create_job_segment_from_injection

job_segments = create_job_segment_from_injection(config.ifo, config.simulation, config.injection)

data = generate_injection(config, job_segments[0])

#%% md
## apply data conditioning to the data
#%%
from pycwb.modules.data_conditioning import data_conditioning

strains, nRMS = data_conditioning(config, data)

#%% md
## calculate coherence
#%%
from pycwb.modules.coherence import coherence

# calculate coherence
fragment_clusters = coherence(config, strains, nRMS)

#%% md
## supercluster

test_data = {
    'strains': strains,
    'nRMS': nRMS,
    'config': config,
    'fragment_clusters': fragment_clusters
}

import pickle
with open('test_data.pkl', 'wb') as f:
    pickle.dump(test_data, f)