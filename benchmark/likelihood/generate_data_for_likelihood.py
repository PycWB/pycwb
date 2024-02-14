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
#%%
from pycwb.modules.super_cluster import supercluster
from pycwb.types.network import Network

network = Network(config, strains, nRMS)

pwc_list = supercluster(config, network, fragment_clusters, strains)

#%% md
## dump data

#%%
# only dump the first supercluster
d = pwc_list[0]

from pycwb.modules.cwb_conversions import convert_fragment_clusters_to_netcluster, convert_netcluster_to_fragment_clusters

pwc = network.get_cluster(0)
wdm_list = network.get_wdm_list()
for wdm in wdm_list:
    wdm.setTDFilter(config.TDSize, config.upTDF)

# load delay index
network.set_delay_index(config.TDRate)

# load time delay data
pwc.cpf(convert_fragment_clusters_to_netcluster(d.dump_cluster(0)), False)
pwc.setcore(False, 1)
pwc.loadTDampSSE(network.net, 'a', config.BATCH, config.BATCH)


from pycwb.modules.likelihoodWP.likelihood import load_data_from_ifo
import numpy as np

acor = network.net.acor
network_energy_threshold = 2 * acor * acor * config.nIFO
gamma_regulator = network.net.gamma * network.net.gamma * 2 / 3
delta_regulator = abs(network.net.delta) if abs(network.net.delta) < 1 else 1
REG = [delta_regulator * np.sqrt(2), 0, 0]
netEC_threshold = network.net.netRHO * network.net.netRHO * 2

n_sky = network.net.index.size()

ml, FP, FX = load_data_from_ifo(network, config.nIFO)

cluster_test = convert_netcluster_to_fragment_clusters(pwc)
pixels = cluster_test.clusters[0].pixels

# save FP, FX, rms, n_sky, gamma_regulator, network_energy_threshold to pickle
test_data = {
    'FP': FP,
    'FX': FX,
    'pixels': pixels,
    'n_ifo': config.nIFO,
    'ml': ml,
    'n_sky': n_sky,
    'delta_regulator': delta_regulator,
    'gamma_regulator': gamma_regulator,
    'netEC_threshold': netEC_threshold,
    'network_energy_threshold': network_energy_threshold,
    'netCC': network.net.netCC,
}
import pickle
with open('test_data.pkl', 'wb') as f:
    pickle.dump(test_data, f)