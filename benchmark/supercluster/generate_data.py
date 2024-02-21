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

from pycwb.types.network import Network
from pycwb.modules.cwb_conversions import convert_fragment_clusters_to_netcluster, convert_sparse_series_to_sseries, \
    convert_netcluster_to_fragment_clusters
from pycwb.modules.sparse_series import sparse_table_from_fragment_clusters
from pycwb.modules.multi_resolution_wdm import create_wdm_for_level
from pycwb.types.network_cluster import FragmentCluster
import copy


tf_maps = strains
network = Network(config, strains, nRMS)

sparse_table_list = sparse_table_from_fragment_clusters(config, tf_maps, fragment_clusters)

skyres = config.MIN_SKYRES_HEALPIX if config.healpix > config.MIN_SKYRES_HEALPIX else 0

if skyres > 0:
    network.update_sky_map(config, skyres)
    network.net.setAntenna()
    network.net.setDelay(config.refIFO)
    network.update_sky_mask(config, skyres)

hot = []
for n in range(config.nIFO):
    hot.append(network.get_ifo(n).getHoT())

# set low-rate TD filters
for level in config.WDM_level:
    wdm = create_wdm_for_level(config, level)
    wdm.set_td_filter(config.TDSize, 1)
    # add wavelets to network
    network.add_wavelet(wdm)

# merge cluster
cluster = copy.deepcopy(fragment_clusters[0])
if len(fragment_clusters) > 1:
    for fragment_cluster in fragment_clusters[1:]:
        cluster.clusters += fragment_cluster.clusters

pwc_list = []

###############################
# cWB2G supercluster
###############################
# convert to netcluster
import time

start_time = time.time()
cluster = convert_fragment_clusters_to_netcluster(cluster)

for n in range(config.nIFO):
    det = network.get_ifo(n)
    det.sclear()
    for sparse_table in sparse_table_list:
        det.vSS.push_back(convert_sparse_series_to_sseries(sparse_table[n]))

j = 0

if config.l_high == config.l_low:
    cluster.pair = False
if network.pattern != 0:
    cluster.pair = False

cluster.supercluster('L',network.net.e2or,config.TFgap,False)

print(f"Time taken for supercluster stage1: {time.time() - start_time}")
fragment_cluster_test1 = convert_netcluster_to_fragment_clusters(cluster)


test_data = {
    'strains': strains,
    'nRMS': nRMS,
    'config': config,
    'fragment_clusters': fragment_clusters,
    'fragment_cluster_stage1': fragment_cluster_test1
}

import pickle
with open('test_data.pkl', 'wb') as f:
    pickle.dump(test_data, f)


