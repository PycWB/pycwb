import os
import time

from pycwb.modules.super_cluster.super_cluster import supercluster_wrapper
from pycwb.modules.xtalk.monster import load_catalog

from pycwb.config import Config
from pycwb.modules.likelihoodWP.likelihood import load_data_from_ifo
from pycwb.modules.logger import logger_init

if not os.environ.get('HOME_WAT_FILTERS'):
    print('Please set HOME_WAT_FILTERS to the directory of WAT filters')
    exit(1)

logger_init()

config = Config('/Users/yumengxu/GWOSC/catalog/GWTC-1-confident/GW150914/pycWB/user_parameters_injection.yaml')
os.chdir('/Users/yumengxu/GWOSC/catalog/GWTC-1-confident/GW150914/pycWB')
#%%
# load xtalk

xtalk_coeff, xtalk_lookup_table, layers, nRes = load_catalog(config.MRAcatalog)


#%% md
## generate injected data for each detector with given parameters in config
#%%
from pycwb.modules.read_data import generate_injection, read_from_job_segment
from pycwb.modules.job_segment import create_job_segment_from_injection, create_job_segment_from_config

job_segments = create_job_segment_from_config(config)

job_seg = job_segments[0]
if job_seg.frames:
    data = read_from_job_segment(config, job_seg)
if job_seg.injections:
    data = generate_injection(config, job_seg, data)

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

#%%
from pycwb.types.network import Network
network = Network(config, strains, nRMS)
#Test
super_fragment_clusters = supercluster_wrapper(config, network, fragment_clusters, strains,
                                               xtalk_coeff, xtalk_lookup_table, layers)
from pycwb.modules.likelihood import likelihood
events, clusters, skymap_statistics = likelihood(config, network, [super_fragment_clusters])

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

# pwc_list = []
# Load tdamp and convert to fragment cluster for testing
start_time_1 = time.time()
net_cluster = convert_fragment_clusters_to_netcluster(cluster)

for n in range(config.nIFO):
    det = network.get_ifo(n)
    det.sclear()
    for sparse_table in sparse_table_list:
        det.vSS.push_back(convert_sparse_series_to_sseries(sparse_table[n]))

pwc = network.get_cluster(0)
pwc.cpf(net_cluster, False)

if config.subacor > 0:
    network.net.acor = config.subacor
if config.subrho > 0:
    network.net.netRHO = config.subrho

network.set_delay_index(hot[0].rate())
pwc.setcore(False)

pwc.loadTDampSSE(network.net, 'a', config.BATCH, config.LOUD)

fragment_clusters = convert_netcluster_to_fragment_clusters(pwc)
print(f"Time taken for convert_fragment_clusters_to_netcluster: {time.time() - start_time_1}")


acor = network.net.acor
network_energy_threshold = 2 * acor * acor * config.nIFO
n_sky = network.net.index.size()
subnet = config.subnet
subcut = config.subcut
subnorm = config.subnorm
subrho = config.subrho
netrho = network.net.netRHO
MRAcatalog = config.MRAcatalog
ml, FP, FX = load_data_from_ifo(network, config.nIFO)

test_data = {
    'strains': strains,
    'nRMS': nRMS,
    'e2or': network.net.e2or,
    'gap': config.gap,
    'Tgap': config.Tgap,
    'Fgap': config.Fgap,
    'n_ifo': config.nIFO,
    'n_sky': n_sky,
    'ml': ml,
    'FP': FP,
    'FX': FX,
    'acor': acor,
    'subnet': subnet,
    'subcut': subcut,
    'subnorm': subnorm,
    'subrho': subrho,
    'netrho': netrho,
    'n_loudest': config.LOUD,
    'xtalk_coeff': xtalk_coeff,
    'xtalk_lookup_table': xtalk_lookup_table,
    'layers': layers,
    'nRes': nRes,
    'fragment_clusters': fragment_clusters,
    'network': network,
    'config': config
}

import pickle
with open('test_data_1.pkl', 'wb') as f:
    pickle.dump(test_data, f)

print("Data generated")