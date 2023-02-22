# load user parameters
from pycwb.config import Config, CWBConfig
from pycwb import logger_init
from pycwb.modules.read_data import read_from_config
from pycwb.utils import convert_pycbc_timeseries_to_wavearray
from pycwb.modules.data_conditioning import regression, whitening
from pycwb.modules.coherence import create_network
from pycwb.modules.coherence import coherence
from pycwb.modules.super_cluster import supercluster

cwb_config = CWBConfig('./config.ini')
cwb_config.export_to_envs()
logger_init()

config = Config('./user_parameters_mdc.yaml')

gps_end_time = 1264131194.580

data = read_from_config(config)

wavearray = [convert_pycbc_timeseries_to_wavearray(d) for d in data]

# data conditioning
data_reg = [regression(config, wavearray[i]) for i in range(len(config.ifo))]

data_w_reg = [whitening(config, data_reg[i]) for i in range(len(config.ifo))]
tf_map = [d['TFmap'] for d in data_w_reg]

# initialize network
net, wdm_list = create_network(1, config, data_w_reg)

sparse_table_list, cluster_list = coherence(config, net, tf_map, wdm_list)

supercluster(config, net, wdm_list, cluster_list, sparse_table_list)
