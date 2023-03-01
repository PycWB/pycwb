import os
from pycwb import logger_init
from pycwb.config import Config, CWBConfig
from pycwb.modules.read_data import read_from_gwf, generate_noise, read_from_config
from pycwb.modules.data_conditioning import data_conditioning
from pycwb.modules.coherence import create_network
from pycwb.modules.coherence import coherence
from pycwb.modules.super_cluster import supercluster
from pycwb.modules.likelihood import likelihood
from pycwb.modules.netevent import Event

def cwb_2g(config='./config.ini', user_parameters='./user_parameters.yaml', start_time=None, end_time=None):
    logger_init()

    # load user parameters
    cwb_config = CWBConfig(config)
    cwb_config.export_to_envs()
    config = Config(user_parameters)

    data = read_from_config(config)

    if start_time is None:
        dc_data = data
    else:
        dc_data = [i.crop(start_time - float(i.start_time), float(i.end_time) - end_time) for i in data]

    # data conditioning
    tf_maps, nRMS_list = data_conditioning(config, dc_data)

    # initialize network
    net, wdm_list = create_network(1, config, tf_maps, nRMS_list)

    # calculate coherence
    sparse_table_list, cluster_list = coherence(config, net, tf_maps, wdm_list)

    # supercluster
    cluster, pwc_list = supercluster(config, net, wdm_list, cluster_list, sparse_table_list)

    # likelihood
    events = likelihood(config, net, sparse_table_list, pwc_list, cluster, wdm_list)

    # save events to pickle
    import pickle
    with open('events.pkl', 'wb') as f:
        pickle.dump(events, f)


def generate_injected(config):
    # load noise
    start_time = 931158100
    noise = [generate_noise(f_low=30.0, sample_rate=1024.0, duration=600, start_time=start_time, seed=i)
             for i, ifo in enumerate(config.ifo)]

    # generate injection from pycbc
    from pycbc.waveform import get_td_waveform
    hp, hc = get_td_waveform(approximant="IMRPhenomPv3",
                             mass1=20,
                             mass2=20,
                             spin1z=0.9,
                             spin2z=0.4,
                             inclination=1.23,
                             coa_phase=2.45,
                             distance=500,
                             delta_t=1.0 / noise[0].sample_rate,
                             f_lower=20)
    declination = 0.65
    right_ascension = 4.67
    polarization = 2.34
    gps_end_time = 931158400
    from pycwb.modules.read_data import project_to_detector
    strain = project_to_detector(hp, hc, right_ascension, declination, polarization, config.ifo, gps_end_time)

    # inject signal into noise and convert to wavearray
    injected = [noise[i].add_into(strain[i]) for i in range(len(config.ifo))]

    return injected
