def cwb_2g():
    # set env LD_LIBRARY_PATH to install_path
    install_path = "/Users/yumengxu/Project/Physics/cwb/cwb_source/tools/install/lib"
    import os
    os.environ['LD_LIBRARY_PATH'] = install_path
    from pycwb import logger_init
    logger_init()

    # load user parameters
    from pycwb.config import Config, CWBConfig
    cwb_config = CWBConfig('./config.ini')
    cwb_config.export_to_envs()
    config = Config('./user_parameters.yaml')

    # load noise from gwf
    from pycwb.modules.read_data import read_from_gwf
    noise = [read_from_gwf(i, config, f"frames/L1H1V1-SimStrain-9311/{ifo}-SimStrain-931158100-600.gwf",
                           config.channelNamesRaw[i], None, None) for i, ifo in enumerate(config.ifo)]

    # generate injection from pycbc
    from pycbc.waveform import get_td_waveform
    hp, hc = get_td_waveform(approximant="IMRPhenomPv3",
                             mass1=20,
                             mass2=20,
                             spin1z=0.9,
                             spin2z=0.4,
                             inclination=1.23,
                             coa_phase=2.45,
                             delta_t=1.0 / noise[0].sample_rate,
                             f_lower=32)
    declination = 0.65
    right_ascension = 4.67
    polarization = 2.34
    gps_end_time = 931158700
    from pycwb.modules.read_data import project_to_detector
    strain = project_to_detector(hp, hc, right_ascension, declination, polarization, config.ifo, gps_end_time)

    # inject signal into noise and convert to wavearray
    injected = [noise[i].add_into(strain[i]) for i in range(len(config.ifo))]

    from pycwb.utils import convert_pycbc_timeseries_to_wavearray
    wavearray = [convert_pycbc_timeseries_to_wavearray(d) for d in injected]

    from pycwb.modules.data_conditioning import regression, whitening
    data_reg = [regression(config, wavearray[i]) for i in range(len(config.ifo))]
    data_w_reg = [whitening(config, data_reg[i]) for i in range(len(config.ifo))]
    tf_map = [d['TFmap'] for d in data_w_reg]

    # initialize network
    from pycwb.modules.coherence import create_network
    net, wdm_list = create_network(1, config, data_w_reg)

    # calculate coherence
    from pycwb.modules.coherence import select_pixels
    sparse_table_list, cluster_list = select_pixels(config, net, tf_map, wdm_list)

    # supercluster
    from pycwb.modules.super_cluster import supercluster
    supercluster(config, net, wdm_list, cluster_list, sparse_table_list)
