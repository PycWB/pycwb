import sys
import time

import ROOT
sys.path.insert(0, "../..")

from pycwb.modules.cwb_coherence.coherence import cluster_pixels, compute_threshold, select_network_pixels
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

from pycwb.modules.data_conditioning.data_conditioning_python import data_conditioning
from pycwb.modules.data_conditioning import data_conditioning as data_conditioning_cwb


data = [check_and_resample(data[i], config, i) for i in range(len(job_segments[0].ifos))]

# strains, nRMS = data_conditioning(config, data)
strains_cwb, nRMS_cwb = data_conditioning_cwb(config, data)


# single level cwb coherence
from pycwb.modules.coherence import coherence_single_res
from wdm_wavelet.wdm import WDM as WDMWavelet
from pycwb.types.time_frequency_series import TimeFrequencyMap
from pycwb.types.time_series import TimeSeries

# use the same strains and nRMS for both coherence calculations to ensure a fair comparison
strains = [TimeSeries.from_input(strain.data) for strain in strains_cwb]

for i in range(config.nRES):
    up_n = int(config.rateANA / 1024)
    if up_n < 1:
        up_n = 1

    # print level infos
    level = config.l_high - i
    layers = 2 ** level if level > 0 else 0
    rate = config.rateANA // 2 ** level

    logger_info = "level : %d\t rate(hz) : %d\t layers : %d\t df(hz) : %f\t dt(ms) : %f \n" % (
            level, rate, layers, config.rateANA / 2. / (2 ** level), 1000. / rate)

    ##########################
    # cWB2G coherence
    ##########################
    print("======== cWB2G coherence =======")
    # at run time output
    from time import perf_counter
    from pycwb.types.network import Network
    from pycwb.modules.cwb_conversions import convert_to_wavearray, convert_netcluster_to_fragment_clusters, WSeries_to_matrix
    from pycwb.modules.multi_resolution_wdm import create_wdm_for_level

    timer_start = perf_counter()
    wdm = create_wdm_for_level(config, config.WDM_level[i])

    net = Network(config, strains_cwb, nRMS_cwb, silent=True)
    # produce TF maps with max over the sky energy
    alp_sum_cwb = 0.0

    # FIXME: max time delay is different to pycbc
    config.max_delay = net.get_max_delay()

    print("Network %d lags" % net.nLag)

    for n in range(len(config.ifo)):
        ts = convert_to_wavearray(strains_cwb[n])
        ts.Edge = config.segEdge
        # TODO: WSeries.putLayer is updated internally, here requires the wave packet pattern
        # https://gwburst.gitlab.io/documentation/latest/html/running.html#wave-packet-parameters
        # The max function is not just calculate the max values, but also set the whole TF map to
        # the max value over delayed time series, this is the most time consuming part in coherence
        alp_sum_cwb += net.get_ifo(n).getTFmap().maxEnergy(ts, wdm.wavelet, config.max_delay, up_n, net.pattern)
        net.get_ifo(n).getTFmap().setlow(config.fLow)
        net.get_ifo(n).getTFmap().sethigh(config.fHigh)

    alp_mean_cwb = alp_sum_cwb / config.nIFO

    if net.pattern != 0:
        Eo_cWB = net.threshold(config.bpp, alp_mean_cwb)
        Eo_cWB_unscaled_shape = net.threshold(config.bpp, alp_sum_cwb)
    else:
        Eo_cWB = net.threshold(config.bpp)
        Eo_cWB_unscaled_shape = Eo_cWB

    # temporary storage for sparse table
    wc = ROOT.netcluster()

    # loop over time lags
    for j in range(int(net.nLag)):
        # select pixels above Eo
        net.get_network_pixels(j, Eo_cWB)
        # get pixel list
        pwc = net.get_cluster(j)
        if net.pattern != 0:
            # cluster pixels
            net.cluster(j, 2, 3)
            wc.cpf(pwc, False)
            # remove pixels below subrho
            wc.select("subrho", config.select_subrho)
            # remove pixels below subnet
            wc.select("subnet", config.select_subnet)
            # copy selected pixels back to pwc
            pwc.cpf(wc, False)
        else:
            net.cluster(j, 1, 1)

        fragment_cluster_cwb = convert_netcluster_to_fragment_clusters(pwc)

    timer_end = perf_counter()
    tf_map_0 = WSeries_to_matrix(net.get_ifo(0).getTFmap())
    print("cWB2G coherence calculation time: %.2f seconds\n" % (timer_end - timer_start))

    ##############################
    # pycwb native coherence
    ##############################

    print("======== pycwb native coherence =======")
    from pycwb.modules.cwb_coherence.lag_plan import build_lag_plan_from_config
    from pycwb.modules.cwb_coherence.coherence import max_energy

    timer_start = perf_counter()
    normalized_strains = [TimeSeries.from_input(strain) for strain in strains]

    wdm_layers = max(1, layers)
    wdm_wavelet = WDMWavelet(
        M=wdm_layers,
        K=wdm_layers,
        beta_order=config.WDM_beta_order,
        precision=config.WDM_precision,
    )

    tf_maps = [
        TimeFrequencyMap.from_timeseries(
            ts=strain,
            wavelet=wdm_wavelet,
            is_whitened=True,
            f_low=getattr(config, "fLow", None),
            f_high=getattr(config, "fHigh", None),
            edge=getattr(config, "segEdge", None),
        )
        for strain in normalized_strains
    ]

    alp_sum_py = 0.0

    max_delay = config.max_delay
    pattern = config.pattern
    lag_plan = build_lag_plan_from_config(config, tf_maps)
    n_lag = lag_plan.n_lag
    print("lag plan built with %d lags" % n_lag)

    for n, tf_map in enumerate(tf_maps):
        alp_sum_py += max_energy(
            tf_map=tf_map,
            max_delay=max_delay,
            up_n=up_n,
            pattern=pattern,
            f_low=config.fLow,
            f_high=config.fHigh,
        )

    alp = alp_sum_py / config.nIFO

    # set threshold
    # threshold is calculated based on the data layers and rate of the default ifo data
    Eo = compute_threshold(
        config.bpp,
        alp if pattern != 0 else None,
        tf_maps=tf_maps,
        edge=config.segEdge,
    )

    for j in range(n_lag):
        # select pixels above Eo
        candidates = select_network_pixels(
            lag_index=j,
            energy_threshold=Eo,
            tf_maps=tf_maps,
            lag_shifts=lag_plan.lag_shifts[j],
            veto=None,
            edge=config.segEdge,
        )
        # get pixel list
        if pattern != 0:
            c = cluster_pixels(min_size=2, max_size=3, pixel_candidates=candidates)
            # remove pixels below subrho
            c.select("subrho", config.select_subrho)
            # remove pixels below subnet
            c.select("subnet", config.select_subnet)
        else:
            c = cluster_pixels(min_size=1, max_size=1, pixel_candidates=candidates)

        fragment_cluster = c

    timer_end = perf_counter()
    print("pycwb native coherence calculation time: %.2f seconds\n" % (timer_end - timer_start))

    print("======== compare =======")

    # Compare the two coherence results
    print("cWB2G max energy: %g" % np.max(tf_map_0))
    print("pycwb native max energy: %g" % np.max(tf_maps[0].data))
    print("cWB2G shape: %s" % str(tf_map_0.shape))
    print("pycwb native shape: %s" % str(tf_maps[0].data.shape))
    # difference in max energy should be small, but can be different due to different wavelet implementations and numerical precision
    # if shape is the same
    # get non zero indexes in the difference map
    tf_map_diff = tf_map_0 - tf_maps[0].data

    non_zero_indexes = np.nonzero(tf_map_diff)
    print("Number of non-zero pixels in the difference map: %d" % len(non_zero_indexes[0]))
    # percentage of non-zero pixels in the difference map
    percent_non_zero = len(non_zero_indexes[0]) / tf_map_diff.size * 100
    print("Percentage of non-zero pixels in the difference map: %.2f%%" % percent_non_zero)
    # number of pixels above a certain threshold in the difference map
    threshold_diff = 0.01 * max(np.max(tf_map_0), np.max(tf_maps[0].data))
    num_pixels_above_threshold_diff = np.sum(np.abs(tf_map_diff) >= threshold_diff)
    print("Number of pixels with difference above threshold: %d" % num_pixels_above_threshold_diff)
    print("Percentage of pixels with difference above threshold: %.2f%%" % (num_pixels_above_threshold_diff / tf_map_diff.size * 100))

    # Eo difference
    print('--------------------------------')
    print("cWB2G shape alp(sum): %g" % alp_sum_cwb)
    print("cWB2G shape alp(mean): %g" % alp_mean_cwb)
    print("pycwb native shape alp(sum): %g" % alp_sum_py)
    print("pycwb native shape alp(mean): %g" % alp)
    print("cWB2G threshold Eo (sum-shape): %g" % Eo_cWB_unscaled_shape)
    print("cWB2G threshold Eo: %g" % Eo_cWB)
    print("pycwb native threshold Eo: %g" % Eo)
    print("Difference in threshold Eo: %g" % abs(Eo_cWB - Eo))
    print("Percentage difference in threshold Eo: %.2f%%" % (abs(Eo_cWB - Eo) / max(Eo_cWB, Eo) * 100))

    print("---------------------------------")
    print("cWB2G: %3d |%9d |%7d \n" % (j, fragment_cluster_cwb.event_count(), fragment_cluster_cwb.pixel_count()))
    print("pycwb: %3d |%9d |%7d \n" % (j, fragment_cluster.event_count(), fragment_cluster.pixel_count()))
    print("Difference in event count: %d" % abs(fragment_cluster_cwb.event_count() - fragment_cluster.event_count()))
    print("Difference in pixel count: %d" % abs(fragment_cluster_cwb.pixel_count() - fragment_cluster.pixel_count()))
