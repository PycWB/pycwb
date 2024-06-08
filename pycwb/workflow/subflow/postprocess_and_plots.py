import os
from typing import Dict, List
import math
from pycbc.types.timeseries import TimeSeries

from pycwb.config import Config
from pycwb.modules.plot.cluster_statistics import plot_statistics
from pycwb.modules.plot_map.world_map import plot_skymap_contour
from pycwb.modules.read_data import generate_strain_from_injection
from pycwb.modules.reconstruction import get_network_MRA_wave
from pycwb.types.job import WaveSegment
from pycwb.types.network_cluster import Cluster
from pycwb.types.network_event import Event


def reconstruct_waveforms_flow(trigger_folder: str, config: Config, ifos: List[str],
                          event: Event, cluster: Cluster,
                          save: bool = True, plot: bool = False,
                          save_injection: bool = True, plot_injection: bool = False) -> Dict[str, TimeSeries]:
    print(f"Reconstructing waveform for event {event.hash_id}")
    reconstructed_waves = get_network_MRA_wave(config, cluster, config.rateANA, config.nIFO, config.TDRate,
                                               'signal', 0, True, whiten=False)

    print(f"Reconstructing whitened waveform for event {event.hash_id}")
    reconstructed_waves_whiten = get_network_MRA_wave(config, cluster, config.rateANA, config.nIFO, config.TDRate,
                                                      'signal', 0, True, whiten=True)

    # reconstructed_waves_whiten_00 = get_network_MRA_wave(config, cluster, config.rateANA, config.nIFO, config.TDRate,
    #                                                      'signal', -1, True, whiten=True)
    # reconstructed_waves_whiten_90 = get_network_MRA_wave(config, cluster, config.rateANA, config.nIFO, config.TDRate,
    #                                                      'signal', 1, True, whiten=True)

    # rescale
    for ts in reconstructed_waves:
        rescale = 1. / math.pow(math.sqrt(2), math.log2(config.inRate / ts.sample_rate))
        ts.data *= rescale

    for ts in reconstructed_waves_whiten:
        rescale = 1. / math.pow(math.sqrt(2), math.log2(config.inRate / ts.sample_rate))
        ts.data *= rescale

    if save:
        if not os.path.exists(trigger_folder):
            os.makedirs(trigger_folder)
            print(f"Creating trigger folder: {trigger_folder}")

        for i, ts in enumerate(reconstructed_waves):
            print(f"Saving reconstructed waveform for {ifos[i]}")
            ts.save(f"{trigger_folder}/reconstructed_waveform_{ifos[i]}.txt")

        for i, ts in enumerate(reconstructed_waves_whiten):
            print(f"Saving reconstructed waveform for {ifos[i]} (whitened)")
            ts.save(f"{trigger_folder}/reconstructed_waveform_{ifos[i]}_whitened.txt")

        # for i, (hp, hc) in enumerate(zip(reconstructed_waves_whiten_00, reconstructed_waves_whiten_90)):
        #     # save strain = hp + 1j hc
        #     print(f"Saving reconstructed strain for {ifos[i]} (whitened)")
        #     hp = hp - 1j * hc
        #     hp.save(f"{trigger_folder}/reconstructed_strain_{ifos[i]}_whitened.txt")

    if plot:
        from matplotlib import pyplot as plt

        for j, reconstructed_wave in enumerate(reconstructed_waves):
            plt.plot(reconstructed_wave.sample_times, reconstructed_wave.data)
            plt.xlim((event.left[0], event.left[0] + event.stop[0] - event.start[0]))
            plt.savefig(f'{trigger_folder}/reconstructed_wave_ifo_{ifos[j]}.png')
            plt.close()

        for j, reconstructed_wave in enumerate(reconstructed_waves_whiten):
            plt.plot(reconstructed_wave.sample_times, reconstructed_wave.data)
            plt.xlim((event.left[0], event.left[0] + event.stop[0] - event.start[0]))
            plt.savefig(f'{trigger_folder}/reconstructed_wave_whiten_ifo_{ifos[j]}.png')
            plt.close()


    if save_injection or plot_injection:
        if event.injection:
            strains = generate_strain_from_injection(event.injection, config, config.inRate, ifos)

            if save_injection:
                for i, strain in enumerate(strains):
                    print(f"Saving injected strain for {ifos[i]}")
                    strain.save(f"{trigger_folder}/injected_strain_{ifos[i]}.txt")

            if plot_injection:
                from matplotlib import pyplot as plt
                for i, strain in enumerate(strains):
                    plt.plot(strain.sample_times, strain.data)
                    plt.savefig(f'{trigger_folder}/injected_strain_{ifos[i]}.png')
                    plt.close()

    data = {}
    for i, ifo in enumerate(ifos):
        data[f"{ifo}_reconstructed_waves"] = reconstructed_waves[i]
        data[f"{ifo}_reconstructed_waves_whiten"] = reconstructed_waves_whiten[i]

    return data


def plot_trigger_flow(trigger_folder: str,
                 event: Event, cluster: Cluster) -> None:
    print(f"Making plots for event {event.hash_id}")

    # plot the likelihood map
    plot_statistics(cluster, 'likelihood', gps_shift=event.gps[0], filename=f'{trigger_folder}/likelihood_map.png')
    plot_statistics(cluster, 'null', gps_shift=event.gps[0], filename=f'{trigger_folder}/null_map.png')


def plot_skymap_flow(trigger_folder: str,
                 event: Event, event_skymap_statistics: Dict[str, List[float]]) -> None:
    print(f"Making skymap plots for event {event.hash_id}")
    # plot_world_map(event.phi[0], event.theta[0], filename=f'{config.outputDir}/world_map_{job_id}_{i+1}.png')
    for key in event_skymap_statistics.keys():
        plot_skymap_contour(event_skymap_statistics,
                            key=key,
                            reconstructed_loc=(event.phi[0], event.theta[0]),
                            detector_loc=(event.phi[3], event.theta[3]),
                            resolution=1,
                            filename=f'{trigger_folder}/{key}.png')
