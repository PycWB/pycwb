import os
import logging
from typing import Dict, List
import math
from pycbc.types.timeseries import TimeSeries
from pycwb.types.time_frequency_series import TimeFrequencySeries
from pycwb.config import Config
from pycwb.modules.plot.cluster_statistics import plot_statistics
from pycwb.modules.plot_map.world_map import plot_skymap_contour
# from pycwb.modules.read_data import generate_strain_from_injection
from pycwb.modules.reconstruction import get_network_MRA_wave, get_INJ_waveform
from pycwb.types.network_cluster import Cluster
from pycwb.types.network_event import Event


logger = logging.getLogger(__name__)

def reconstruct_waveforms_flow(trigger_folder: str, config: Config, ifos: List[str],
                          event: Event, cluster: Cluster, epoch: float = 0.,
                          save: bool = True, plot: bool = False) -> Dict[str, TimeSeries]:

    # vREC: reconstructed signal
    logger.info(f"Reconstructing waveform for event {event.hash_id}")
    reconstructed_signals = get_network_MRA_wave(config, cluster, config.rateANA, config.nIFO, config.TDRate,
                                               'signal', 0, True, whiten=False, in_rate=config.inRate)
    
    # whitened vREC: whitened reconstructed signal
    logger.info(f"Reconstructing whitened waveform for event {event.hash_id}")
    reconstructed_signals_whiten = get_network_MRA_wave(config, cluster, config.rateANA, config.nIFO, config.TDRate,
                                                      'signal', 0, True, whiten=True, in_rate=config.inRate)

    # vDAT: reconstructed data (signal + noise)
    logger.info(f"Reconstructing strain for event {event.hash_id}")
    reconstructed_data  = get_network_MRA_wave(config, cluster, config.rateANA, config.nIFO, config.TDRate,
                                                'strain', 0, True, whiten=False, in_rate=config.inRate)
    
    # whitened_vDAT: whitened reconstructed data (signal+noise)
    logger.info(f"Reconstructing whitened strain for event {event.hash_id}")
    reconstructed_data_whiten = get_network_MRA_wave(config, cluster, config.rateANA, config.nIFO, config.TDRate,
                                                      'strain', 0, True, whiten=True, in_rate=config.inRate)

    # vNUL: reconstructed noise (vDAT - vREC)
    reconstructed_nulls = [reconstructed_data[i] - reconstructed_signals[i] for i in range(len(ifos))]

    # whitened vNUL: whitened reconstructed noise (whitened_vDAT - whitened_vREC)
    reconstructed_nulls_whiten = [reconstructed_data_whiten[i] - reconstructed_signals_whiten[i] for i in range(len(ifos))]

    # apply epoch
    for w in reconstructed_signals: w.start_time += epoch
    for w in reconstructed_signals_whiten: w.start_time += epoch
    for w in reconstructed_data: w.start_time += epoch
    for w in reconstructed_data_whiten: w.start_time += epoch
    for w in reconstructed_nulls: w.start_time += epoch
    for w in reconstructed_nulls_whiten: w.start_time += epoch

    # reconstructed_signals_whiten_00 = get_network_MRA_wave(config, cluster, config.rateANA, config.nIFO, config.TDRate,
    #                                                      'signal', -1, True, whiten=True)
    # reconstructed_signals_whiten_90 = get_network_MRA_wave(config, cluster, config.rateANA, config.nIFO, config.TDRate,
    #                                                      'signal', 1, True, whiten=True)

    # # rescale
    # for ts in reconstructed_signals:
    #     rescale = 1. / math.pow(math.sqrt(2), math.log2(config.inRate / ts.sample_rate))
    #     ts.data *= rescale

    # for ts in reconstructed_signals_whiten:
    #     rescale = 1. / math.pow(math.sqrt(2), math.log2(config.inRate / ts.sample_rate))
    #     ts.data *= rescale

    if save:
        if not os.path.exists(trigger_folder):
            os.makedirs(trigger_folder)
            logger.info(f"Creating trigger folder: {trigger_folder}")

        for i, ts in enumerate(reconstructed_signals):
            logger.info(f"Saving reconstructed SIGNAL for {ifos[i]}")
            # ts.save(f"{trigger_folder}/reconstructed_waveform_{ifos[i]}.{config.save_waveform_format}")
            ts.save(f"{trigger_folder}/{ifos[i]}_wf_REC.{config.save_waveform_format}")

        for i, ts in enumerate(reconstructed_signals_whiten):
            logger.info(f"Saving reconstructed SIGNAL for {ifos[i]} (whitened)")
            # ts.save(f"{trigger_folder}/reconstructed_waveform_{ifos[i]}_whitened.{config.save_waveform_format}")
            ts.save(f"{trigger_folder}/{ifos[i]}_wf_REC_whiten.{config.save_waveform_format}")

        for i, ts in enumerate(reconstructed_data):
            logger.info(f"Saving reconstructed DATA for {ifos[i]}")
            ts.save(f"{trigger_folder}/{ifos[i]}_wf_DAT.{config.save_waveform_format}")

        for i, ts in enumerate(reconstructed_data_whiten):
            logger.info(f"Saving reconstructed DATA for {ifos[i]} (whitened)")
            ts.save(f"{trigger_folder}/{ifos[i]}_wf_DAT_whiten.{config.save_waveform_format}")

        for i, ts in enumerate(reconstructed_nulls):
            logger.info(f"Saving reconstructed NULL for {ifos[i]}")
            ts.save(f"{trigger_folder}/{ifos[i]}_wf_NUL.{config.save_waveform_format}")

        for i, ts in enumerate(reconstructed_nulls_whiten):
            logger.info(f"Saving reconstructed NULL for {ifos[i]} (whitened)")
            ts.save(f"{trigger_folder}/{ifos[i]}_wf_NUL_whiten.{config.save_waveform_format}")

        # for i, (hp, hc) in enumerate(zip(reconstructed_signals_whiten_00, reconstructed_signals_whiten_90)):
        #     # save strain = hp + 1j hc
        #     logger.info(f"Saving reconstructed strain for {ifos[i]} (whitened)")
        #     hp = hp - 1j * hc
        #     hp.save(f"{trigger_folder}/reconstructed_data_{ifos[i]}_whitened.txt")

    if plot:
        from pycwb.modules.plot.waveform import plot
        from matplotlib import pyplot as plt

        for j, wave in enumerate(reconstructed_signals):
            plot(wave, ifo = ifos[j])
            # plt.plot(wave.sample_times, wave.data)
            # plt.xlim((event.left[0], event.left[0] + event.stop[0] - event.start[0]))
            # plt.savefig(f'{trigger_folder}/reconstructed_signal_ifo_{ifos[j]}.png')
            plt.savefig(f'{trigger_folder}/{ifos[j]}_wf_REC.png')
            plt.close()

        for j, wave in enumerate(reconstructed_signals_whiten):
            plot(wave, ifo = ifos[j])
            # plt.plot(wave.sample_times, wave.data)
            # plt.xlim((event.left[0], event.left[0] + event.stop[0] - event.start[0]))
            # plt.savefig(f'{trigger_folder}/reconstructed_signal_whiten_ifo_{ifos[j]}.png')
            plt.savefig(f'{trigger_folder}/{ifos[j]}_wf_REC_whiten.png')
            plt.close()

        for j, wave in enumerate(reconstructed_data):
            plot(wave, ifo = ifos[j])
            # plt.plot(wave.sample_times, wave.data)
            # plt.xlim((event.left[0], event.left[0] + event.stop[0] - event.start[0]))
            # plt.savefig(f'{trigger_folder}/reconstructed_data_ifo_{ifos[j]}.png')
            plt.savefig(f'{trigger_folder}/{ifos[j]}_wf_DAT.png')
            plt.close()
        
        for j, wave in enumerate(reconstructed_data_whiten):
            plot(wave, ifo = ifos[j])
            # plt.plot(wave.sample_times, wave.data)
            # plt.xlim((event.left[0], event.left[0] + event.stop[0] - event.start[0]))
            # plt.savefig(f'{trigger_folder}/reconstructed_data_whiten_ifo_{ifos[j]}.png')
            plt.savefig(f'{trigger_folder}/{ifos[j]}_wf_DAT_whiten.png')
            plt.close()

        for j, wave in enumerate(reconstructed_nulls):
            plot(wave, ifo = ifos[j])
            # plt.plot(wave.sample_times, wave.data)
            # plt.xlim((event.left[0], event.left[0] + event.stop[0] - event.start[0]))
            # plt.savefig(f'{trigger_folder}/reconstructed_null_ifo_{ifos[j]}.png')
            plt.savefig(f'{trigger_folder}/{ifos[j]}_wf_NUL.png')
            plt.close()
        
        for j, wave in enumerate(reconstructed_nulls_whiten):
            plot(wave, ifo = ifos[j])
            # plt.plot(wave.sample_times, wave.data)
            # plt.xlim((event.left[0], event.left[0] + event.stop[0] - event.start[0]))
            # plt.savefig(f'{trigger_folder}/reconstructed_null_whiten_ifo_{ifos[j]}.png')
            plt.savefig(f'{trigger_folder}/{ifos[j]}_wf_NUL_whiten.png')
            plt.close()

    # if save_injection or plot_injection:
    #     if event.injection:
    #         strains = generate_strain_from_injection(event.injection, config, config.inRate, ifos)

    #         if save_injection:
    #             for i, strain in enumerate(strains):
    #                 logger.info(f"Saving injected strain for {ifos[i]}")
    #                 strain.save(f"{trigger_folder}/injected_strain_{ifos[i]}.{config.save_waveform_format}")

    #         if plot_injection:
    #             from matplotlib import pyplot as plt
    #             for i, strain in enumerate(strains):
    #                 plt.plot(strain.sample_times, strain.data)
    #                 plt.savefig(f'{trigger_folder}/injected_strain_{ifos[i]}.png')
    #                 plt.close()

    data = {}
    for i, ifo in enumerate(ifos):
        data[f"{ifo}_reconstructed_signals"] = reconstructed_signals[i]
        data[f"{ifo}_reconstructed_signals_whiten"] = reconstructed_signals_whiten[i]
        data[f"{ifo}_reconstructed_data"] = reconstructed_data[i]
        data[f"{ifo}_reconstructed_data_whiten"] = reconstructed_data_whiten[i]
        data[f"{ifo}_reconstructed_nulls"] = reconstructed_nulls[i]
        data[f"{ifo}_reconstructed_nulls_whiten"] = reconstructed_nulls_whiten[i]
    
    return data

def reconstruct_INJwaveforms_flow(trigger_folder: str, config: Config, ifos: list[str], event: Event,
                                HoT_list: list[TimeFrequencySeries], mdc_maps: list[TimeFrequencySeries], window: float, offset: float, inRate: float, 
                                save: bool = True, plot: bool = False) -> Dict[str, TimeSeries]:
    # [get_INJ_waveform(hot, mdc_map, np.array([inj['gps_time'] for inj in sub_job_seg.injections]), 10/2, config.segEdge, config.inRate) for hot, mdc_map in zip(HoT_list, mdc_maps)]
    logger.info(f"Reconstructing injected waveform for event {event.hash_id}")
    data = [get_INJ_waveform(hot, mdc_map, event.injection['gps_time'], window, offset, inRate) for hot, mdc_map in zip(HoT_list, mdc_maps)]
    
    if save:
        try:
            for i, ifo in enumerate(ifos):
                logger.info(f"Saving injected waveform for {ifos[i]}")
                data[i]['injected_strain'].save(f"{trigger_folder}/{ifos[i]}_wf_INJ.{config.save_waveform_format}")

                logger.info(f"Saving whitened injected waveform for {ifos[i]}")
                data[i]['whitened_injected_waveform'].save(f"{trigger_folder}/{ifos[i]}_wf_INJ_whiten.{config.save_waveform_format}")
        except Exception as e:
            logger.warning(f"Error saving waveform for {ifo}: {e}")

    if plot:
        from pycwb.modules.plot.waveform import plot
        from matplotlib import pyplot as plt

        for i, ifo in enumerate(ifos):
            try:
                plot(data[i]['injected_strain'], ifo = ifo)
                plt.savefig(f'{trigger_folder}/{ifos[i]}_wf_INJ.png')
                plt.close()

                plot(data[i]['whitened_injected_waveform'], ifo = ifo)
                plt.savefig(f'{trigger_folder}/{ifos[i]}_wf_INJ_whiten.png')
                plt.close()
            except Exception as e:
                logger.warning(f"Error plotting waveform for {ifo}: {e}")
    
    # output structure {key1: [ifo1, ifo2, ...], key2: [ifo1, ifo2, ...], ...}
    data = {key: [d[key] for d in data] for key in data[0]}
    
    return data

def plot_trigger_flow(trigger_folder: str,
                 event: Event, cluster: Cluster) -> None:
    logger.info(f"Making plots for event {event.hash_id}")

    # plot the likelihood map
    plot_statistics(cluster, 'likelihood', gps_shift=event.gps[0], filename=f'{trigger_folder}/likelihood_map.png')
    plot_statistics(cluster, 'null', gps_shift=event.gps[0], filename=f'{trigger_folder}/null_map.png')

def plot_skymap_flow(trigger_folder: str,
                 event: Event, event_skymap_statistics: Dict[str, List[float]]) -> None:
    logger.info(f"Making skymap plots for event {event.hash_id}")
    # plot_world_map(event.phi[0], event.theta[0], filename=f'{config.outputDir}/world_map_{job_id}_{i+1}.png')
    for key in event_skymap_statistics.keys():
        plot_skymap_contour(event_skymap_statistics,
                            key=key,
                            reconstructed_loc=(event.phi[0], event.theta[0]),
                            detector_loc=(event.phi[3], event.theta[3]),
                            resolution=1,
                            filename=f'{trigger_folder}/{key}.png')
