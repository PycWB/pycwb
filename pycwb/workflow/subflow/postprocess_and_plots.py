import os
import logging
from typing import Dict, List
import math
import numpy as np
import pycwb
from pycbc.types.timeseries import TimeSeries
from pycwb.modules.plot import event
from pycwb.modules.reconstruction.getResiduals import get_ASD
from pycwb.types.time_frequency_series import TimeFrequencySeries
from pycwb.config import Config
from pycwb.modules.plot.cluster_statistics import plot_statistics
from pycwb.modules.plot_map.world_map import plot_skymap_contour
# from pycwb.modules.read_data import generate_strain_from_injection
from pycwb.modules.reconstruction import get_network_MRA_wave, get_INJ_waveform, get_residuals
from pycwb.types.network_cluster import Cluster
from pycwb.types.network_event import Event
from filelock import SoftFileLock
import h5py as h5

logger = logging.getLogger(__name__)

def reconstruct_waveforms_flow(trigger_folder: str, config: Config, ifos: List[str],
                          event: Event, cluster: Cluster, epoch: float = 0.,
                          wave_file: str = '', save: bool = True, plot: bool = False) -> Dict[str, TimeSeries]:

    # vREC: reconstructed signal
    logger.info(f"Reconstructing waveform for event {event.hash_id}")
    reconstructed_signals = get_network_MRA_wave(config, cluster, config.rateANA, config.nIFO, config.TDRate,
                                               'signal', 0, True, whiten=False)
    
    # whitened vREC: whitened reconstructed signal
    logger.info(f"Reconstructing whitened waveform for event {event.hash_id}")
    reconstructed_signals_whiten = get_network_MRA_wave(config, cluster, config.rateANA, config.nIFO, config.TDRate,
                                                      'signal', 0, True, whiten=True)

    # vDAT: reconstructed data (signal + noise)
    logger.info(f"Reconstructing strain for event {event.hash_id}")
    reconstructed_data  = get_network_MRA_wave(config, cluster, config.rateANA, config.nIFO, config.TDRate,
                                                'strain', 0, True, whiten=False)
    
    # whitened_vDAT: whitened reconstructed data (signal+noise)
    logger.info(f"Reconstructing whitened strain for event {event.hash_id}")
    reconstructed_data_whiten = get_network_MRA_wave(config, cluster, config.rateANA, config.nIFO, config.TDRate,
                                                      'strain', 0, True, whiten=True)

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

    data = {}
    for i, ifo in enumerate(ifos):
        data[f"{ifo}_wf_REC"] = reconstructed_signals[i]
        data[f"{ifo}_wf_REC_whiten"] = reconstructed_signals_whiten[i]
        data[f"{ifo}_wf_DAT"] = reconstructed_data[i]
        data[f"{ifo}_wf_DAT_whiten"] = reconstructed_data_whiten[i]
        data[f"{ifo}_wf_NUL"] = reconstructed_nulls[i]
        data[f"{ifo}_wf_NUL_whiten"] = reconstructed_nulls_whiten[i]
    
    if save:
        # rescaling factor to preserve amplitude
        rescale = 1. / np.sqrt(2) ** (np.log2(config.inRate / (config.inRate >> config.levelR)))

        logger.info(f"Save reconstructed waveforms of event {event.hash_id} to {wave_file}")     
        rescaled_data = {}
        for key, value in data.items():
            rescaled_data[key] = value * rescale
            
        add_wf_to_wave(config, wave_file, event.hash_id, rescaled_data)
        
    if plot:
        from pycwb.modules.plot.waveform import plot
        from matplotlib import pyplot as plt
        # rescaling factor to preserve amplitude
        rescale = 1. / np.sqrt(2) ** (np.log2(config.inRate / (config.inRate >> config.levelR)))

        for j, wave in enumerate(reconstructed_signals):            
            plot(wave*rescale, ifo = ifos[j])
            plt.savefig(f'{trigger_folder}/{ifos[j]}_wf_REC.png')
            plt.close()

        for j, wave in enumerate(reconstructed_signals_whiten):
            plot(wave*rescale, ifo = ifos[j])
            plt.savefig(f'{trigger_folder}/{ifos[j]}_wf_REC_whiten.png')
            plt.close()

        for j, wave in enumerate(reconstructed_data):
            plot(wave*rescale, ifo = ifos[j])
            plt.savefig(f'{trigger_folder}/{ifos[j]}_wf_DAT.png')
            plt.close()
        
        for j, wave in enumerate(reconstructed_data_whiten):
            plot(wave*rescale, ifo = ifos[j])
            plt.savefig(f'{trigger_folder}/{ifos[j]}_wf_DAT_whiten.png')
            plt.close()

        for j, wave in enumerate(reconstructed_nulls):
            plot(wave*rescale, ifo = ifos[j])
            plt.savefig(f'{trigger_folder}/{ifos[j]}_wf_NUL.png')
            plt.close()
        
        for j, wave in enumerate(reconstructed_nulls_whiten):
            plot(wave*rescale, ifo = ifos[j])
            plt.savefig(f'{trigger_folder}/{ifos[j]}_wf_NUL_whiten.png')
            plt.close()

    return data

def reconstruct_INJwaveforms_flow(trigger_folder: str, config: Config, ifos: list[str], event: Event,
                                HoT_list, mdc_maps, window: float, offset: float, inRate: float,
                                wave_file: str = None, save: bool = True, plot: bool = False) -> Dict[str, TimeSeries]:
    
    logger.info(f"Reconstructing injected waveform for event {event.hash_id}")
    data = [get_INJ_waveform(hot, mdc_map, event.injection['gps_time'], window, offset, inRate) for hot, mdc_map in zip(HoT_list, mdc_maps)]
    
    if save:
        try:
            rescaled_data = {}
            for i, ifo in enumerate(ifos):
                logger.info(f"Saving injected waveform for {ifos[i]} in {wave_file}")
                # rescaling factor to preserve amplitude
                rescale = 1. / np.sqrt(2) ** (np.log2(config.inRate / (config.inRate >> config.levelR)))

                rescaled_data[f'{ifo}_wf_INJ'] = data[i]['injected_strain']  * rescale
                rescaled_data[f'{ifo}_wf_INJ_whiten'] = data[i]['whitened_injected_waveform'] * rescale
                
            logger.info(f"Save injected waveforms of event {event.hash_id} to {wave_file}")
            add_wf_to_wave(config, wave_file, event.hash_id, rescaled_data)

        except Exception as e:
            logger.warning(f"Error saving waveform for {ifo}: {e}")

    if plot:
        from pycwb.modules.plot.waveform import plot
        from matplotlib import pyplot as plt

        if not os.path.exists(trigger_folder):
            os.makedirs(trigger_folder)
            logger.info(f"Creating trigger folder: {trigger_folder}")

        # rescaling factor to preserve amplitude
        rescale = 1. / np.sqrt(2) ** (np.log2(config.inRate / (config.inRate >> config.levelR)))
        
        for i, ifo in enumerate(ifos):
            try:
                plot(data[i]['injected_strain']*rescale, ifo = ifo)
                plt.savefig(f'{trigger_folder}/{ifos[i]}_wf_INJ.png')
                plt.close()

                plot(data[i]['whitened_injected_waveform']*rescale, ifo = ifo)
                plt.savefig(f'{trigger_folder}/{ifos[i]}_wf_INJ_whiten.png')
                plt.close()
            except Exception as e:
                logger.warning(f"Error plotting waveform for {ifo}: {e}")
    
    # output structure {key1: [ifo1, ifo2, ...], key2: [ifo1, ifo2, ...], ...}
    data = {key: [d[key] for d in data] for key in data[0]}
    
    return data

def add_wf_to_wave(config: Config, wave_file: str, event_id: str, waves: dict) -> None:
    """
    Add events to waveform file

    A soft lock is used (default filelock does not work on CIT)

    Parameters
    ----------
    wave_file : str
        wave_file of the waveform
    event_id : str
        event id to be added
    waves : dict
        dictionary of waveforms to be added, with keys as the waveform
        names and values as the waveform data
    Returns
    -------
    None
    """
    
    if wave_file is None:
        wave_file = 'wave.h5'
    wave_file = f'{config.outputDir}/{wave_file}'

    with SoftFileLock(wave_file + ".lock", timeout=10):
        with h5.File(wave_file, 'a') as f:
            # if event_id does not exist in the file, create a new group
            if f'{event_id}' not in f:
                grp = f.create_group(f'{event_id}')
            else:
                grp = f[f'{event_id}']
            
            # create datasets with waveform data
            for key, value in waves.items():
                if key in grp:
                    # delete existing dataset if it exists
                    del grp[key]              
                
                grp[key] = value.data
                grp[key].attrs['sample_rate'] = value.sample_rate
                grp[key].attrs['start_time'] = float(value.start_time)
                logger.info(f"Added waveform {key} for event {event_id} to {wave_file}")

def load_wf_from_wave(wave_file: str, ifo: str, keys: List[str]) -> Dict[str, list]:
    """
    Load waveforms for a given IFO and keys from the consolidated wave HDF5 file.
    Paired reader for add_wf_to_wave.

    Parameters
    ----------
    wave_file : str
        Path to the wave HDF5 file.
    ifo : str
        Interferometer name.
    keys : list[str]
        Waveform keys to load, e.g. ['wf_REC', 'wf_INJ']. Each key is prefixed
        with ``ifo_`` when looking up in the file.

    Returns
    -------
    dict[str, list[TimeSeries]]
        Mapping from each key to a list of TimeSeries (one per event, sorted by event id).
    """
    result: Dict[str, list] = {key: [] for key in keys}
    with h5.File(wave_file, 'r') as f:
        for event_id in sorted(f.keys()):
            for key in keys:
                full_key = f'{ifo}_{key}'
                if full_key in f[event_id]:
                    dataset = f[event_id][full_key]
                    ts = TimeSeries(dataset[:],
                                   delta_t=1.0 / dataset.attrs['sample_rate'],
                                   epoch=dataset.attrs['start_time'])
                    result[key].append(ts)
    return result

def reconstruct_residuals_flow(trigger_folder: str, config: Config, ifos: List[str], event: Event, data: list[TimeSeries], reconst_data: Dict[str, TimeSeries], tf_maps: list[TimeFrequencySeries],
                             nrms: list[TimeFrequencySeries], save: bool = True, save_gwf: bool = False, plot: bool = False) -> Dict[str, TimeSeries]:
    """ Reconstruct residuals from the reconstructed data and event information.
    """
    from numpy import savetxt
    logger.info(f"Reconstructing residuals for event {event.hash_id}") 

    from pycwb.modules.reconstruction.getResiduals import get_residuals, get_ASD 

    # Calculate the residuals and ASD for strain data
    strain_residuals = [get_residuals(data[i], reconst_data[f"{ifos[i]}_reconstructed_signals"], config.inRate,\
                                      full_segment=False, rescale=True) for i in range(len(ifos))]
    
    try: 
        strain_ASD = [get_ASD(strain_residuals[i]) for i in range(len(ifos))]  
    except Exception as e:
        logger.warning(f"Error calculating ASD for {ifo}: {e}") 
        strain_ASD = None 
        
    whitened_residuals = [get_residuals(tf_maps[i].data, reconst_data[f"{ifos[i]}_reconstructed_signals_whiten"],\
                                        config.inRate, full_segment=False, rescale = False) for i in range(len(ifos))]

    if save: 
        if not os.path.exists(trigger_folder):
            os.makedirs(trigger_folder)
            logger.info(f"Creating trigger folder: {trigger_folder}")
        for i, ifo in enumerate(ifos):

            try:
                logger.info(f"Saving residuals for {ifo}")
                strain_residuals[i].save(f"{trigger_folder}/{ifo}_RES.{config.save_waveform_format}")

                logger.info(f"Saving whitened residuals for {ifo}")
                whitened_residuals[i].save(f"{trigger_folder}/{ifo}_RES_whiten.{config.save_waveform_format}")
                
                nrms[i].data.save(f"{trigger_folder}/{ifo}_nRMS.{config.save_waveform_format}")
                logger.info(f"Saving ASD for {ifo}")
                if strain_ASD is not None:
                    savetxt(f"{trigger_folder}/{ifo}_RES_ASD.txt", strain_ASD[i])
            except Exception as e:
                logger.warning(f"Error saving residuals for {ifo}: {e}")  

    
    if save_gwf: 
        
        from pycwb.modules.read_data import save_to_gwf

        logger.info(f"Saving residuals to GWF for event {event.hash_id}")
        strain_residuals_full = [get_residuals(tf_maps[i].data, reconst_data[f"{ifos[i]}_reconstructed_signals_whiten"], config.inRate,\
                                      full_segment=True, rescale=False) for i in range(len(ifos))]

        channel_name = f"RES_WHITEN_{int(strain_residuals_full[0].sample_rate)}Hz"
        label = "RES_WHITEN"
        start_time = float(strain_residuals_full[0].start_time)
        duration = float(strain_residuals_full[0].duration)
        
        save_to_gwf(strain_residuals_full, ifos, channel_name, trigger_folder, start_time, duration, label)
        
        for i, ifo in enumerate(ifos):
            strain_residuals_full[i].save(f"{trigger_folder}/{ifo}_RES_whiten_full.{config.save_waveform_format}")
            with open(f"{trigger_folder}/{ifo}_frames.in", 'w') as f:
                f.write(f"input/frames/{ifo}-{label}-{int(start_time)}-{int(duration)}.gwf\n")
                f.write(f"# channel name is {ifo}:{channel_name}")


    if plot:
        from pycwb.modules.plot.waveform import plot
        from matplotlib import pyplot as plt

        if not os.path.exists(trigger_folder):
            os.makedirs(trigger_folder)
            logger.info(f"Creating trigger folder: {trigger_folder}")

        for i, ifo in enumerate(ifos):
            try:
                plot(strain_residuals[i], ifo = ifo)
                plt.savefig(f'{trigger_folder}/{ifo}_RES.png')
                plt.close()

                plot(whitened_residuals[i], ifo = ifo)
                plt.savefig(f'{trigger_folder}/{ifo}_RES_whiten.png')
                plt.close()

      
            except Exception as e:
                logger.warning(f"Error plotting residuals for {ifo}: {e}")

def plot_trigger_flow(trigger_folder: str,
                 event: Event, cluster: Cluster) -> None:
    if not os.path.exists(trigger_folder):
        os.makedirs(trigger_folder)
        logger.info(f"Creating trigger folder: {trigger_folder}")

    logger.info(f"Making plots for event {event.hash_id}")

    # plot the likelihood map
    plot_statistics(cluster, 'likelihood', gps_shift=event.gps[0], filename=f'{trigger_folder}/likelihood_map.png')
    plot_statistics(cluster, 'null', gps_shift=event.gps[0], filename=f'{trigger_folder}/null_map.png')

def plot_skymap_flow(trigger_folder: str,
                 event: Event, event_skymap_statistics: Dict[str, List[float]]) -> None:
    if not os.path.exists(trigger_folder):
        os.makedirs(trigger_folder)
        logger.info(f"Creating trigger folder: {trigger_folder}")

    logger.info(f"Making skymap plots for event {event.hash_id}")
    # plot_world_map(event.phi[0], event.theta[0], filename=f'{config.outputDir}/world_map_{job_id}_{i+1}.png')
    for key in event_skymap_statistics.keys():
        plot_skymap_contour(event_skymap_statistics,
                            key=key,
                            reconstructed_loc=(event.phi[0], event.theta[0]),
                            detector_loc=(event.phi[3], event.theta[3]),
                            resolution=1,
                            filename=f'{trigger_folder}/{key}.png')
