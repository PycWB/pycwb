import logging
import os
import psutil
import numpy as np
from copy import copy
from pycwb.config import Config
from pycwb.types.time_series import TimeSeries
from pycwb.modules.catalog import add_events_to_catalog
from pycwb.modules.super_cluster.supercluster import supercluster
from pycwb.modules.xtalk.monster import load_catalog
from pycwb.modules.coherence.coherence import coherence
from pycwb.modules.read_data import generate_strain_from_injection, generate_noise_for_job_seg, read_from_job_segment, check_and_resample
from pycwb.modules.data_conditioning import data_conditioning, whitening_mdc
from pycwb.modules.likelihood import likelihood
from pycwb.modules.qveto.qveto import get_qveto
from pycwb.types.job import WaveSegment
from pycwb.types.network import Network
from pycwb.modules.workflow_utils.job_setup import print_job_info
from pycwb.modules.workflow_utils import create_single_trigger_folder, save_trigger, add_event_to_catalog
from pycwb.workflow.subflow.postprocess_and_plots import plot_trigger_flow, reconstruct_waveforms_flow, reconstruct_INJwaveforms_flow, plot_skymap_flow
from pycwb.modules.reconstruction import estimate_snr

logger = logging.getLogger(__name__)


def process_job_segment(working_dir: str, config: Config, job_seg: WaveSegment, compress_json: bool = True,
                        catalog_file: str = None, queue=None, production_mode: bool = False, skip_lags: list = None):
    """
    The core workflow to process single job segment with trails or lags. 

    Parameters
    ----------
    working_dir : str
        The working directory for the run
    config : Config
        The configuration object
    job_seg : WaveSegment
        The job segment to process
    compress_json : bool
        Whether to compress the json files
    catalog_file : str
        The catalog file to save the triggers
    queue : Queue
        The queue to send the triggers to the collector for saving in production mode
    production_mode : bool
        Whether to run in production mode, if True, the triggers will be sent to the queue instead of saving them in this function
    skip_lags : list
        The options to skip certain lags. It is used for resuming the processing after a crash
    
    """
    print_job_info(job_seg)

    if not job_seg.frames and not job_seg.noise and not job_seg.injections:
        raise ValueError("No data to process")

    base_data = None

    if job_seg.frames:
        base_data = read_from_job_segment(config, job_seg)
    if job_seg.noise:
        base_data = generate_noise_for_job_seg(job_seg, config.inRate, f_low=config.fLow, data=base_data)

    # get all the trial_idx from the injections, if there is no injections, use 0
    trial_idxs = {0}
    if job_seg.injections:
        trial_idxs = set([injection.get('trial_idx', 0) for injection in job_seg.injections])

    wave_file = os.path.basename(catalog_file).replace('catalog', 'wave').replace('.json','.h5') if catalog_file is not None else None
    # loop over all the trial_idx, if there is no injections or only one trial, only loop once for trial_idx=0
    for trial_idx in trial_idxs:
        #creates new "clean" data for each trial_idx to avoid mixing different trial idxs at different cycles 
        data = base_data
        if job_seg.injections:
            # use sub_job_seg for each trial_idx to avoid passing the trial_idx to the following functions. 
            sub_job_seg = copy(job_seg)
            sub_job_seg.injections = [injection for injection in job_seg.injections if injection.get('trial_idx', 0) == trial_idx]
            logger.info(f"Processing trial_idx: {trial_idx} with {len(sub_job_seg.injections)} injections: {sub_job_seg.injections}")
            
            mdc = [TimeSeries(data=np.zeros(int(base_data[i].duration * base_data[i].sample_rate)), t0=base_data[i].start_time, dt=1/base_data[i].sample_rate) for i in range(len(sub_job_seg.ifos))]

            for injection in sub_job_seg.injections:
                inj = generate_strain_from_injection(injection, config, base_data[0].sample_rate, sub_job_seg.ifos) 
                #default argument copy = True prevents original base_data to be modified due to aliasing 
                mdc = [mdc[i].inject(inj[i]) for i in range(len(sub_job_seg.ifos))]
                data = [data[i].inject(inj[i]) for i in range(len(sub_job_seg.ifos))] 
        else:
            logger.info(f"Processing trial_idx: {trial_idx} without injections")
            sub_job_seg = job_seg
        # add trial_idx to the sub_job_seg
        sub_job_seg.trial_idx = trial_idx

        # check and resample the data
        data = [check_and_resample(data[i], config, i) for i in range(len(job_seg.ifos))]
        logger.info("Memory usage: %f.2 MB", psutil.Process().memory_info().rss / 1024 / 1024)

        # data conditioning
        tf_maps, nRMS_list = data_conditioning(config, data)
        logger.info("Memory usage: %f.2 MB", psutil.Process().memory_info().rss / 1024 / 1024)

        # if injection, check and resample the mdc
        if job_seg.injections:
            mdc = [check_and_resample(mdc[i], config, i) for i in range(len(job_seg.ifos))]
            
            # whitening mdc using nRMS of data
            mdc_maps, HoT_list = zip(*[whitening_mdc(config, m, nrms) for m, nrms in zip(mdc, nRMS_list)]) 

        # initialize network object 
        network = Network(config, tf_maps, nRMS_list)

        # coherence
        fragment_clusters = coherence(config, tf_maps, nRMS_list, net=network)
        logger.info("Memory usage: %f.2 MB", psutil.Process().memory_info().rss / 1024 / 1024)

        # supercluster
        super_fragment_clusters = supercluster(config, network, fragment_clusters, tf_maps)

        logger.info("Memory usage: %f.2 MB", psutil.Process().memory_info().rss / 1024 / 1024)

        # compute likelihood for each lag
        for lag, fragment_cluster in enumerate(super_fragment_clusters):
            events, clusters, skymap_statistics = likelihood(config, network, fragment_cluster,
                                                            lag=lag, shifts=sub_job_seg.shift, job_id=sub_job_seg.index)
            
            logger.info("Memory usage: %f.2 MB", psutil.Process().memory_info().rss / 1024 / 1024)

            # only return selected events
            events_data = []
            for i, cluster in enumerate(clusters):
                if cluster.cluster_status != -1:
                    continue
                event = events[i]
                event_skymap_statistics = skymap_statistics[i]
                events_data.append((event, cluster, event_skymap_statistics))

                # associate the injections if there are any
                if sub_job_seg.injections:
                    for injection in sub_job_seg.injections:
                        # FIXME: for overlap signal, this won't work
                        if event.start[0] - 0.1 < injection['gps_time'] < event.stop[0] + 0.1:
                            event.injection = injection
            logger.info("Memory usage: %f.2 MB", psutil.Process().memory_info().rss / 1024 / 1024)

            #################### Save triggers and post-process ####################
            # save triggers
            trigger_folders = []
            for trigger in events_data:
                trigger_folder = create_single_trigger_folder(working_dir, config.trigger_dir, sub_job_seg, trigger)
                trigger_folders.append(trigger_folder)
                
                save_trigger(trigger_folder=trigger_folder, trigger_data=trigger,
                            save_cluster=config.save_cluster, save_sky_map=config.save_sky_map)

            logger.info("Memory usage: %f.2 MB", psutil.Process().memory_info().rss / 1024 / 1024)

            # post process and plot
            for trigger_folder, trigger in zip(trigger_folders, events_data):
                # FIXME: add gps time and segment time on the x ticks
                event, cluster, event_skymap_statistics = trigger
                event.hybrid = True
                # estimate reconstructed_waveforms
                reconst_data = reconstruct_waveforms_flow(trigger_folder, config, sub_job_seg.ifos,
                                        event, cluster, epoch=sub_job_seg.padded_start,
                                        wave_file=wave_file,save=config.save_waveform, plot=config.plot_waveform)
                
                # if injection, estimate injected_waveforms and calculate related statistics
                if event.injection: 
                    injected_data = reconstruct_INJwaveforms_flow(trigger_folder, config, sub_job_seg.ifos, event,
                                                                HoT_list, mdc_maps, config.iwindow/2, config.segEdge, config.inRate,
                                                                wave_file=wave_file, save=config.save_injection, plot=config.plot_injection)

                    # if config.save_injection: 
                    #     event.wf_sINJ   = injected_data['injected_strain']            # estimated injected strain
                    #     event.wf_wINJ   = injected_data['whitened_injected_waveform'] # estimated injected whitened waveform
                    event.hrss      += injected_data['hrss']                          # hrss[nifo+i]: injected hrss in ifo[i]
                    event.time      += injected_data['central_time']                  # time[nifo+i]: estimated injected central_time in ifo[i]
                    event.iSNR      = injected_data['snr']                            # estimated injected snr
                    event.frequency += injected_data['central_freq']                  # frequency[nifo+i]:  estimated injected central frequency in ifo[i]
                    event.bandwidth += injected_data['bandwidth']                     # bandwidth[nifo+i]:  estimated injected bandwidth in ifo[i]
                    event.duration  += injected_data['duration']                      # duration[nifo+i]:   estimated injected duration
                    
                    # snr statistics
                    inj_waveforms = injected_data['whitened_injected_waveform']
                    rec_waveforms = [reconst_data[f'{ifo}_wf_REC_whiten'] for ifo in sub_job_seg.ifos]
                    event.oSNR     = [estimate_snr(rec_waveform) for rec_waveform in rec_waveforms]
                    event.ioSNR    = [estimate_snr(inj_waveform, rec_waveform) if (inj_waveform is not None) and (rec_waveform is not None) else None for inj_waveform, rec_waveform in zip(inj_waveforms, rec_waveforms)]
                    
                # calculate Qveto and Qfactor, add to the event for dumping to the catalog
                try:
                    min_qveto = 1e23
                    min_qfactor = 1e23

                    # find the minimum Qveto and Qfactor for the event in all ifos and reconstructed strain/waves
                    for ifo in sub_job_seg.ifos:
                        # for a_type in ['data', 'signals']:
                        for a_type in ['DAT','REC']:
                            [qveto, qfactor] = get_qveto(reconst_data[f'{ifo}_wf_{a_type}_whiten'])
                            min_qveto = min(min_qveto, qveto)
                            min_qfactor = min(min_qfactor, qfactor)
                    
                    event.Qveto = [min_qveto, min_qfactor]
                    event.qveto = min_qveto
                    event.qfactor = min_qfactor
                    logger.info(f"Qveto for event {event.hash_id}: {event.qveto}, Qfactor: {event.qfactor}")
                except Exception as e:
                    logger.error(f"Error calculating Qveto for event {event.hash_id}: {e}")

                if config.plot_trigger:
                    plot_trigger_flow(trigger_folder, event, cluster)

                if config.plot_sky_map:
                    plot_skymap_flow(trigger_folder, event, event_skymap_statistics)

            #################### Add event to catalog ####################
            for trigger in events_data:
                catalog_file = add_event_to_catalog(working_dir, config.catalog_dir, trigger_data=trigger,
                                                    catalog_file=catalog_file)

            logger.info("Memory usage: %f.2 MB", psutil.Process().memory_info().rss / 1024 / 1024)


# create_single_trigger_folder, save_trigger, add_event_to_catalog are imported above
# from pycwb.modules.workflow_utils and kept available here for backward compatibility.


# def process_job_segment_dask(working_dir, config, job_seg, plot=False, compress_json=True, client=None):
#     print_job_info(job_seg)
#
#     if not job_seg.frames and not job_seg.noise and not job_seg.injections:
#         raise ValueError("No data to process")
#
#     if job_seg.frames:
#         frame_data = client.map(read_single_frame_from_job_segment,
#                                 [config] * len(job_seg.frames),
#                                 [job_seg] * len(job_seg.frames),
#                                 job_seg.frames)
#         data = client.submit(merge_frames, job_seg, frame_data, config.segEdge)
#     else:
#         data = None
#
#     if job_seg.noise:
#         data = client.submit(generate_noise_for_job_seg, job_seg, config.inRate, data=data)
#     if job_seg.injections:
#         data = client.submit(generate_injection, config, job_seg, data)
#
#     xtalk_catalog = client.submit(load_catalog, config.MRAcatalog)
#     conditioned_data = client.submit(data_conditioning, config, data)
#     fragment_clusters_multi_res = client.map(coherence_single_res_wrapper, list(range(config.nRES)),
#                                              [config] * config.nRES, [conditioned_data] * config.nRES)
#
#     trigger_folders = client.submit(supercluster_and_likelihood, working_dir, config, job_seg,
#                                  fragment_clusters_multi_res, conditioned_data, xtalk_catalog)
#
#     return client.gather(trigger_folders)
