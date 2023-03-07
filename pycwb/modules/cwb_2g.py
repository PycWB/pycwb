import os, time
from concurrent.futures import ProcessPoolExecutor as Pool
import multiprocessing
import pycwb
from pycwb import logger_init
from pycwb.config import Config, CWBConfig
from pycwb.modules.plot import plot_spectrogram
from pycwb.modules.read_data import read_from_gwf, generate_noise, read_from_config, read_from_job_segment
from pycwb.modules.data_conditioning import data_conditioning
from pycwb.modules.coherence import create_network
from pycwb.modules.coherence import coherence
from pycwb.modules.super_cluster import supercluster
from pycwb.modules.likelihood import likelihood
from pycwb.modules.job_segment import select_job_segment
from pycwb.modules.catalog import create_catalog, add_events_to_catalog
import logging

logger = logging.getLogger(__name__)


def analyze_job_segment(config, job_seg):
    # config, job_seg = args
    start_time = time.perf_counter()

    job_id = job_seg.index
    # log job info
    logger.info(f"Job ID: {job_id}")
    logger.info(f"Start time: {job_seg.start_time}")
    logger.info(f"End time: {job_seg.end_time}")
    logger.info(f"Duration: {job_seg.end_time - job_seg.start_time}")

    data = read_from_job_segment(config, job_seg)

    # data conditioning
    tf_maps, nRMS_list = data_conditioning(config, data)

    # initialize network
    net, wdm_list = create_network(1, config, tf_maps, nRMS_list)

    # calculate coherence
    sparse_table_list, cluster_list = coherence(config, net, tf_maps, wdm_list)

    # supercluster
    pwc_list = supercluster(config, net, cluster_list, sparse_table_list)

    # likelihood
    events = likelihood(config, net, sparse_table_list, pwc_list, wdm_list)

    # save events to pickle
    # import pickle
    # with open(f'{config.outputDir}/events_{job_id}.pkl', 'wb') as f:
    #     pickle.dump(events, f)

    import matplotlib.pyplot as plt
    plot = plot_spectrogram(tf_maps[0], figsize=(24, 6), gwpy_plot=True)

    # plot boxes on the plot
    i = 0
    boxes = [[e.start[i], e.stop[i], e.low[i], e.high[i]] for e in events if len(e.start) > 0]

    for box in boxes:
        ax = plot.gca()
        ax.add_patch(plt.Rectangle((box[0], box[2]), box[1] - box[0], box[3] - box[2], linewidth=0.5, fill=False,
                                   color='red'))

    # save to png
    plot.savefig(f'{config.outputDir}/events_{job_id}_all.png')

    # save event to txt
    for i, event in enumerate(events):
        try:
            output = event.json()
            with open(f'{config.outputDir}/event_{job_id}_{i + 1}.json', 'a') as f:
                f.write(output)
        except Exception as e:
            logger.error(e)

    event_summary = [event.summary() for event in events]
    try:
        add_events_to_catalog("catalog.json", event_summary)
    except Exception as e:
        logger.error(e)

    del data, tf_maps, nRMS_list, net, wdm_list, sparse_table_list, cluster_list, pwc_list, events

    # calculate the performance
    end_time = time.perf_counter()
    logger.info("-" * 80)
    logger.info(f"Job {job_id} finished in {round(end_time - start_time, 1)} seconds")
    logger.info(f"Speed factor: {round((job_seg.end_time - job_seg.start_time) / (end_time - start_time), 1)}X")
    logger.info("-" * 80)


def cwb_2g(user_parameters='./user_parameters.yaml', log_file=None, log_level='INFO', no_subprocess=False):
    logger_init(log_file, log_level)

    # set env HOME_WAT_FILTERS
    if not os.environ.get('HOME_WAT_FILTERS'):
        pycwb_path = os.path.dirname(os.path.abspath(pycwb.__file__))
        os.environ['HOME_WAT_FILTERS'] = f"{os.path.abspath(pycwb_path)}/vendor"

    config = Config(user_parameters)

    # create folder for output and log
    if not os.path.exists(config.outputDir):
        os.makedirs(config.outputDir)
    if not os.path.exists(config.logDir):
        os.makedirs(config.logDir)

    job_segments = select_job_segment(config.dq_files, config.ifo, config.frFiles,
                                      config.segLen, config.segMLS, config.segEdge, config.segOverlap,
                                      config.rateANA, config.l_high)

    # data = read_from_config(config)
    # log number of segments
    logger.info(f"Number of segments: {len(job_segments)}")
    logger.info("-" * 80)

    create_catalog("catalog.json", config, job_segments)

    for job_seg in job_segments:
        # analyze_job_segment(config, job_seg)
        # gc.collect()
        if no_subprocess:
            analyze_job_segment(config, job_seg)
        else:
            process = multiprocessing.Process(target=analyze_job_segment, args=(config, job_seg))
            process.start()
            process.join()
        # with Pool(max_workers=1) as pool:
        #     pool.map(analyze_job_segment, [(config, job_seg)])


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
