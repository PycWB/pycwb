import os

from prefect import flow, unmapped, task
from prefect_dask import get_dask_client
from dask_jobqueue import SLURMCluster, HTCondorCluster
from prefect_dask.task_runners import DaskTaskRunner
from dask.distributed import Client, LocalCluster
from .tasks.builtin import check_env, read_config, create_working_directory, print_job_info, \
    check_if_output_exists, create_output_directory, create_job_segment, \
    generate_noise_for_job_seg_task, \
    create_catalog_file, create_web_dir, save_trigger, generate_injection_task, \
    read_file_from_job_segment, merge_frame_task, data_conditioning_task, \
    coherence_task, supercluster_and_likelihood_task, load_xtalk_catalog, reconstruct_waveform, plot_triggers


@flow
def process_job_segment(working_dir, config, job_seg,
                        plot=False, compress_json=True):
    print_job_info(job_seg)
    data = []

    # Data retrieval or generation
    if not job_seg.frames and not job_seg.noise and not job_seg.injections:
        raise ValueError("No data to process")

    if job_seg.frames:
        frame_data = read_file_from_job_segment.map(config, job_seg, job_seg.frames)
        data = merge_frame_task.submit(job_seg, frame_data, config.segEdge)
    if job_seg.noise:
        data = generate_noise_for_job_seg_task.submit(job_seg, config.inRate, data=data)
    if job_seg.injections:
        data = generate_injection_task.submit(config, job_seg, data)

    xtalk_catalog = load_xtalk_catalog.submit(config.MRAcatalog)
    conditioned_data = data_conditioning_task.map(config, unmapped(data), range(len(job_seg.ifos)))
    fragment_clusters_multi_res = coherence_task.map(config, unmapped(conditioned_data), range(config.nRES))

    # TODO: maybe need to save trigger first, then for each post-production task,
    #  load the trigger to avoid the head-node bottleneck
    triggers_data = supercluster_and_likelihood_task.submit(config, fragment_clusters_multi_res,
                                                            conditioned_data, xtalk_catalog)

    save_trigger.map(working_dir, config, job_seg, triggers_data)
    reconstructed_waves = reconstruct_waveform.map(working_dir, config, job_seg, triggers_data)

    if plot:
        plot_triggers.map(working_dir, config, job_seg, triggers_data, reconstructed_waves)


@flow(log_prints=True)
def search(file_name, working_dir='.', overwrite=False, submit=False, log_file=None,
           n_proc=1, plot=False, compress_json=True, dry_run=False):
    working_dir = os.path.abspath(working_dir)
    create_working_directory(working_dir)
    check_env()
    config = read_config(file_name)

    job_segments = create_job_segment(config)

    # dry run
    if dry_run:
        return job_segments

    # override n_proc in config
    if n_proc != 0:
        config.nproc = n_proc

    # Create runner
    if not submit:
        if config.nproc > 1 and not submit:
            cluster = LocalCluster(n_workers=n_proc, processes=True, threads_per_worker=1)
            cluster.scale(n_proc)
            client = Client(cluster)
            address = client.scheduler.address
            subflow = process_job_segment.with_options(task_runner=DaskTaskRunner(address=address),
                                                       log_prints=True, retries=0)
        elif config.nproc == 1:
            subflow = process_job_segment
        else:
            raise ValueError("Cannot run with n_proc < 0")
    else:
        if submit == 'condor':
            raise NotImplementedError
        elif submit == 'slurm':
            raise NotImplementedError

    check_if_output_exists(working_dir, config.outputDir, overwrite)
    create_output_directory(working_dir, config.outputDir, config.logDir, file_name)

    create_catalog_file(working_dir, config, job_segments)
    create_web_dir(working_dir, config.outputDir)
    load_xtalk_catalog(config.MRAcatalog)
    # slags = job_generator(len(config.ifo), config.slagMin, config.slagMax, config.slagOff, config.slagSize)

    for job_seg in job_segments:
        # TODO: customize the name
        subflow(working_dir, config, job_seg, plot, compress_json)


