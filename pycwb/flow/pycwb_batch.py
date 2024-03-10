import os

from prefect.utilities.annotations import quote
from prefect import flow, unmapped, task, context
from dask_jobqueue import SLURMCluster, HTCondorCluster
from prefect_dask.task_runners import DaskTaskRunner
from dask.distributed import Client, LocalCluster
from .tasks.builtin import check_env, read_config, create_working_directory, print_job_info, \
    check_if_output_exists, create_output_directory, create_job_segment, \
    generate_noise_for_job_seg_task, \
    create_catalog_file, create_web_dir, save_trigger, generate_injection_task, \
    read_file_from_job_segment, merge_frame_task, data_conditioning_task, \
    coherence_task, supercluster_and_likelihood_task, load_xtalk_catalog, reconstruct_waveform, plot_triggers


@task
def analysis_sequence(working_dir, config, job_seg, plot):
    if job_seg.frames:
        frame_data = read_file_from_job_segment.fn(config, job_seg, job_seg.frames)
        data = merge_frame_task.fn(job_seg, frame_data, config.segEdge)
    else:
        data = None

    if job_seg.noise:
        data = generate_noise_for_job_seg_task.fn(job_seg, config, data=data)
    if job_seg.injections:
        data = generate_injection_task.fn(config, job_seg, data)

    conditioned_data = []
    for ifo in range(len(job_seg.ifos)):
        conditioned_data.append(data_conditioning_task.fn(config, data, ifo))

    xtalk_catalog = load_xtalk_catalog.fn(config.MRAcatalog)

    fragment_clusters_multi_res = []
    for res in range(config.nRES):
        fragment_clusters_multi_res.append(coherence_task.fn(config, conditioned_data, res))

    triggers_data = supercluster_and_likelihood_task.fn(config, fragment_clusters_multi_res,
                                                        conditioned_data, xtalk_catalog)

    trigger_folders = []
    for trigger_data in triggers_data:
        trigger_folder = save_trigger.fn(working_dir, config, job_seg, trigger_data)
        # TODO: allow disable reconstruct_waveform
        reconstruct_waveform.fn(trigger_folder, config, job_seg, trigger_data, plot=plot)
        trigger_folders.append(trigger_folder)

    if plot:
        for trigger_folder, trigger_data in zip(trigger_folders, triggers_data):
            plot_triggers.fn(trigger_folder, trigger_data)


@flow
def process_job_segments(working_dir, config, job_segments,
                         plot=False, compress_json=True):
    for job_seg in job_segments:
        analysis_sequence.submit(working_dir, config, job_seg, plot)


@flow
def prepare_job_runs(working_dir, config_file, n_proc=1, dry_run=False, overwrite=False):
    # convert to absolute path in case the current working directory is changed
    working_dir = os.path.abspath(working_dir)
    file_name = os.path.abspath(config_file)

    # create working directory and change the current working directory to the given working directory
    create_working_directory(working_dir)

    # check environment
    check_env()

    # read user parameters
    config = read_config(file_name)

    job_segments = create_job_segment(config)

    if not dry_run:
        # override n_proc in config
        if n_proc != 0:
            config.nproc = n_proc

        check_if_output_exists(working_dir, config.outputDir, overwrite)
        create_output_directory(working_dir, config.outputDir, config.logDir, file_name)

        create_catalog_file(working_dir, config, job_segments)
        create_web_dir(working_dir, config.outputDir)
        load_xtalk_catalog(config.MRAcatalog)

    # slags = job_generator(len(config.ifo), config.slagMin, config.slagMax, config.slagOff, config.slagSize)
    return job_segments, config


@flow(log_prints=True)
def search(file_name, working_dir='.', overwrite=False, submit=False, log_file=None,
                 n_proc=1, plot=False, compress_json=True, dry_run=False):
    # create job segments
    job_segments, config = prepare_job_runs(working_dir, file_name, n_proc, dry_run, overwrite)
    # dry run
    if dry_run:
        return job_segments

    # Create runner
    if not submit:
        if config.nproc > 1 and not submit:
            cluster = LocalCluster(n_workers=n_proc, processes=True, threads_per_worker=1)
            cluster.scale(n_proc)
            client = Client(cluster)
            address = client.scheduler.address
            subflow = process_job_segments.with_options(task_runner=DaskTaskRunner(address=address),
                                                        log_prints=True, retries=0)
        elif config.nproc == 1:
            subflow = process_job_segments
        else:
            raise ValueError("Cannot run with n_proc < 0")
    else:
        # create workers for job submission system
        import getpass

        cpu_per_worker = 1
        mem_per_worker = int(3 * cpu_per_worker)
        # workers = math.ceil(n_proc / cpu_per_worker)
        if submit == 'condor':
            job_script_prologue = [f'cd {working_dir}', f'source {working_dir}/start.sh']
            # TODO: customize the account group
            cluster = HTCondorCluster(cores=cpu_per_worker, memory=f"{mem_per_worker}GB", disk="1GB",
                                      job_extra_directives={
                                          'universe': 'vanilla',
                                          'accounting_group': 'ligo.dev.o4.burst.ebbh.cwbonline',
                                          'accounting_group_user': getpass.getuser()
                                      },
                                      log_directory='logs', python='python3',
                                      job_script_prologue=job_script_prologue)
            print(cluster.job_script())
            cluster.scale(n_proc)
            client = Client(cluster)
            address = client.scheduler.address
            subflow = process_job_segments.with_options(task_runner=DaskTaskRunner(address=address),
                                                        log_prints=True, retries=0)
        elif submit == 'slurm':
            raise NotImplementedError
        else:
            raise ValueError("Unknown submit option, only support 'condor' and 'slurm'")

    subflow(working_dir, config, job_segments, plot, compress_json)
