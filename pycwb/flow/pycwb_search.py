from prefect import flow, unmapped
from prefect_dask.task_runners import DaskTaskRunner
from dask.distributed import Client, LocalCluster
from .tasks.builtin import check_env, read_config, create_working_directory, print_job_info, \
                        check_if_output_exists, create_output_directory, create_job_segment, \
                        create_catalog_file, create_web_dir, save_trigger, \
                        read_file_from_job_segment, merge_frame_task, data_conditioning_task, \
                        coherence_task, supercluster_and_likelihood_task, load_xtalk_catalog


@flow
def process_job_segment(working_dir, config, job_seg, xtalk_catalog):
    print_job_info(job_seg)
    data = read_file_from_job_segment.map(config, job_seg, job_seg.frames)
    data_merged = merge_frame_task.submit(job_seg, data, config.segEdge)
    conditioned_data = data_conditioning_task.map(config, data_merged)
    fragment_clusters_multi_res = coherence_task.map(config, unmapped(conditioned_data), range(config.nRES))

    triggers_data = supercluster_and_likelihood_task.submit(config, fragment_clusters_multi_res,
                                                            conditioned_data, xtalk_catalog)

    save_trigger.map(working_dir, config, job_seg, triggers_data)


@flow
def search(file_name, working_dir='.', overwrite=False, submit=False, log_file=None,
           n_proc=1, plot=False, compress_json=True):
    create_working_directory(working_dir)
    check_env()
    config = read_config(file_name)

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
        raise ValueError("Job submission not implemented yet.")

    check_if_output_exists(working_dir, config.outputDir, overwrite)
    create_output_directory(working_dir, config.outputDir, config.logDir, file_name)

    job_segments = create_job_segment(config)
    create_catalog_file(working_dir, config, job_segments)
    create_web_dir(working_dir, config.outputDir)
    xtalk_catalog = load_xtalk_catalog.submit(config.MRAcatalog)
    # slags = job_generator(len(config.ifo), config.slagMin, config.slagMax, config.slagOff, config.slagSize)

    for job_seg in job_segments:
        # TODO: customize the name
        subflow(working_dir, config, job_seg, xtalk_catalog)


