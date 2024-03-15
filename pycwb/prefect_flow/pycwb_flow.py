import asyncio
import os

from prefect.utilities.annotations import quote
from prefect import flow, unmapped, task, context
from dask_jobqueue import SLURMCluster, HTCondorCluster
from prefect_dask.task_runners import DaskTaskRunner
from dask.distributed import Client, LocalCluster

from pycwb.workflow.subflow import prepare_job_runs
from .tasks.builtin import print_job_info, \
    generate_noise_for_job_seg_task, \
    save_trigger, generate_injection_task, \
    read_file_from_job_segment, merge_frame_task, data_conditioning_task, \
    coherence_task, supercluster_and_likelihood_task, load_xtalk_catalog, reconstruct_waveform, plot_triggers


@task
def map_wrapper(data):
    return list(range(len(data)))


# TODO: add flow_run_name
@flow
async def process_job_segment(working_dir, config, job_seg,
                        plot=False, compress_json=True):
    print_job_info.submit(job_seg)

    # Data retrieval or generation

    if not job_seg.frames and not job_seg.noise and not job_seg.injections:
        raise ValueError("No data to process")

    if job_seg.frames:
        frame_data = read_file_from_job_segment.map(config, job_seg, job_seg.frames)
        data = merge_frame_task.submit(job_seg, frame_data, config.segEdge)
    else:
        data = None
    if job_seg.noise:
        data = generate_noise_for_job_seg_task.submit(job_seg, config, data=data)
    if job_seg.injections:
        data = generate_injection_task.submit(config, job_seg, data)

    xtalk_catalog = load_xtalk_catalog.submit(config.MRAcatalog)
    conditioned_data = data_conditioning_task.map(quote(config), unmapped(data), range(len(job_seg.ifos)))
    fragment_clusters_multi_res = coherence_task.map(config, unmapped(conditioned_data), range(config.nRES))

    # TODO: maybe need to save trigger first, then for each post-production task,
    #  load the trigger to avoid the head-node bottleneck
    triggers_data = supercluster_and_likelihood_task.submit(config, fragment_clusters_multi_res,
                                                            conditioned_data, xtalk_catalog)

    triggers_indexes = map_wrapper.submit(triggers_data)
    trigger_folders = save_trigger.map(working_dir, config, job_seg, unmapped(triggers_data), triggers_indexes)
    reconstruct_waveform.map(unmapped(trigger_folders), config, job_seg, unmapped(triggers_data), triggers_indexes, plot)

    if plot:
        plot_triggers.map(unmapped(trigger_folders), unmapped(triggers_data), triggers_indexes)


@flow(log_prints=True)
async def search(file_name, working_dir='.', overwrite=False, submit=False, log_file=None,
           n_proc=1, plot=False, compress_json=True, dry_run=False):
    # create job segments and read user parameters
    job_segments, config, working_dir = flow(prepare_job_runs)(working_dir, file_name, n_proc, dry_run, overwrite)

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
            subflow = process_job_segment.with_options(task_runner=DaskTaskRunner(address=address),
                                                       log_prints=True, retries=0)
        elif config.nproc == 1:
            subflow = process_job_segment
        else:
            raise ValueError("Cannot run with n_proc < 0")
    else:
        # create workers for job submission system
        import getpass

        cpu_per_worker = 2
        mem_per_worker = int(3 * cpu_per_worker)
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
            subflow = process_job_segment.with_options(task_runner=DaskTaskRunner(address=address),
                                                       log_prints=True, retries=0)
        elif submit == 'slurm':
            raise NotImplementedError
        else:
            raise ValueError("Unknown submit option, only support 'condor' and 'slurm'")

    # run subflows
    parent_job_name = context.get_run_context().flow_run.dict()['name']
    subjobs = []
    for i, job_seg in enumerate(job_segments):
        # TODO: customize the name
        subjobs.append(subflow.with_options(name=f"{parent_job_name}-{i}")(working_dir, config, job_seg, plot, compress_json))

    await asyncio.gather(*subjobs)


