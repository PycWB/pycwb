import os
import getpass
import shutil
from pathlib import Path

from htcondor import dags

from pycwb.modules.logger import logger_init, log_prints
from pycwb.workflow.subflow import prepare_job_runs, load_batch_run
from pycwb.workflow.subflow.process_job_segment import process_job_segment


def batch_setup(file_name, working_dir='.',
                overwrite=False, log_file=None, log_level="INFO",
                compress_json=True, cluster="condor", conda_env=None,
                accounting_group=None, job_per_worker=10, n_proc=1, dry_run=False, submit=False):
    import htcondor

    logger_init(log_file, log_level)
    job_segments, config, working_dir = prepare_job_runs(working_dir, file_name, n_proc, dry_run, overwrite,
                                                         compress_json=compress_json)

    if dry_run:
        return job_segments

    if accounting_group is None:
        raise ValueError("Accounting group is required for condor batch submission")

    n_workers = (len(job_segments) + job_per_worker - 1) // job_per_worker
    jobs = [{
        'jobs': f"{i * job_per_worker}-{min((i + 1) * job_per_worker, len(job_segments)) - 1}"
    } for i in range(n_workers)]

    batch_job = htcondor.Submit({
        "executable": "source",
        "arguments": f"/cvmfs/oasis.opensciencegrid.org/ligo/sw/conda/etc/profile.d/conda.sh && "
                     f"conda activate {conda_env} && "
                     f"pycwb batch-run {working_dir}/config/{config} --work-dir={working_dir} "
                     f"--jobs=$(jobs) --n-proc={n_proc}",
        "transfer_input_files": f"{working_dir}/job_status, {working_dir}/config",
        "should_transfer_files": "yes",
        "output": "log/batch-$(ProcId).out",
        "error": "log/batch-$(ProcId).err",
        "log": "log/batch-$(ProcId).log",
        "accounting_group": accounting_group,
        "accounting_group_user": getpass.getuser(),
        "request_cpus": f"{n_proc}",
        "request_memory": "4GB",
        "request_disk": "4GB",
    })

    # merge_job = htcondor.Submit(
    #     executable='pycwb batch-merge',
    #     arguments=f"{working_dir}/config/{config}",
    #     transfer_input_files="",
    #     log='merge.log',
    #     output='merge.out',
    #     error='merge.err',
    #     request_cpus='1',
    #     request_memory='128MB',
    #     request_disk='1GB',
    # )

    dag = dags.DAG()

    # create the tile layer, passing in the submit description for a tile job and the tile vars
    batch_layer = dag.layer(
        name='pycwb_batch',
        submit_description=batch_job,
        vars=jobs,
    )

    # merge_layer = batch_layer.child_layer(
    #     name = 'merge',
    #     submit_description = merge_job,
    # )

    dag_dir = (Path.cwd() / 'condor').absolute()

    # blow away any old files
    shutil.rmtree(dag_dir, ignore_errors=True)

    # make the magic happen!
    dag_file = dags.write_dag(dag, dag_dir)

    print(f'DAG directory: {dag_dir}')
    print(f'DAG description file: {dag_file}')

    if submit:
        dag_submit = htcondor.Submit.from_dag(str(dag_file), {'force': 1})
        print(dag_submit)


def batch_run(config_file, working_dir='.', log_file=None, log_level="INFO",
              jobs=None, n_proc=1, compress_json=True):
    job_segments, config, working_dir, catalog_file = load_batch_run(working_dir, config_file, jobs,
                                                       n_proc=n_proc, compress_json=compress_json)
    logger_init(log_file, log_level)

    for job_seg in job_segments:
        # check if the job is done
        if os.path.exists(f"{working_dir}/job_status/job_{job_seg.index}.done"):
            print(f"Job segment {job_seg.index} is already done")
            continue

        try:
            process_job_segment(working_dir, config, job_seg, compress_json, catalog_file=catalog_file)

            # create a flag file to indicate the job is done
            try:
                with open(f"{working_dir}/job_status/job_{job_seg.index}.done", 'w') as f:
                    f.write("")
            except Exception as e:
                print(f"Failed to create job done flag file: {e}")

        except Exception as e:
            print(f"Error processing job segment: {job_seg}")
            print(e)

            # create a flag file to indicate the job is failed
            try:
                with open(f"{working_dir}/job_status/job_{job_seg.index}.failed", 'w') as f:
                    f.write("")
            except Exception as e:
                print(f"Failed to create job failed flag file: {e}")
