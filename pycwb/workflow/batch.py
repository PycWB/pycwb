import multiprocessing
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
                compress_json=True, cluster="condor", conda_env=None, additional_init="",
                accounting_group=None, job_per_worker=10, n_proc=1, dry_run=False, submit=False):
    import htcondor

    logger_init(log_file, log_level)
    job_segments, config, working_dir = prepare_job_runs(working_dir, file_name, n_proc, dry_run, overwrite,
                                                         compress_json=compress_json)

    if dry_run:
        return job_segments

    if accounting_group is None:
        raise ValueError("Accounting group is required for condor batch submission")

    # create the DAG directory
    dag_dir = (Path.cwd() / 'condor').absolute()

    # blow away any old files
    shutil.rmtree(dag_dir, ignore_errors=True)
    os.makedirs(dag_dir, exist_ok=True)

    # create a bash script to run the job
    n_workers = (len(job_segments) + job_per_worker - 1) // job_per_worker
    jobs = [{
        'jobs': f"{i * job_per_worker}-{min((i + 1) * job_per_worker, len(job_segments)) - 1}"
    } for i in range(n_workers)]
    config_file_name = os.path.basename(file_name)

    # create run.sh
    with open(f"{dag_dir}/run.sh", 'w') as f:
        f.write(f"""#!/bin/bash
source /cvmfs/oasis.opensciencegrid.org/ligo/sw/conda/etc/profile.d/conda.sh
conda activate {conda_env}
{additional_init}
pycwb batch-runner {working_dir}/config/user_parameters.yaml --work-dir={working_dir} --jobs=$1 --n-proc={n_proc}
        """)

    # add execute permission to run.sh
    os.chmod(f"{dag_dir}/run.sh", 0o755)

    # create the submit description for the batch job
    batch_job = htcondor.Submit({
        "executable": "run.sh",
        "arguments": f"$(jobs)",  # Passing jobs as an argument
        "transfer_input_files": f"{working_dir}/job_status, {working_dir}/config, "
                                f"{working_dir}/input, {working_dir}/wdmXTalk",
        "should_transfer_files": "yes",
        "output": "../log/batch-$(jobs).out",
        "error": "../log/batch-$(jobs).err",
        "log": "../log/batch-$(jobs).log",
        "accounting_group": accounting_group,
        "accounting_group_user": getpass.getuser(),
        "request_cpus": f"{n_proc}",
        "request_memory": "6GB",
        "request_disk": "4GB",
    })

    # create merge.sh
    with open(f"{dag_dir}/merge.sh", 'w') as f:
        f.write(f"""#!/bin/bash
source /cvmfs/oasis.opensciencegrid.org/ligo/sw/conda/etc/profile.d/conda.sh
conda activate {conda_env}
{additional_init}
pycwb merge-catalog --work-dir={working_dir}
        """)

    # add execute permission to merge.sh
    os.chmod(f"{dag_dir}/merge.sh", 0o755)

    merge_job = htcondor.Submit(
        executable="merge.sh",
        transfer_input_files=f"{working_dir}/catalog",
        should_transfer_files="yes",
        log='../log/merge.log',
        output='../log/merge.out',
        error='../log/merge.err',
        accounting_group=accounting_group,
        accounting_group_user=getpass.getuser(),
        request_cpus=n_proc,
        request_memory='8GB',
        request_disk='4GB',
    )

    dag = dags.DAG()

    # create the tile layer, passing in the submit description for a tile job and the tile vars
    batch_layer = dag.layer(
        name='pycwb_batch',
        submit_description=batch_job,
        vars=jobs,
    )

    merge_layer = batch_layer.child_layer(
        name = 'merge',
        submit_description = merge_job,
    )

    # make the magic happen!
    dag_file = dags.write_dag(dag, dag_dir, dag_file_name=f'pycwb_{os.path.basename(working_dir)}.dag')

    print(f'DAG directory: {dag_dir}')
    print(f'DAG description file: {dag_file}')

    if submit:
        dag_submit = htcondor.Submit.from_dag(str(dag_file), {'force': 1})
        print('------------------------')
        print(dag_submit)
        print('------------------------')
        os.chdir(dag_dir)

        schedd = htcondor.Schedd()
        cluster_id = schedd.submit(dag_submit).cluster()

        print(f"DAGMan job cluster is {cluster_id}")

        os.chdir(working_dir)


def batch_run(config_file, working_dir='.', log_file=None, log_level="INFO",
              jobs=None, n_proc=1, compress_json=True):
    job_segments, config, working_dir, catalog_file = load_batch_run(working_dir, config_file, jobs,
                                                       n_proc=n_proc, compress_json=compress_json)
    logger_init(log_file, log_level)

    exceptions = []
    for job_seg in job_segments:
        # check if the job is done
        if os.path.exists(f"{working_dir}/job_status/job_{job_seg.index}.done"):
            print(f"Job segment {job_seg.index} is already done")
            continue

        try:
            # TODO: run the job in a separate process to prevent memory leak
            process = multiprocessing.Process(target=process_job_segment,
                                              args=(working_dir, config, job_seg, compress_json, catalog_file))
            process.start()
            process.join()

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

            exceptions.append(e)

    if exceptions:
        raise ExceptionGroup("JobFailure", exceptions)
