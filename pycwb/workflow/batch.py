from pycwb.modules.logger import logger_init, log_prints
from pycwb.workflow.subflow import prepare_job_runs, load_batch_run
from pycwb.workflow.subflow.process_job_segment import process_job_segment


def batch_setup(file_name, working_dir='.', overwrite=False, log_file=None, log_level="INFO",
                 job_per_worker=10, n_proc=1, dry_run=False):
    import htcondor

    logger_init(log_file, log_level)
    job_segments, config, working_dir = prepare_job_runs(working_dir, file_name, n_proc, dry_run, overwrite)

    n_workers = (len(job_segments) + job_per_worker - 1) // job_per_worker

    cat_job = htcondor.Submit({
        "executable": "pycwb batch-run",
        "arguments": f"--job-file={working_dir}/config/job_segments.json "
                     f"--job-start=$(job_start) --job-end=$(job_end) --n-proc={n_proc}",
        # "transfer_input_files": "$(input_file)",    # we also need HTCondor to move the file to the execute node
        # "should_transfer_files": "yes",             # force HTCondor to transfer files even though we're running entirely inside a container (and it normally wouldn't need to)
        "output": "cat-$(ProcId).out",
        "error": "cat-$(ProcId).err",
        "log": "cat.log",
        "request_cpus": f"{n_proc}",
        "request_memory": "4GB",
        "request_disk": "4GB",
    })

    schedd = htcondor.Schedd()

    jobs = [{
        'job_start': i * job_per_worker,
        'job_end': min((i + 1) * job_per_worker, len(job_segments))
    } for i in range(n_workers)]

    # TODO: check if the jobs are complete, skip the completed job

    submit_result = schedd.submit(cat_job, itemdata = iter(jobs))  # submit one job for each item in the itemdata

    print(submit_result)


def batch_run(config_file, working_dir='.', log_file=None, log_level="INFO",
              job_file=None, job_start=0, job_end=1,
              n_proc=1, plot=None, compress_json=True):
    job_segments, config, working_dir = load_batch_run(working_dir, config_file, job_file, job_start, job_end, n_proc)
    logger_init(log_file, log_level)

    for job_seg in job_segments:
        process_job_segment(working_dir, config, job_seg, plot, compress_json)
