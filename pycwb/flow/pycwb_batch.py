from prefect import flow
from pycwb.flow.sqeuences.builtin import prepare_job_runs


@flow(log_prints=True)
def search(file_name, working_dir='.', overwrite=False, submit=False, log_file=None,
                 n_proc=1, plot=False, compress_json=True, dry_run=False):
    # create job segments
    job_segments, config = flow(prepare_job_runs)(working_dir, file_name, n_proc, dry_run, overwrite)
    # dry run
    if dry_run:
        return job_segments

    # Create runner
