from pycwb.modules.logger import logger_init
from pycwb.workflow.subflow import prepare_job_runs
from pycwb.workflow.subflow.process_job_segment import process_job_segment


def search(file_name, working_dir='.', overwrite=False, log_file=None, log_level="INFO",
           n_proc=1, plot=None, compress_json=False, dry_run=False):
    # TODO: optimize the plot control
    logger_init(log_file, log_level)
    job_segments, config, working_dir = prepare_job_runs(working_dir, file_name, n_proc, dry_run, overwrite,
                                                         plot=plot, compress_json=compress_json)

    if dry_run:
        return job_segments
    # log_prints()
    # cluster = LocalCluster(n_workers=n_proc, processes=True, threads_per_worker=1)
    # cluster.scale(n_proc)
    # client = Client(cluster)

    for job_seg in job_segments:
        process_job_segment(working_dir, config, job_seg, compress_json=compress_json)

    # client.close()