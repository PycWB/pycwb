import multiprocessing
import os
import getpass
import shutil
import logging
from pathlib import Path
from pycwb.modules.logger import logger_init, log_prints
from pycwb.workflow.subflow import prepare_job_runs, load_batch_run
from pycwb.utils.module import import_function
from pycwb.modules.condor.condor import HTCondor

logger = logging.getLogger(__name__)


def batch_setup(file_name, working_dir='.',
                overwrite=False, log_file=None, log_level="INFO",
                compress_json=True, cluster="condor", conda_env=None, additional_init="",
                accounting_group=None, job_per_worker=10, n_proc=1, memory="6GB", disk="4GB",
                dry_run=False, submit=False):
    import htcondor
    from htcondor import dags

    logger_init(log_file, log_level)
    job_segments, config, working_dir = prepare_job_runs(working_dir, file_name, n_proc, dry_run, overwrite,
                                                         compress_json=compress_json)

    if dry_run:
        return job_segments

    if cluster == "condor":
        condor = HTCondor(working_dir, conda_env, additional_init, accounting_group, job_per_worker,
                          n_proc, memory, disk)
        condor.create(job_segments, submit=submit)
    else:
        raise ValueError(f"Unsupported cluster type: {cluster}, only support condor for now")


def batch_run(config_file, working_dir='.', log_file=None, log_level="INFO",
              jobs=None, n_proc=1, compress_json=True):
    job_segments, config, working_dir, catalog_file = load_batch_run(working_dir, config_file, jobs,
                                                       n_proc=n_proc, compress_json=compress_json)
    logger_init(log_file, log_level)

    logger.info(f"Loading segment processer: {config.segment_processer}")
    main_func = import_function(config.segment_processer)
    logger.info(f"Segment processer loaded: {main_func}")

    exceptions = []
    for job_seg in job_segments:
        # check if the job is done
        if os.path.exists(f"{working_dir}/job_status/job_{job_seg.index}.done"):
            print(f"Job segment {job_seg.index} is already done")
            continue

        try:
            # TODO: run the job in a separate process to prevent memory leak
            # process = multiprocessing.Process(target=process_job_segment,
            #                                   args=(working_dir, config, job_seg, compress_json, catalog_file))
            # process.start()
            # process.join()
            main_func(working_dir, config, job_seg,
                                compress_json=compress_json, catalog_file=catalog_file)

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
