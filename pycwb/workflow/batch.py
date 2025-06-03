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
from pycwb.modules.slurm.slurm import Slurm
logger = logging.getLogger(__name__)


def batch_setup(file_name, working_dir='.',
                overwrite=False, log_file=None, log_level="INFO",
                compress_json=True, cluster="condor", conda_env=None, additional_init="",
                accounting_group=None, job_per_worker=10, n_proc=1, memory="6GB", disk="4GB",
                dry_run=False, submit=False):
    logger_init(log_file, log_level)
    job_segments, config, working_dir = prepare_job_runs(working_dir, file_name, n_proc, dry_run, overwrite,
                                                         compress_json=compress_json)

    if dry_run:
        return job_segments

    if cluster == "condor":
        condor = HTCondor(working_dir, conda_env, additional_init, accounting_group, job_per_worker,
                          n_proc, memory, disk)
        condor.create(job_segments, submit=submit)
    elif cluster == "slurm":
        slurm = Slurm(working_dir, conda_env, additional_init, job_per_worker,
                      n_proc, memory, disk)
        slurm.create(job_segments, submit=submit)
    else:
        raise ValueError(f"Unsupported cluster type: {cluster}, only support condor and slurm")


def data_collector(working_dir, queue):
    logger_init(log_file=None, log_level="INFO", worker_prefix=f'DataCollector')
    while True:
        item = queue.get()
        if item is None:  # Sentinel to terminate
            break
        try:
            logger.info(f"Merging {item}")
        except Exception as e:
            logger.error(f"Error merging {item}: {e}")


def process_single_with_flag(main_func, queue, working_dir, config, job_seg, compress_json=True, catalog_file=None):
    logger_init(log_file=f"{working_dir}/log/job_{job_seg.index}.log", log_level="INFO")
    logger.info(f"Processing job segment {job_seg.index} with {getpass.getuser()} on {multiprocessing.current_process()}")
    if os.path.exists(f"{working_dir}/job_status/job_{job_seg.index}.done"):
        logger.info(f"Job segment {job_seg.index} is already done")
        return

    try:
        main_func(working_dir, config, job_seg,
                            compress_json=compress_json, catalog_file=catalog_file, queue=queue)

        # create a flag file to indicate the job is done
        try:
            with open(f"{working_dir}/job_status/job_{job_seg.index}.done", 'w') as f:
                f.write("")
        except Exception as e:
            logger.error(f"Failed to create job done flag file: {e}")

    except Exception as e:
        logger.error(f"Error processing job segment: {job_seg}")
        logger.error(e)
        # create a flag file to indicate the job is failed
        try:
            with open(f"{working_dir}/job_status/job_{job_seg.index}.failed", 'w') as f:
                f.write("")
        except Exception as e:
            logger.error(f"Failed to create job failed flag file: {e}")
            return e


def batch_run(config_file, working_dir='.', log_file=None, log_level="INFO",
              jobs=None, n_proc=1, compress_json=True, n_workers=1):
    job_segments, config, working_dir, catalog_file = load_batch_run(working_dir, config_file, jobs,
                                                       n_proc=n_proc, compress_json=compress_json)
    logger_init(log_file, log_level)

    logger.info(f"Loading segment processer: {config.segment_processer}")
    main_func = import_function(config.segment_processer)
    logger.info(f"Segment processer loaded: {main_func}")

    # create a queue and a worker to collect the data
    manager = multiprocessing.Manager()
    queue = manager.Queue()
    data_collector_worker = multiprocessing.Process(target=data_collector, args=(working_dir, queue))
    data_collector_worker.start()

    exceptions = []
    with multiprocessing.Pool(
        n_workers,
        maxtasksperchild=1,  # Restart workers after each task to avoid memory leaks
        ) as pool:
        logger.info(f"Starting {n_workers} workers")
        results = []
        for job_seg in job_segments:
            r = pool.apply_async(process_single_with_flag,
                                    args=(main_func, queue, working_dir, config, job_seg,
                                        compress_json, catalog_file))
            results.append(r)

        for r in results:
            try:
                err = r.get()
                if err:
                    exceptions.append(err)
            except Exception as e:
                exceptions.append(e)

        pool.close()
        pool.join()
        logger.info("All jobs are done, waiting for data collector to finish")

        # send a sentinel to terminate the data collector
        queue.put(None)
        data_collector_worker.join()

    if exceptions:
        raise ExceptionGroup("JobFailure", exceptions)
