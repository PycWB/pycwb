from pycwb.modules.logger import logger_init
from pycwb.workflow.subflow.prepare_job_runs import prepare_job_runs
from pycwb.utils.module import import_function
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import faulthandler
import logging
import os

logger = logging.getLogger(__name__)


def _worker_initializer():
    """Enable faulthandler in each worker to capture segmentation violations."""
    faulthandler.enable()


def process_single_with_flag(working_dir, config, job_seg, compress_json, overwrite):
    logger_init(log_file=f"{working_dir}/log/job_{job_seg.index}.log", log_level="INFO")
    logger.info(f"Processing job segment {job_seg.index} on {multiprocessing.current_process()}")
    logger.info(f"Loading segment processer: {config.segment_processer}")
    main_func = import_function(config.segment_processer)
    logger.info(f"Segment processer loaded: {main_func}")

    if not overwrite:
        if os.path.exists(f"{working_dir}/job_status/job_{job_seg.index}.done"):
            logger.info(f"Job segment {job_seg.index} is already done")
            return

    try:
        main_func(working_dir, config, job_seg, compress_json=compress_json)

        # create a flag file to indicate the job is done
        try:
            with open(f"{working_dir}/job_status/job_{job_seg.index}.done", 'w') as f:
                f.write("")
        except Exception as e:
            logger.error(f"Failed to create job done flag file: {e}")

    except Exception as e:
        logger.error(f"Error processing job segment: {job_seg.index}")
        logger.exception(e)
        # create a flag file to indicate the job is failed
        try:
            with open(f"{working_dir}/job_status/job_{job_seg.index}.failed", 'w') as f:
                f.write("")
        except Exception as e:
            logger.error(f"Failed to create job failed flag file: {e}")
            return e


def search(file_name, working_dir='.', overwrite=False, log_file=None, log_level="INFO",
           n_proc=1, plot=None, config_vars: str = None, input_dir=None,
           compress_json=False, dry_run=False):
    # TODO: optimize the plot control
    logger_init(log_file, log_level)
    job_segments, config, working_dir = prepare_job_runs(working_dir, file_name, 1, dry_run, overwrite,
                                                         config_vars=config_vars, input_dir=input_dir,
                                                         plot=plot, compress_json=compress_json)

    if dry_run:
        return job_segments
    # log_prints()
    # cluster = LocalCluster(n_workers=n_proc, processes=True, threads_per_worker=1)
    # cluster.scale(n_proc)
    # client = Client(cluster)

    logger.info(f"Test loading segment processer: {config.segment_processer}")
    main_func = import_function(config.segment_processer)
    logger.info(f"Segment processer loaded: {main_func}")

    if len(job_segments) == 0:
        logger.warning("No job segments to process")
        return
    
    if len(job_segments) == 1:
        main_func(working_dir, config, job_segments[0], compress_json=compress_json)
        return
    
    # Use ProcessPoolExecutor for modern parallel processing with segfault capture via faulthandler.
    # max_tasks_per_child=1 restarts workers after each task to avoid memory leaks (requires Python 3.11+).
    # _worker_initializer enables faulthandler so segmentation violations print a native traceback
    # to stderr instead of silently terminating the worker.
    exceptions = []
    executor = ProcessPoolExecutor(
        max_workers=n_proc,
        max_tasks_per_child=1,  # Restart workers after each task to avoid memory leaks
        initializer=_worker_initializer,
    )
    try:
        logger.info(f"Starting {n_proc} workers")
        futures = {
            executor.submit(process_single_with_flag, working_dir, config, job_seg, compress_json, overwrite): job_seg
            for job_seg in job_segments
        }

        for future in as_completed(futures):
            job_seg = futures[future]
            try:
                err = future.result()
                if err:
                    exceptions.append(err)
            except Exception as e:
                logger.error(f"Job segment {job_seg.index} raised an exception: {e}")
                exceptions.append(e)

        logger.info("All jobs are done")
    finally:
        # Explicitly terminate all worker processes before shutdown.
        # ProcessPoolExecutor's internal management thread is non-daemon and can hang
        # indefinitely waiting on a pipe from workers killed by a signal (e.g. SIGSEGV).
        print("Terminating worker processes...")
        for proc in list(executor._processes.values()):
            try:
                proc.terminate()
            except Exception:
                pass
        executor.shutdown(wait=False, cancel_futures=True)
        if exceptions:
            os._exit(1)  # nuclear option: skip ExceptionGroup, just die

    if exceptions:
        raise ExceptionGroup("JobFailure", exceptions)
    # for job_seg in job_segments:
    #     main_func(working_dir, config, job_seg, compress_json=compress_json)

    # client.close()