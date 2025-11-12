from pycwb.modules.logger import logger_init
from pycwb.workflow.subflow import prepare_job_runs
from pycwb.utils.module import import_function
import multiprocessing
import logging
import os

logger = logging.getLogger(__name__)


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
        logger.error(f"Error processing job segment: {job_seg}")
        logger.error(e)
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
    
    # use multiprocessing to process the job segments in parallel and prevent memory leak
    # the maxtasksperchild=1 should be removed if the memory leak is fixed in the future
    exceptions = []
    with multiprocessing.Pool(
        n_proc,
        maxtasksperchild=1,  # Restart workers after each task to avoid memory leaks
        ) as pool:
        logger.info(f"Starting {n_proc} workers")
        results = []
        for job_seg in job_segments:
            r = pool.apply_async(process_single_with_flag,
                                    args=(working_dir, config, job_seg, compress_json, overwrite))
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

    if exceptions:
        raise ExceptionGroup("JobFailure", exceptions)
    # for job_seg in job_segments:
    #     main_func(working_dir, config, job_seg, compress_json=compress_json)

    # client.close()