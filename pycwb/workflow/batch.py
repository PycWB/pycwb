import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import faulthandler
import os
import getpass
import logging
import sys
from pathlib import Path
from typing import Any
from pycwb.modules.logger import logger_init, log_prints
from pycwb.workflow.subflow.prepare_job_runs import prepare_job_runs, load_batch_run
from pycwb.utils.module import import_function
from pycwb.modules.condor.condor import HTCondor
from pycwb.modules.slurm.slurm import Slurm

# ExceptionGroup is available in Python 3.11+; use backport for earlier versions
if sys.version_info >= (3, 11):
    from builtins import ExceptionGroup
else:
    from exceptiongroup import ExceptionGroup

logger = logging.getLogger(__name__)


def batch_setup(file_name, working_dir='.',
                overwrite=False, log_file=None, log_level="INFO",
                compress_json=True, cluster=None, conda_env=None, additional_init="",
                accounting_group=None, job_per_worker=None, n_proc=None, memory=None, disk=None,
                container_image=None, should_transfer_files=False,
                walltime=None, slurm_constraint=None, slurm_partition=None, n_retries=5,
                config_vars: str = None, input_dir=None,
                dry_run=False, submit=False):
    logger_init(log_file, log_level)
    job_segments, config, working_dir = prepare_job_runs(working_dir, file_name, n_proc, dry_run, overwrite,
                                                         config_vars=config_vars, input_dir=input_dir,
                                                         compress_json=compress_json)

    if dry_run:
        return job_segments

    if not cluster: cluster = config.cluster
    if not conda_env: conda_env = config.conda_env
    if not additional_init: additional_init = config.additional_init
    if not accounting_group: accounting_group = config.accounting_group
    if not job_per_worker: job_per_worker = config.job_per_worker
    if not n_proc: n_proc = config.nproc
    if not memory: memory = config.job_memory
    if not disk: disk = config.job_disk
    if not container_image: container_image = config.container_image
    if not should_transfer_files: should_transfer_files = config.should_transfer_files
    if not walltime: walltime = config.job_walltime
    if not slurm_constraint: slurm_constraint = getattr(config, 'slurm_constraint', None) or slurm_constraint
    if not slurm_partition: slurm_partition = getattr(config, 'slurm_partition', None) or slurm_partition
    if n_retries == 5: n_retries = getattr(config, 'n_retries', 5)  # 5 is the default; prefer config if set

    logger.info(f"Job submission info:")
    logger.info(f"  Cluster type: {cluster}")
    logger.info(f"  Conda environment: {conda_env}")
    logger.info(f"  Additional init script: {additional_init}")
    logger.info(f"  Accounting group: {accounting_group}")
    logger.info(f"  Jobs per worker: {job_per_worker}")
    logger.info(f"  Number of processors per job: {n_proc}")
    logger.info(f"  Memory per job: {memory}")
    logger.info(f"  Disk per job: {disk}")
    logger.info(f"  Container image: {container_image}")
    logger.info(f"  Should transfer files: {should_transfer_files} (will be set true of image is defined)")
    
    if cluster == "condor":
        condor = HTCondor(working_dir, conda_env, additional_init, accounting_group, job_per_worker,
                          container_image, should_transfer_files,
                          n_proc, memory, disk, n_retries=n_retries)
        condor.create(job_segments, submit=submit)
    elif cluster == "slurm":
        slurm = Slurm(working_dir, conda_env, additional_init, job_per_worker,
                      n_proc, memory, disk,
                      time=walltime, constraint=slurm_constraint,
                      partition=slurm_partition, n_retries=n_retries)
        slurm.create(job_segments, submit=submit)
    else:
        raise ValueError(f"Unsupported cluster type: {cluster}, only support condor and slurm")


def _worker_initializer():
    """Enable faulthandler in each worker to capture segmentation violations."""
    faulthandler.enable()


def data_collector(working_dir: str, config: Any, catalog_file: str, queue: Any) -> None:
    """Worker process that serializes all persistent writes from job workers.

    Handles three message types via the shared queue:

    - ``{"type": "progress", ...}`` → append to progress Parquet file
    - ``{"type": "trigger", "trigger": Trigger}`` → batch and write to catalog Parquet
    - ``{"type": "wave", ...}`` → write to HDF5 wave file
    - ``None`` → sentinel to flush buffers and terminate

    Parameters
    ----------
    working_dir : str
        Root working directory for the run.
    config : Config
        Parsed pycWB configuration object.
    catalog_file : str
        Absolute path to the catalog Parquet file.
    queue : multiprocessing.Queue
        Shared queue carrying typed message dicts from worker processes.
    """
    logger_init(log_file=None, log_level="INFO", worker_prefix='DataCollector')
    from pycwb.modules.catalog.catalog import Catalog
    from pycwb.workflow.subflow.postprocess_and_plots import add_wf_to_wave

    stats = {"progress": 0, "trigger": 0, "wave": 0}
    trigger_buffer = []
    TRIGGER_FLUSH_SIZE = 50

    def flush_triggers():
        if not trigger_buffer:
            return
        try:
            cat = Catalog.open(catalog_file)
            cat.add_triggers(list(trigger_buffer))
            logger.info("Flushed %d triggers to catalog", len(trigger_buffer))
        except Exception as e:
            logger.error("Error flushing %d triggers: %s", len(trigger_buffer), e)
        trigger_buffer.clear()

    while True:
        item = queue.get()
        if item is None:
            flush_triggers()
            break

        try:
            msg_type = item["type"]

            if msg_type == "progress":
                # Flush pending triggers before writing progress so the
                # catalog is up-to-date when the lag is marked complete.
                flush_triggers()
                record = {k: v for k, v in item.items() if k != "type"}
                cat = Catalog.open(catalog_file)
                cat.add_lag_progress(**record)
                stats["progress"] += 1
                logger.info(
                    "Progress: job=%d trial=%d lag=%d triggers=%d livetime=%.2fs [%d total]",
                    record["job_id"], record["trial_idx"], record["lag_idx"],
                    record["n_triggers"], record["livetime"], stats["progress"],
                )

            elif msg_type == "trigger":
                trigger_buffer.append(item["trigger"])
                stats["trigger"] += 1
                if len(trigger_buffer) >= TRIGGER_FLUSH_SIZE:
                    flush_triggers()

            elif msg_type == "wave":
                add_wf_to_wave(config, item["wave_file"], item["event_id"], item["waves"])
                stats["wave"] += 1
                logger.info("Wave: event=%s → %s [%d total]",
                            item["event_id"], item["wave_file"], stats["wave"])

            else:
                logger.warning("Unknown message type: %s", msg_type)

        except Exception as e:
            logger.error("Error processing %s message: %s", item.get("type", "?"), e)

    logger.info("DataCollector finished: %d progress, %d triggers, %d waves",
                stats["progress"], stats["trigger"], stats["wave"])


def process_single_with_resume(main_func, queue, working_dir, config, job_seg, compress_json=True, catalog_file=None):
    logger_init(log_file=f"{working_dir}/log/job_{job_seg.index}.log", log_level="INFO")
    logger.info(f"Processing job segment {job_seg.index} with {getpass.getuser()} on {multiprocessing.current_process()}")

    # Determine expected trial indices (mirrors logic in process_job_segment_*.py)
    if job_seg.injections:
        expected_trial_idxs = set(inj.get('trial_idx', 0) for inj in job_seg.injections)
    else:
        expected_trial_idxs = {0}
    n_lag = job_seg.n_lag

    # Build skip_lags from progress file for per-lag rescue; also check completeness.
    skip_lags = None
    if catalog_file and os.path.exists(catalog_file):
        try:
            from pycwb.modules.catalog.catalog import Catalog
            cat = Catalog.open(catalog_file)
            completed = cat.get_completed_lags(job_seg.index)

            # Check if every expected trial has all n_lag lags done → job is complete
            if completed and all(
                tid in completed and len(completed[tid]) == n_lag
                for tid in expected_trial_idxs
            ):
                logger.info(
                    "Job segment %d is already complete (%d lags × %d trials)",
                    job_seg.index, n_lag, len(expected_trial_idxs),
                )
                return

            # Remove triggers written by incomplete lag-trial pairs from a prior
            # interrupted run so they are not duplicated when those lags re-run.
            # Pass completed even when empty: an empty dict means nothing finished,
            # so every trigger for this job is stale and must be purged.
            n_removed = cat.remove_stale_triggers(job_seg.index, completed)
            if n_removed:
                logger.info(
                    "Removed %d stale trigger(s) for job %d before resume",
                    n_removed, job_seg.index,
                )

            # Build per-trial skip sets: {trial_idx: {lag_idx, ...}}
            # Only trials that have at least one completed lag get an entry.
            if completed:
                skip_lags = {tid: lags for tid, lags in completed.items() if lags}
                total_skipped = sum(len(v) for v in skip_lags.values())
                logger.info("Rescue: skipping %d lag-trial pairs for job %d",
                            total_skipped, job_seg.index)
        except Exception as e:
            logger.warning("Could not read progress for rescue: %s", e)

    try:
        main_func(working_dir, config, job_seg,
                  compress_json=compress_json, catalog_file=catalog_file, queue=queue,
                  skip_lags=skip_lags)
    except Exception as e:
        logger.error(f"Error processing job segment: {job_seg}")
        logger.error(e)
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
    data_collector_worker = multiprocessing.Process(target=data_collector, args=(working_dir, config, catalog_file, queue))
    data_collector_worker.start()

    # Use ProcessPoolExecutor for modern parallel processing with segfault capture via faulthandler.
    # max_tasks_per_child=1 restarts workers after each task to avoid memory leaks (requires Python 3.11+).
    # _worker_initializer enables faulthandler so segmentation violations print a native traceback
    # to stderr instead of silently terminating the worker.
    exceptions = []
    executor = ProcessPoolExecutor(
        max_workers=n_workers,
        max_tasks_per_child=1,  # Restart workers after each task to avoid memory leaks
        initializer=_worker_initializer,
    )
    try:
        logger.info(f"Starting {n_workers} workers")
        futures = {
            executor.submit(process_single_with_resume, main_func, queue, working_dir, config, job_seg,
                            compress_json, catalog_file): job_seg
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
    finally:
        # Explicitly terminate all worker processes before shutdown.
        # ProcessPoolExecutor's internal management thread is non-daemon and can hang
        # indefinitely waiting on a pipe from workers killed by a signal (e.g. SIGSEGV).
        # Terminating the processes unblocks the management thread so it can exit cleanly.
        print("Terminating worker processes...")
        for proc in list(executor._processes.values()):
            try:
                proc.terminate()
            except Exception:
                pass
        executor.shutdown(wait=False, cancel_futures=True)
        if exceptions:
            os._exit(1)  # nuclear option: skip ExceptionGroup, just die

    logger.info("All jobs are done, waiting for data collector to finish")

    # send a sentinel to terminate the data collector
    queue.put(None)
    data_collector_worker.join()

    if exceptions:
        raise ExceptionGroup("JobFailure", exceptions)
