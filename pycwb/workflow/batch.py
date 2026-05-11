import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import faulthandler
import os
import getpass
import logging
import sys
from typing import Any
from pycwb.modules.logger import logger_init
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

    if not cluster:
        cluster = config.cluster
    if not conda_env:
        conda_env = config.conda_env
    if not additional_init:
        additional_init = config.additional_init
    if not accounting_group:
        accounting_group = config.accounting_group
    if not job_per_worker:
        job_per_worker = config.job_per_worker
    if not n_proc:
        n_proc = config.nproc
    if not memory:
        memory = config.job_memory
    if not disk:
        disk = config.job_disk
    if not container_image:
        container_image = config.container_image
    if not should_transfer_files:
        should_transfer_files = config.should_transfer_files
    if not walltime:
        walltime = config.job_walltime
    if not slurm_constraint:
        slurm_constraint = getattr(config, 'slurm_constraint', None) or slurm_constraint
    if not slurm_partition:
        slurm_partition = getattr(config, 'slurm_partition', None) or slurm_partition
    if n_retries == 5:
        n_retries = getattr(config, 'n_retries', 5)  # 5 is the default; prefer config if set

    logger.info("Job submission info:")
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


def _cleanup_stale_lock(lock_path: str) -> None:
    """Remove a stale SoftFileLock file if the holder process is no longer alive.

    ``SoftFileLock`` stores the holder's PID (one per line) in the lock file.
    On the same host we can verify liveness via ``kill(pid, 0)``.  If the
    process is gone the lock is stale and safe to remove.  On a different host
    the PID is meaningless, so we also remove the lock if the hostname stored
    in the file does not match the current host — the original holder cannot
    possibly still be running on this machine.
    """
    logger_init(log_file=None, log_level="INFO", worker_prefix='LockCleanup')

    import socket
    if not os.path.exists(lock_path):
        return
    try:
        with open(lock_path) as fh:
            lines = [ln.strip() for ln in fh.readlines() if ln.strip()]
        # SoftFileLock format: "<pid>\n<hostname>" (one entry per line pair)
        # Older versions wrote only the PID; handle both.
        pid = None
        hostname = None
        if lines:
            try:
                pid = int(lines[0])
            except ValueError:
                pass
        if len(lines) >= 2:
            hostname = lines[1]

        current_host = socket.gethostname()
        stale = False

        if hostname and hostname != current_host:
            # Lock was held by a process on a different node — always stale here.
            stale = True
        elif pid is not None:
            try:
                os.kill(pid, 0)  # signal 0: just check existence
            except ProcessLookupError:
                stale = True  # process does not exist
            except PermissionError:
                pass  # process exists but owned by another user — not stale
        else:
            # Cannot determine holder; remove to be safe at startup.
            stale = True

        if stale:
            os.remove(lock_path)
            logger.warning("Removed stale lock file: %s (pid=%s, host=%s)", lock_path, pid, hostname)
    except Exception as exc:
        logger.warning("Could not inspect lock file %s: %s", lock_path, exc)


def processor_wrapper(main_func, queue, working_dir, config, job_seg, compress_json=True, catalog_file=None, skip_lags: dict[int, set[int]] | None = None):
    logger_init(log_file=f"{working_dir}/log/job_{job_seg.index}.log", log_level="INFO")
    logger.info(f"Processing job segment {job_seg.index} with {getpass.getuser()} on {multiprocessing.current_process()}")
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
    logger_init(log_file=None, log_level="INFO", worker_prefix=f'BatchRun-{jobs or "all"}')

    # ------------------------------------------------------------------ #
    # 1. Load configuration and job segments                               #
    # ------------------------------------------------------------------ #
    job_segments, config, working_dir, catalog_file = load_batch_run(working_dir, config_file, jobs,
                                                       n_proc=n_proc, compress_json=compress_json)
    logger_init(log_file, log_level)

    # ------------------------------------------------------------------ #
    # 1b. Pre-flight stale lock cleanup                                    #
    #     Must run before any subprocess spawns so there are no racing     #
    #     writers when we inspect or delete lock files.                    #
    # ------------------------------------------------------------------ #
    if catalog_file:
        for _lock in (catalog_file + ".lock",
                      catalog_file.replace("catalog", "progress", 1) + ".lock"):
            _cleanup_stale_lock(_lock)

    logger.info(f"Loading segment processer: {config.segment_processer}")
    main_func = import_function(config.segment_processer)
    logger.info(f"Segment processer loaded: {main_func}")

    # ------------------------------------------------------------------ #
    # 2. Pre-flight resume check (serial, before any subprocess spawns)   #
    #    - Skip already-complete jobs                                      #
    #    - Remove stale triggers from interrupted runs                     #
    #    - Build skip_lags_map so workers resume at the right lag          #
    # ------------------------------------------------------------------ #
    pending_segments = []
    skip_lags_map = {}  # {job_index: {trial_idx: {lag_idx, ...}}}
    if catalog_file and os.path.exists(catalog_file):
        try:
            from pycwb.modules.catalog.catalog import Catalog
            cat = Catalog.open(catalog_file)
            for job_seg in job_segments:
                # injections is None  → non-simulation run (field never populated)
                # injections == []    → simulation run, but no injections overlap this segment
                # injections == [...] → simulation run with injections in this segment
                if job_seg.injections is not None:
                    expected_trial_idxs = (
                        set(inj.get('trial_idx', 0) for inj in job_seg.injections) or {0}
                    )
                    # Simulation mode: if no injections overlap this segment and
                    # analyze_injection_only is set, there is no valid time window to
                    # analyse — skip the job entirely rather than running a vacuous trial.
                    if not job_seg.injections and getattr(config, 'analyze_injection_only', False):
                        logger.info(
                            "Job segment %d has no injections in window and "
                            "analyze_injection_only is set, skipping",
                            job_seg.index,
                        )
                        continue
                else:
                    expected_trial_idxs = {0}

                completed = cat.get_completed_lags(job_seg.index)

                # check if all expected trial_idx and lag_idx combinations are marked complete; if so, skip the job entirely
                if completed and all(
                    tid in completed and len(completed[tid]) == job_seg.n_lag
                    for tid in expected_trial_idxs
                ):
                    logger.info(
                        "Job segment %d is already complete (%d lags × %d trials), skipping",
                        job_seg.index, job_seg.n_lag, len(expected_trial_idxs),
                    )
                    continue

                n_removed = cat.remove_stale_triggers(job_seg.index, completed)
                if n_removed:
                    logger.info("Removed %d stale trigger(s) for job %d before resume",
                                n_removed, job_seg.index)

                if completed:
                    skip_lags = {tid: lags for tid, lags in completed.items() if lags}
                    if skip_lags:
                        total_skipped = sum(len(v) for v in skip_lags.values())
                        logger.info("Rescue: skipping %d lag-trial pairs for job %d",
                                    total_skipped, job_seg.index)
                        skip_lags_map[job_seg.index] = skip_lags

                pending_segments.append(job_seg)
        except Exception as e:
            logger.warning("Could not read progress for pre-flight resume check: %s", e)
            pending_segments = list(job_segments)
    else:
        pending_segments = list(job_segments)

    if not pending_segments:
        logger.info("All job segments are already complete, nothing to do.")
        return

    # ------------------------------------------------------------------ #
    # 3. Start the data-collector process                                  #
    #    Single writer that serialises all catalog/wave I/O from workers. #
    # ------------------------------------------------------------------ #
    manager = multiprocessing.Manager()
    queue = manager.Queue()
    data_collector_worker = multiprocessing.Process(
        target=data_collector, args=(working_dir, config, catalog_file, queue)
    )
    data_collector_worker.start()

    # ------------------------------------------------------------------ #
    # 4. Dispatch job segments to the process pool                        #
    #    - max_tasks_per_child=1: restart each worker after one job to    #
    #      avoid memory leaks from JAX/Numba state accumulation           #
    #    - initializer: enable faulthandler so SIGSEGV prints a traceback #
    # ------------------------------------------------------------------ #
    exceptions = []
    executor = ProcessPoolExecutor(
        max_workers=n_workers,
        max_tasks_per_child=1,
        initializer=_worker_initializer,
    )
    try:
        logger.info(f"Starting {n_workers} workers")
        futures = {
            executor.submit(processor_wrapper, main_func, queue, working_dir, config, job_seg,
                            compress_json, catalog_file, skip_lags_map.get(job_seg.index)): job_seg
            for job_seg in pending_segments
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
        # ------------------------------------------------------------------ #
        # 5. Tear down the pool                                               #
        #    Explicitly terminate workers: the executor's management thread   #
        #    is non-daemon and can hang waiting on a pipe from a dead worker  #
        #    (e.g. killed by SIGSEGV). Terminating unblocks it cleanly.      #
        # ------------------------------------------------------------------ #
        print("Terminating worker processes...")
        for proc in list(executor._processes.values()):
            try:
                proc.terminate()
            except Exception:
                pass
        executor.shutdown(wait=False, cancel_futures=True)
        if exceptions:
            os._exit(1)  # hard exit: skip cleanup to avoid hanging on broken state

    # ------------------------------------------------------------------ #
    # 6. Drain the data-collector and wait for it to finish               #
    # ------------------------------------------------------------------ #
    logger.info("All jobs are done, waiting for data collector to finish")
    queue.put(None)  # sentinel: signals the collector to flush and exit
    data_collector_worker.join()

    if exceptions:
        raise ExceptionGroup("JobFailure", exceptions)
