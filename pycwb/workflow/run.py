import copy
import logging

from pycwb.modules.logger import logger_init
from pycwb.workflow.subflow.prepare_job_runs import prepare_job_runs
from pycwb.utils.module import import_function
from pycwb.utils.parser import parse_id_string, parse_lag_string

logger = logging.getLogger(__name__)


def search(file_name, working_dir='.', overwrite=False, log_file=None, log_level="INFO",
           n_proc=1, plot=None, config_vars: str = None, input_dir=None,
           compress_json=False, dry_run=False,
           jobs: str = None, trial_idx: str = None, lags: str = None):
    # TODO: optimize the plot control
    logger_init(log_file, log_level)

    job_segments, config, working_dir = prepare_job_runs(working_dir, file_name, 1, dry_run, overwrite,
                                                         config_vars=config_vars, input_dir=input_dir,
                                                         plot=plot, compress_json=compress_json)

    if jobs is not None:
        job_ids = set(parse_id_string(jobs))
        job_segments = [seg for seg in job_segments if seg.index in job_ids]

    if lags is not None:
        lag_array = parse_lag_string(lags)
        for job_seg in job_segments:
            if job_seg.lag_file is not None:
                raise ValueError(
                    f"Job segment {job_seg.index}: --lags cannot be used when lag_file "
                    f"is already set in the configuration"
                )
            job_seg.lag_array = lag_array
            job_seg.lag_size = 0

    if dry_run:
        return job_segments

    # Log the active run options so they appear in the log for later debugging
    logger.info("Run options:")
    if jobs is not None:
        logger.info("  --jobs      : %s  (selected job indices: %s)", jobs,
                    sorted(seg.index for seg in job_segments))
    else:
        logger.info("  --jobs      : all (%d segments)", len(job_segments))
    if trial_idx is not None:
        logger.info("  --trial-idx : %s", trial_idx)
    else:
        logger.info("  --trial-idx : all (from segment injections)")
    if lags is not None:
        logger.info("  --lags      : %s  (%d lag vectors)", lags,
                    len(job_segments[0].lag_array) if job_segments else 0)
    else:
        logger.info("  --lags      : from config (lag_size=%s)",
                    job_segments[0].lag_size if job_segments else "N/A")

    trial_ids = parse_id_string(trial_idx) if trial_idx is not None else None

    logger.info(f"Loading segment processer: {config.segment_processer}")
    main_func = import_function(config.segment_processer)
    logger.info(f"Segment processer loaded: {main_func}")

    if len(job_segments) == 0:
        logger.warning("No job segments to process")
        return

    for job_seg in job_segments:
        if trial_ids is not None:
            for tid in trial_ids:
                sub_job_seg = copy.deepcopy(job_seg)
                sub_job_seg.trial_idx = tid
                if job_seg.injections:
                    sub_job_seg.injections = [inj for inj in job_seg.injections
                                              if inj.get('trial_idx', 0) == tid]
                logger.info(f"Processing job segment {job_seg.index}, trial_idx={tid}")
                main_func(working_dir, config, sub_job_seg, compress_json=compress_json)
        else:
            logger.info(f"Processing job segment {job_seg.index}")
            main_func(working_dir, config, job_seg, compress_json=compress_json)