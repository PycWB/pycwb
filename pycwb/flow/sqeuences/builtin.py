import asyncio
import os

from ..tasks.builtin import check_env, read_config, create_working_directory, \
    check_if_output_exists, create_output_directory, create_job_segment, \
    create_catalog_file, create_web_dir, load_xtalk_catalog


def prepare_job_runs(working_dir, config_file, n_proc=1, dry_run=False, overwrite=False):
    # convert to absolute path in case the current working directory is changed
    working_dir = os.path.abspath(working_dir)
    file_name = os.path.abspath(config_file)

    # create working directory and change the current working directory to the given working directory
    create_working_directory(working_dir)

    # check environment
    check_env()

    # read user parameters
    config = read_config(file_name)

    job_segments = create_job_segment(config)

    if not dry_run:
        # override n_proc in config
        if n_proc != 0:
            config.nproc = n_proc

        check_if_output_exists(working_dir, config.outputDir, overwrite)
        create_output_directory(working_dir, config.outputDir, config.logDir, file_name)

        create_catalog_file(working_dir, config, job_segments)
        create_web_dir(working_dir, config.outputDir)
        load_xtalk_catalog(config.MRAcatalog)

    # slags = job_generator(len(config.ifo), config.slagMin, config.slagMax, config.slagOff, config.slagSize)
    return job_segments, config
