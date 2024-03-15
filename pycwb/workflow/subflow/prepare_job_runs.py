import os

from pycwb.config import Config
from pycwb.modules.catalog import create_catalog
from pycwb.modules.job_segment import create_job_segment_from_config
from pycwb.modules.web_viewer.create import create_web_viewer
from pycwb.modules.workflow_utils.job_setup import create_working_directory, \
    check_if_output_exists, create_output_directory, \
    check_MRACatalog_setting


def prepare_job_runs(working_dir, config_file, n_proc=1, dry_run=False, overwrite=False):
    # convert to absolute path in case the current working directory is changed
    working_dir = os.path.abspath(working_dir)
    file_name = os.path.abspath(config_file)

    # create working directory and change the current working directory to the given working directory
    create_working_directory(working_dir)
    os.chdir(working_dir)

    # check environment
    check_MRACatalog_setting()

    # read user parameters
    config = Config(file_name)

    job_segments = create_job_segment_from_config(config)

    if not dry_run:
        print(f"Number of jobs: {len(job_segments)}")
        # override n_proc in config
        if n_proc != 0:
            config.nproc = n_proc

        check_if_output_exists(working_dir, config.outputDir, overwrite)
        create_output_directory(working_dir, config.outputDir, config.logDir, file_name)

        create_catalog(f"{working_dir}/{config.outputDir}/catalog.json", config, job_segments)
        create_web_viewer(f"{working_dir}/{config.outputDir}")

    # slags = job_generator(len(config.ifo), config.slagMin, config.slagMax, config.slagOff, config.slagSize)
    return job_segments, config, working_dir
