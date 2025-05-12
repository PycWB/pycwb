import os
import logging
import shutil
from datetime import datetime
from pycwb.types.job import WaveSegment
import filecmp


logger = logging.getLogger(__name__)

def create_working_directory(working_dir: str) -> None:
    working_dir = os.path.abspath(working_dir)
    if not os.path.exists(working_dir):
        logger.info(f"Creating working directory: {working_dir}")
        os.makedirs(working_dir)


def check_if_output_exists(working_dir: str, output_dir: str, overwrite: bool = False) -> None:
    output_dir = f"{working_dir}/{output_dir}"
    logger.info(f"Checking if output directory exist: {output_dir}")
    if os.path.exists(output_dir) and os.listdir(output_dir):
        if overwrite:
            logger.info(f"Overwrite output directory {output_dir}")
        else:
            logger.info(f"Output directory {output_dir} is not empty")
            raise ValueError(f"Output directory {output_dir} is not empty, please set overwrite to True "
                             f"if you want to overwrite it.")


def create_output_directory(working_dir: str, output_dir: str, log_dir: str, catalog_dir: str,
                            trigger_dir: str, user_parameter_file: str) -> None:
    # create folder for output and log
    config_dir = f"{working_dir}/config"
    input_dir = f"{working_dir}/input"
    job_status_dir = f"{working_dir}/job_status"
    public_dir = f"{working_dir}/public"
    logger.info(f"Output folder: {working_dir}/{output_dir}")
    logger.info(f"Trigger folder: {working_dir}/{trigger_dir}")
    logger.info(f"Log folder: {working_dir}/{log_dir}")
    logger.info(f"Config folder: {config_dir}")
    logger.info(f"Catalog folder: {working_dir}/{catalog_dir}")
    logger.info(f"Job status folder: {job_status_dir}")
    logger.info(f"Public folder: {public_dir}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)
    if not os.path.exists(catalog_dir):
        os.makedirs(catalog_dir)
    if not os.path.exists(trigger_dir):
        os.makedirs(trigger_dir)
    if not os.path.exists(job_status_dir):
        os.makedirs(job_status_dir)
    if not os.path.exists(public_dir):
        os.makedirs(public_dir)
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)

    if os.path.exists(f"{config_dir}/user_parameters.yaml"):
        # check if the files are the same with md5, if not, backup the old file
        if not filecmp.cmp(user_parameter_file, f"{config_dir}/user_parameters.yaml"):
            logger.info(f"Old user_parameters.yaml file is different from the new one.")
            # rename the old user parameter file to user_parameters_old_{date}.yaml
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            shutil.move(f"{config_dir}/user_parameters.yaml", f"{config_dir}/user_parameters_old_{timestamp}.yaml")
            logger.info(f"Old user_parameters.yaml file is renamed to user_parameters_old_{timestamp}.yaml")
    else:
        shutil.copyfile(user_parameter_file, f"{config_dir}/user_parameters.yaml")


def check_MRACatalog_setting() -> bool:
    if not os.environ.get('HOME_WAT_FILTERS'):
        logger.info("HOME_WAT_FILTERS is not set.")
        logger.info("Please download the latest version of cwb config "
              "and set HOME_WAT_FILTERS to the path of folder XTALKS.")
        logger.info("Make sure you have installed git lfs before cloning the repository.")
        logger.info("For example:")
        logger.info("    git lfs install")
        logger.info("    git clone https://gitlab.com/gwburst/public/config_o3")
        logger.info("    export HOME_WAT_FILTERS=$(pwd)/config_o3/XTALKS")
        raise ValueError("HOME_WAT_FILTERS is not set.")
    return True


def print_job_info(job_seg: WaveSegment) -> None:
    job_id = job_seg.index
    logger.info(f"Job ID: {job_id}")
    logger.info(f"Start time: {job_seg.start_time}")
    logger.info(f"End time: {job_seg.end_time}")
    logger.info(f"Duration: {job_seg.end_time - job_seg.start_time}")
    logger.info(f"Frames: {job_seg.frames}")
    logger.info(f"Noise: {job_seg.noise}")
    logger.info(f"Injections: {job_seg.injections}")
