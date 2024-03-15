import os
import shutil
from datetime import datetime
from pycwb.types.job import WaveSegment
import filecmp


def create_working_directory(working_dir: str) -> None:
    working_dir = os.path.abspath(working_dir)
    if not os.path.exists(working_dir):
        print(f"Creating working directory: {working_dir}")
        os.makedirs(working_dir)


def check_if_output_exists(working_dir: str, output_dir: str, overwrite: bool = False) -> None:
    output_dir = f"{working_dir}/{output_dir}"
    if os.path.exists(output_dir) and os.listdir(output_dir):
        if overwrite:
            print(f"Overwrite output directory {output_dir}")
        else:
            print(f"Output directory {output_dir} is not empty")
            raise ValueError(f"Output directory {output_dir} is not empty, please set overwrite to True "
                             f"if you want to overwrite it.")


def create_output_directory(working_dir: str, output_dir: str, log_dir: str, user_parameter_file: str) -> None:
    # create folder for output and log
    print(f"Output folder: {working_dir}/{output_dir}")
    print(f"Log folder: {working_dir}/{log_dir}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if os.path.exists(f"{output_dir}/user_parameters.yaml"):
        # check if the files are the same with md5
        if not filecmp.cmp(user_parameter_file, f"{output_dir}/user_parameters.yaml"):
            print(f"Old user_parameters.yaml file is different from the new one.")
            # rename the old user parameter file to user_parameters_old_{date}.yaml
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            shutil.move(f"{output_dir}/user_parameters.yaml", f"{output_dir}/user_parameters_old_{timestamp}.yaml")
            print(f"Old user_parameters.yaml file is renamed to user_parameters_old_{timestamp}.yaml")

    shutil.copyfile(user_parameter_file, f"{output_dir}/user_parameters.yaml")


def check_MRACatalog_setting() -> bool:
    if not os.environ.get('HOME_WAT_FILTERS'):
        print("HOME_WAT_FILTERS is not set.")
        print("Please download the latest version of cwb config "
              "and set HOME_WAT_FILTERS to the path of folder XTALKS.")
        print("Make sure you have installed git lfs before cloning the repository.")
        print("For example:")
        print("    git lfs install")
        print("    git clone https://gitlab.com/gwburst/public/config_o3")
        print("    export HOME_WAT_FILTERS=$(pwd)/config_o3/XTALKS")
        raise ValueError("HOME_WAT_FILTERS is not set.")
    return True


def print_job_info(job_seg: WaveSegment) -> None:
    job_id = job_seg.index
    print(f"Job ID: {job_id}")
    print(f"Start time: {job_seg.start_time}")
    print(f"End time: {job_seg.end_time}")
    print(f"Duration: {job_seg.end_time - job_seg.start_time}")
    print(f"Frames: {job_seg.frames}")
    print(f"Noise: {job_seg.noise}")
    print(f"Injections: {job_seg.injections}")
