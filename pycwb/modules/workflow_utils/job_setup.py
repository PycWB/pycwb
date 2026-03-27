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
    logger.info("Job ID:           %s", job_seg.index)
    logger.info("Analyze window:   [%.1f, %.1f]  (duration %.1f s)",
                job_seg.analyze_start, job_seg.analyze_end, job_seg.duration)
    logger.info("Padded window:    [%.1f, %.1f]  (duration %.1f s, seg_edge=%.1f s)",
                job_seg.padded_start, job_seg.padded_end, job_seg.padded_duration, job_seg.seg_edge)
    logger.info("Frames:           %s", job_seg.frames)
    logger.info("Noise:            %s", job_seg.noise)
    logger.info("Injections:       %s", job_seg.injections)


def print_node_info() -> None:
    """Log CPU model, architecture, frequency, core counts, and available memory."""
    import platform
    import subprocess
    import psutil

    node = os.uname().nodename
    arch = platform.machine()

    # CPU model — platform.processor() is unreliable on some Linux/macOS builds,
    # so fall back to OS-specific sources.
    cpu_model = platform.processor() or ""
    if not cpu_model:
        try:
            if platform.system() == "Darwin":
                cpu_model = subprocess.check_output(
                    ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
                ).strip()
            elif platform.system() == "Linux":
                with open("/proc/cpuinfo") as _f:
                    for _line in _f:
                        if _line.startswith("model name"):
                            cpu_model = _line.split(":", 1)[1].strip()
                            break
        except Exception:
            cpu_model = "unknown"

    freq = psutil.cpu_freq()
    if freq:
        freq_str = (f"{freq.current:.0f} MHz  (max {freq.max:.0f} MHz)"
                    if freq.max else f"{freq.current:.0f} MHz")
    else:
        freq_str = "N/A"

    cpu_phys    = psutil.cpu_count(logical=False) or psutil.cpu_count()
    cpu_logical = psutil.cpu_count(logical=True)
    mem         = psutil.virtual_memory()

    logger.info("============================================")
    logger.info("Node:        %s", node)
    logger.info("CPU model:   %s", cpu_model)
    logger.info("Arch:        %s", arch)
    logger.info("CPUs:        %d physical / %d logical", cpu_phys, cpu_logical)
    logger.info("Frequency:   %s", freq_str)
    logger.info("Memory:      %.1f GB total / %.1f GB available",
                mem.total / 1024 ** 3, mem.available / 1024 ** 3)
    logger.info("============================================")
