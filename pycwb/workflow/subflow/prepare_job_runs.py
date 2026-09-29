import os
import uuid
import tempfile
import shutil
import logging
from typing import List
from dacite import from_dict, Config as DaciteConfig
from jinja2 import Template 
from pycwb.config.processing import check_recorded_execution_profile
from pycwb.config import Config
from pycwb.modules.catalog import Catalog, read_catalog_metadata
from pycwb.modules.job_segment import create_job_segment_from_config
from pycwb.modules.workflow_utils.job_setup import create_working_directory, \
    check_if_output_exists, create_output_directory
from pycwb.types.job import WaveSegment
from pycwb.utils.parser import parse_id_string, parse_vars
from .config_consistency import validate_run_config

logger = logging.getLogger(__name__)


def overwrite_config(config: Config, n_proc: int = None, plot_trigger: bool = None, save_waveform: bool = None,
                     plot_waveform: bool = None, save_sky_map: bool = None, plot_sky_map: bool = None,
                     compress_output_json: bool = None) -> Config:
    """
    This is a helper function for the CLI to overwrite certain keys in the config object.

    :param config: The config object to be modified
    :param n_proc: The number of cores to use
    :param plot_trigger: The switch on plotting the trigger's likelihood map and null map
    :param save_waveform: The switch on saving the reconstructed waveforms in txt file
    :param plot_waveform: The switch on plotting the reconstructed waveforms
    :param save_sky_map: The switch on saving the reconstructed skymap in json file
    :param plot_sky_map: The switch on plotting the reconstructed skymap
    :param compress_output_json: Whether to compress the output json
    :return: Config
    """
    if n_proc is not None:
        config.nproc = n_proc
    if plot_trigger is not None:
        config.plot_trigger = plot_trigger
    if save_waveform is not None:
        config.save_waveform = save_waveform
    if plot_waveform is not None:
        config.plot_waveform = plot_waveform
    if save_sky_map is not None:
        config.save_sky_map = save_sky_map
    if plot_sky_map is not None:
        config.plot_sky_map = plot_sky_map
    if compress_output_json is not None:
        config.compress_output_json = compress_output_json
    return config


def prepare_job_runs(working_dir: str, config_file: str, n_proc: int = 1,
                     dry_run: bool = False, overwrite: bool = False,
                     config_vars: str = None, input_dir: str = None,
                     plot: bool = None, compress_json: bool = None) -> tuple[list[WaveSegment], Config, str]:
    """
    This is the helper function to create the run directories, create catalog file,
    make a copy of user parameter file, and generate the job segments from the Config.
    It also provides several check to prevent override of existing run.

    :param working_dir: The working dirs for the run
    :param config_file: The path of user parameter YAML file
    :param n_proc: The number of processes to use, will overwrite the setting the YAML file
    :param dry_run: If set true, only the working directory and xtalk will be created.
    :param overwrite: If set true, the previous run will be overwritten
    :param plot: If set true, all the output settings will be switched on
    :param compress_json: If set true, the output json will be compressed
    :return: tuple[list[WaveSegment], Config, str]
    """
    # convert to absolute path in case the current working directory is changed
    working_dir = os.path.abspath(working_dir)
    file_name = os.path.abspath(config_file)
    input_dir = os.path.abspath(input_dir) if input_dir else None

    # create working directory and change the current working directory to the given working directory
    create_working_directory(working_dir)
    os.chdir(working_dir)

    # check environment
    # check_MRACatalog_setting()

    # if input_dir is given, copy the input files to the working directory
    if input_dir is not None:
        copy_input_files(input_dir, working_dir)

    # if config_vars is geven, parse it and update the config as a template
    if config_vars is not None:
        file_name = generate_config(file_name, config_vars)

    validate_run_config(file_name, working_dir)

    # read user parameters
    config = Config()
    config.load_from_yaml(file_name)

    job_segments = create_job_segment_from_config(config)
    # slags = generate_slags(len(config.ifo), config.slagMin, config.slagMax, config.slagOff, config.slagSize)

    if not dry_run:
        logger.info(f"Number of jobs: {len(job_segments)}")
        # override n_proc in config
        config = overwrite_config(config, n_proc=n_proc, save_waveform=plot, save_sky_map=plot,
                                  plot_trigger=plot, plot_waveform=plot, plot_sky_map=plot,
                                  compress_output_json=compress_json)

        catalog_file = f"{working_dir}/{config.catalog_dir}/{Catalog.DEFAULT_FILENAME}"
        if os.path.exists(catalog_file):
            check_recorded_execution_profile(config, read_catalog_metadata(catalog_file)["config"])

        check_if_output_exists(working_dir, config.outputDir, overwrite)
        create_output_directory(working_dir, config.outputDir, config.logDir, config.catalog_dir,
                                config.trigger_dir, file_name)

        catalog_file = f"{working_dir}/{config.catalog_dir}/{Catalog.DEFAULT_FILENAME}"

        if not os.path.exists(catalog_file):
            Catalog.create(catalog_file, config, job_segments)

    return job_segments, config, working_dir


def load_batch_run(working_dir: str, config_file: str, jobs: str, compress_json: bool = True,
                   n_proc: int = 1, batch_id: str = None) -> tuple[List[WaveSegment], Config, str, str]:
    """
    This function provides the functionality to return the job segments with given jobs id/range.
    For example, the argument jobs can be 10-15 or 11,12 or even 10, 15-16.
    Only the required job segments will be returned. This function is mainly used for the batch runs

    :param working_dir: The working dirs for the run
    :param config_file: The path of user parameter YAML file
    :param jobs: the ids seperated by comma or a range with dash, such as 10-15 or 11,12 or even 10, 15-16
    :param compress_json: If set true, the output json will be compressed
    :param n_proc: The number of processes to use, will overwrite the setting the YAML file
    :return:
    """
    if jobs is not None and batch_id is not None:
        raise ValueError("Use either --jobs or --batch-id, not both")
    if batch_id is not None:
        from pycwb.workflow.execution.scheduling import validate_batch_id
        validate_batch_id(batch_id)
    job_ids = parse_id_string(jobs) if jobs is not None else None

    # convert to absolute path in case the current working directory is changed
    working_dir = os.path.abspath(working_dir)
    file_name = os.path.abspath(config_file)

    os.chdir(working_dir)

    # YAML is the runtime source of truth. Metadata is provenance and stores
    # the prepared job selection; it must never silently override the YAML.
    validate_run_config(file_name, working_dir, fragment_id=batch_id or jobs)
    config = Config()
    config.load_from_yaml(file_name)

    # Stable batch IDs always select their prepared fragment. Explicit --jobs
    # selections prefer the root catalog, with a fragment fallback for transfer mode.
    default_catalog_path = f'{config.catalog_dir}/{Catalog.DEFAULT_FILENAME}'
    fragment_id = batch_id or jobs
    per_job_catalog_path = f'{config.catalog_dir}/fragment/catalog_{fragment_id}{Catalog.DEFAULT_EXTENSION}'
    if batch_id is not None:
        if not os.path.exists(per_job_catalog_path):
            raise FileNotFoundError(
                f"Prepared batch fragment not found: {per_job_catalog_path}. "
                "Run batch-setup to prepare the catalog fragments."
            )
        catalog_meta_file = per_job_catalog_path
    elif os.path.exists(default_catalog_path):
        catalog_meta_file = default_catalog_path
    elif os.path.exists(per_job_catalog_path):
        catalog_meta_file = per_job_catalog_path
        logger.info(f"Root catalog not found; reading metadata from per-job fragment: {per_job_catalog_path}")
    else:
        raise FileNotFoundError(
            f"Catalog metadata not found: tried {default_catalog_path} and {per_job_catalog_path}"
        )
    catalog = read_catalog_metadata(catalog_meta_file)
    check_recorded_execution_profile(config, catalog['config'])
    logger.info("Loaded config from YAML: %s", file_name)
    job_segments = catalog['jobs']
    logger.info(f"Loaded {len(job_segments)} job segments from catalog")

    by_id = {job["index"]: job for job in job_segments}
    if job_ids is None:
        job_ids = list(by_id)
        jobs = ",".join(str(index) for index in job_ids)
    fragment_id = batch_id or jobs
    missing = set(job_ids) - by_id.keys()
    if missing:
        raise ValueError(f"Unknown job IDs: {sorted(missing)}")
    selected_job_segments = [from_dict(WaveSegment, by_id[index], config=DaciteConfig(cast=[tuple]))
                             for index in job_ids]
    config = overwrite_config(config, n_proc=n_proc or None, compress_output_json=compress_json)

    create_output_directory(working_dir, config.outputDir, config.logDir, config.catalog_dir,
                            config.trigger_dir, file_name)

    catalog_file = f"{working_dir}/{config.catalog_dir}/fragment/catalog_{fragment_id}{Catalog.DEFAULT_EXTENSION}"

    if not os.path.exists(catalog_file):
        # Fragments are transferred to execute nodes without the run-level
        # manifest, so keep their small selected job list inline.
        Catalog.create(catalog_file, config, selected_job_segments, jobs_in_metadata=True)

    return selected_job_segments, config, working_dir, catalog_file


def copy_input_files(input_dir: str, working_dir: str):
    """
    Copy input files from the input directory to the working directory.
    This is used to ensure that the input files are available in the working directory.

    :param input_dir: The directory containing the input files
    :param working_dir: The working directory where the input files will be copied
    """
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory {input_dir} does not exist.")
    logging.info(f"Copying input files from {input_dir} to {working_dir}")
    # copy the input directory to the working directory
    input_dir_name = os.path.basename(input_dir)
    input_target_dir = os.path.join(working_dir, input_dir_name)
    # if the input directory already exists, and not empty, skip copying
    if os.path.exists(input_target_dir) and os.listdir(input_target_dir):
        logging.warning(f"Input directory {input_target_dir} already exists and is not empty. Skipping copying.")
    else:
        # copy the input directory to the working directory
        shutil.copytree(input_dir, input_target_dir)
        logging.info(f"Copied input files to {input_target_dir}")


def generate_config(file_name: str, config_vars: str) -> str:
    config_vars = parse_vars(config_vars)
    print(f"Parsed config vars: {config_vars}")
    template = Template(open(file_name, 'r').read())
    file_name = os.path.join(tempfile.gettempdir(), f"config_{uuid.uuid4().hex}.yaml")
    print(f"Writing config to temporary file: {file_name}")
    with open(file_name, 'w') as f:
        f.write(template.render(config_vars))

    return file_name
