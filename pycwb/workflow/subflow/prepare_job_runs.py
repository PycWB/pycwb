import os
import orjson
from typing import List
from dacite import from_dict

from pycwb.config import Config
from pycwb.modules.catalog import create_catalog
from pycwb.modules.job_segment import create_job_segment_from_config
from pycwb.modules.web_viewer.create import create_web_viewer
from pycwb.modules.workflow_utils.job_setup import create_working_directory, \
    check_if_output_exists, create_output_directory
from pycwb.types.job import WaveSegment
from pycwb.utils.parser import parse_id_string


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

    # create working directory and change the current working directory to the given working directory
    create_working_directory(working_dir)
    os.chdir(working_dir)

    # check environment
    # check_MRACatalog_setting()

    # read user parameters
    config = Config()
    config.load_from_yaml(file_name)

    job_segments = create_job_segment_from_config(config)
    # slags = generate_slags(len(config.ifo), config.slagMin, config.slagMax, config.slagOff, config.slagSize)

    if not dry_run:
        print(f"Number of jobs: {len(job_segments)}")
        # override n_proc in config
        config = overwrite_config(config, n_proc=n_proc, save_waveform=plot, save_sky_map=plot,
                                  plot_trigger=plot, plot_waveform=plot, plot_sky_map=plot,
                                  compress_output_json=compress_json)

        check_if_output_exists(working_dir, config.outputDir, overwrite)
        create_output_directory(working_dir, config.outputDir, config.logDir, config.catalog_dir,
                                config.trigger_dir, file_name)

        catalog_file = f"{working_dir}/{config.catalog_dir}/catalog.json"

        if not os.path.exists(catalog_file):
            create_catalog(catalog_file, config, job_segments)
        create_web_viewer(f"{working_dir}/public")
        # save_job_segments_to_json(job_segments, f"{working_dir}/config/job_segments.json")

    return job_segments, config, working_dir


def load_batch_run(working_dir: str, config_file: str, jobs: str, compress_json: bool = True,
                   n_proc: int = 1) -> tuple[List[WaveSegment], Config, str, str]:
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
    job_ids = parse_id_string(jobs)

    # convert to absolute path in case the current working directory is changed
    working_dir = os.path.abspath(working_dir)
    file_name = os.path.abspath(config_file)

    os.chdir(working_dir)

    # check_MRACatalog_setting()

    # # TODO: should it be loaded from the catalog?
    # config = Config()
    # config.load_from_yaml(file_name)

    # config = overwrite_config(config, n_proc=n_proc,
    #                           compress_output_json=compress_json)

    # # TODO: load job segments from catalog
    # job_segments = create_job_segment_from_config(config)

    catalog = orjson.loads(open('catalog/catalog.json', 'rb').read())
    config = Config()
    config.load_from_dict(catalog['config'])
    print(f"Loaded config from catalog: {config}")
    job_segments = catalog['jobs']
    print(f"Loaded {len(job_segments)} job segments from catalog")

    if max(job_ids) - 1 > len(job_segments):
        raise ValueError(f"job_start {max(job_ids)} is larger than the number of jobs {len(job_segments)}")

    # selected_job_segments = [job_segments[i-1] for i in job_ids]
    selected_job_segments = [from_dict(WaveSegment, job_segments[i-1]) for i in job_ids]

    create_output_directory(working_dir, config.outputDir, config.logDir, config.catalog_dir,
                            config.trigger_dir, file_name)

    catalog_file = f"{working_dir}/{config.catalog_dir}/catalog_{jobs}.json"

    if not os.path.exists(catalog_file):
        create_catalog(catalog_file, config, selected_job_segments)

    return selected_job_segments, config, working_dir, catalog_file
