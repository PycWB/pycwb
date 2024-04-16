import os
from typing import List

from pycwb.config import Config
from pycwb.modules.catalog import create_catalog
from pycwb.modules.job_segment import create_job_segment_from_config, save_job_segments_to_json, \
    load_job_segments_from_json
from pycwb.modules.web_viewer.create import create_web_viewer
from pycwb.modules.workflow_utils.job_setup import create_working_directory, \
    check_if_output_exists, create_output_directory, \
    check_MRACatalog_setting
from pycwb.types.job import WaveSegment


def overwrite_config(config: Config, n_proc: int = None, plot_trigger: bool = None, save_waveform: bool = None,
                     plot_waveform: bool = None, save_sky_map: bool = None, plot_sky_map: bool = None,
                     compress_output_json: bool = None) -> Config:
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
    # convert to absolute path in case the current working directory is changed
    working_dir = os.path.abspath(working_dir)
    file_name = os.path.abspath(config_file)

    # create working directory and change the current working directory to the given working directory
    create_working_directory(working_dir)
    os.chdir(working_dir)

    # check environment
    # check_MRACatalog_setting()

    # read user parameters
    config = Config(file_name)

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

        create_catalog(f"{working_dir}/{config.catalog_dir}/catalog.json", config, job_segments)
        create_web_viewer(f"{working_dir}/{config.outputDir}")
        save_job_segments_to_json(job_segments, f"{working_dir}/config/job_segments.json")

    return job_segments, config, working_dir


def load_batch_run(working_dir: str, config_file: str, job_file: str, job_start: int, job_end: int,
                   n_proc: int = 1) -> tuple[List[WaveSegment], Config, str]:
    # convert to absolute path in case the current working directory is changed
    working_dir = os.path.abspath(working_dir)
    file_name = os.path.abspath(config_file)

    os.chdir(working_dir)

    # check_MRACatalog_setting()

    config = Config(file_name)

    if n_proc != 0:
        config.nproc = n_proc

    job_segments = load_job_segments_from_json(job_file)

    if job_start > len(job_segments):
        raise ValueError(f"job_start {job_start} is larger than the number of jobs {len(job_segments)}")

    if job_end > len(job_segments):
        job_end = len(job_segments)

    create_output_directory(working_dir, config.outputDir, config.logDir, config.catalog_dir,
                            config.trigger_dir, file_name)

    create_catalog(f"{working_dir}/{config.catalog_dir}/catalog_{job_start}_{job_end}.json",
                   config, job_segments)

    return job_segments[job_start:job_end], config, working_dir
