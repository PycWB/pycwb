"""Merge catalog and waveform files from batch runs into single output files."""

import os
import glob
import logging
import click
import h5py as h5
from dacite import from_dict
import pyarrow as pa
import pyarrow.parquet as pq
from pycwb.config import Config
from pycwb.types.job import WaveSegment
from pycwb.modules.catalog import Catalog

logger = logging.getLogger(__name__)


def _collect_jobs_from_fragments(catalog_files: list) -> list:
    """Collect and sort all WaveSegment jobs across fragment catalog files."""
    jobs = []
    for f in catalog_files:
        for job in Catalog.open(f).jobs:
            jobs.append(from_dict(WaveSegment, job))
    jobs.sort(key=lambda j: j.index)
    return jobs


def merge_catalog(working_dir: str = '.', catalog_dir: str = 'catalog', merge_label: str = None):
    """
    Merge the catalog files in the catalog directory from batch runs into a single catalog file.
    
    Parameters
    ----------
    working_dir : str
        The working directory.
    catalog_dir : str
        The directory containing the catalog files to be merged.
    merge_label : str, optional
        The label to be added to the merged catalog file name. If None, the merged catalog file will be named as "catalog.parquet".
    """
    catalog_files = sorted(glob.glob(f"{working_dir}/{catalog_dir}/catalog_*{Catalog.DEFAULT_EXTENSION}"))
    if not catalog_files:
        logger.warning("No catalog files found")
        return

    default_catalog_file = os.path.abspath(f"{working_dir}/catalog/{Catalog.DEFAULT_FILENAME}")

    if merge_label is not None:
        # Build labeled catalog entirely from fragments — no dependency on default_catalog_file
        merged_catalog_file = default_catalog_file.replace(
            Catalog.DEFAULT_EXTENSION, f".{merge_label}{Catalog.DEFAULT_EXTENSION}"
        )
        if os.path.exists(merged_catalog_file):
            if not click.confirm(f"Merged catalog file {merged_catalog_file} already exists. Overwrite?", default=False):
                return
            os.remove(merged_catalog_file)
        config = Config()
        config.load_from_dict(Catalog.open(catalog_files[0]).config)
        jobs = _collect_jobs_from_fragments(catalog_files)
        Catalog.create(merged_catalog_file, config, jobs)
    else:
        # Bootstrap default catalog from fragments if it was never created
        if not os.path.exists(default_catalog_file):
            logger.info("Main catalog not found — creating from fragments: %s", default_catalog_file)
            config = Config()
            config.load_from_dict(Catalog.open(catalog_files[0]).config)
            jobs = _collect_jobs_from_fragments(catalog_files)
            Catalog.create(default_catalog_file, config, jobs)
        merged_catalog_file = default_catalog_file

    # Concatenate trigger rows from all fragments into the target catalog
    tables = []
    for catalog_file in catalog_files:
        table = Catalog.open(os.path.abspath(catalog_file)).triggers()
        tables.append(table)
        logger.info(f"Merging {catalog_file} - {len(table)} events")

    merged_table = pa.concat_tables(tables, promote_options="default")
    logger.info(f"Total number of events: {len(merged_table)}")

    logger.info(f"Save {merged_catalog_file}")
    meta = pq.read_schema(merged_catalog_file).metadata
    pq.write_table(merged_table.replace_schema_metadata(meta), merged_catalog_file, compression="snappy")

def merge_wave(working_dir: str = '.', output_dir: str = 'output', merge_label: str = None):
    """
    Merge the reconstructed waveforms in the output directory from batch runs into a single wave file.

    Parameters
    ----------
    working_dir : str
        The working directory.
    output_dir : str
        The directory containing the wave files to be merged. 
    merge_label : str, optional
        The label to be added to the merged wave file name. If None, the merged wave file will be named as "wave.h5".
    """
    default_wave_file = os.path.abspath(f"{working_dir}/output/wave.h5")
    if merge_label is not None:
        merged_wave_file = default_wave_file.replace('.h5', f'.{merge_label}.h5')

        # check if the wave file with the specified merge label already exists
        if os.path.exists(merged_wave_file):
            logger.warning(f"Merged wave file {merged_wave_file} already exists.")
            return
    else:
        merged_wave_file = default_wave_file

    # check if the merged wave file already exists
    if os.path.exists(merged_wave_file):
        logger.warning(f"Merged wave file {merged_wave_file} already exists.")
        return

    # get the list of hdf5 files wave_*.h5
    wave_files = glob.glob(f"{working_dir}/{output_dir}/wave_*.h5")
    if len(wave_files) == 0:
        logger.warning("No waveform files found")
        return

    # read the merged file wave.h5
    logger.info(f"Create merged wave file {merged_wave_file}")
    with h5.File(merged_wave_file, 'w') as f:
        # read the sub wave files
        for wave_file in wave_files:
            with h5.File(wave_file, 'r') as f_sub:
                logger.info(f"Adding waveforms of {len(f_sub.keys())} events from {wave_file}")
                for event_id in f_sub.keys():
                    f_sub.copy(event_id, f)


def merge_progress(working_dir: str = '.', catalog_dir: str = 'catalog', merge_label: str = None) -> None:
    """Merge per-batch progress Parquet files into a single progress.parquet.

    Parameters
    ----------
    working_dir : str
        The working directory.
    catalog_dir : str
        The directory containing the progress files to be merged.
    merge_label : str, optional
        Label appended to the output file name. If None, the output is
        ``<catalog_dir>/progress.parquet``.
    """
    from pycwb.modules.catalog.catalog import PROGRESS_SCHEMA

    progress_files = glob.glob(f"{working_dir}/{catalog_dir}/progress_*{Catalog.DEFAULT_EXTENSION}")
    if not progress_files:
        logger.info("No progress files to merge")
        return

    default_progress_file = os.path.abspath(f"{working_dir}/{catalog_dir}/progress{Catalog.DEFAULT_EXTENSION}")
    if merge_label is not None:
        merged_progress_file = default_progress_file.replace(
            Catalog.DEFAULT_EXTENSION, f".{merge_label}{Catalog.DEFAULT_EXTENSION}"
        )
        if os.path.exists(merged_progress_file):
            logger.warning(f"Merged progress file {merged_progress_file} already exists.")
            return
    else:
        merged_progress_file = default_progress_file

    tables = []
    for pf in progress_files:
        table = pq.read_table(pf, schema=PROGRESS_SCHEMA)
        tables.append(table)
        logger.info("Merging progress from %s (%d rows)", pf, table.num_rows)

    merged = pa.concat_tables(tables)
    logger.info("Merged progress: %d rows -> %s", len(merged), merged_progress_file)
    pq.write_table(merged, merged_progress_file, compression="snappy")
