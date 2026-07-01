"""Merge catalog and waveform files from batch runs into single output files."""

import os
import glob
import logging
import click
import h5py as h5
import itertools
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dacite import from_dict, Config as DaciteConfig
import pyarrow as pa
import pyarrow.parquet as pq
from pycwb.config import Config
from pycwb.types.job import WaveSegment
from pycwb.modules.catalog import Catalog
import time

logger = logging.getLogger(__name__)


def _read_wave_file(wave_file: str) -> tuple:
    """Read all event data from one wave HDF5 file into plain Python/numpy objects.

    Returns ``(wave_file, events)`` where *events* maps
    ``event_id -> {'attrs': dict, 'datasets': {name -> {'data': ndarray, 'attrs': dict}}}``.
    Must be a module-level function so ProcessPoolExecutor can pickle it.
    """
    import h5py
    events = {}
    with h5py.File(wave_file, 'r') as f:
        for event_id in f.keys():
            grp = f[event_id]
            datasets = {}
            for ds_name in grp.keys():
                ds = grp[ds_name]
                datasets[ds_name] = {'data': ds[()], 'attrs': dict(ds.attrs)}
            events[event_id] = {'attrs': dict(grp.attrs), 'datasets': datasets}
    return wave_file, events


def _collect_jobs_from_fragments(catalog_files: list) -> list:
    """Collect and sort all WaveSegment jobs across fragment catalog files."""
    jobs = []
    for f in catalog_files:
        for job in Catalog.open(f).jobs:
            jobs.append(from_dict(WaveSegment, job, config=DaciteConfig(cast=[tuple])))
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
    catalog_files = sorted(glob.glob(f"{working_dir}/{catalog_dir}/fragment/catalog_*{Catalog.DEFAULT_EXTENSION}"))
    if not catalog_files:
        logger.warning("No catalog files found")
        return

    default_catalog_file = os.path.abspath(f"{working_dir}/catalog/{Catalog.DEFAULT_FILENAME}")

    # Determine output path
    merged_catalog_file = (
        default_catalog_file.replace(Catalog.DEFAULT_EXTENSION, f".{merge_label}{Catalog.DEFAULT_EXTENSION}")
        if merge_label is not None
        else default_catalog_file
    )

    # Handle existing file: labeled runs ask for confirmation; unlabeled reuses existing structure
    if os.path.exists(merged_catalog_file):
        if merge_label is None:
            existing_master_catalog = Catalog.open(merged_catalog_file)
            n_events = len(existing_master_catalog.triggers())
            if n_events == 0:
                logger.info("Reusing existing empty master catalog %s", merged_catalog_file)
                # File structure is intact; pq.write_table below will populate trigger rows
            else:
                if not click.confirm(
                    f"Master catalog {merged_catalog_file} already contains {n_events} events. "
                    "Remove all trigger rows and re-merge?",
                    default=False,
                ):
                    return
                # pq.write_table at the end will overwrite the trigger rows
        else:
            if not click.confirm(f"Merged catalog file {merged_catalog_file} already exists. Overwrite?", default=False):
                return
            os.remove(merged_catalog_file)

    # Create catalog structure from fragments if not present
    if not os.path.exists(merged_catalog_file):
        logger.info("Creating catalog from fragments: %s", merged_catalog_file)
        config = Config()
        config.load_from_dict(Catalog.open(catalog_files[0]).config)
        Catalog.create(merged_catalog_file, config, _collect_jobs_from_fragments(catalog_files))

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

def merge_wave(working_dir: str = '.', output_dir: str = 'output', merge_label: str = None, n_proc: int = None):
    """
    Merge the reconstructed waveforms in the output directory from batch runs into a single wave file.

    Source files are read in parallel using ``n_proc`` worker processes; the
    single output file is written serially as each worker finishes.

    Parameters
    ----------
    working_dir : str
        The working directory.
    output_dir : str
        The directory containing the wave files to be merged.
    merge_label : str, optional
        The label to be added to the merged wave file name. If None, the merged wave file will be named as "wave.h5".
    n_proc : int, optional
        Number of worker processes for parallel reading. Defaults to ``os.cpu_count()``, capped at the number of source files.
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

    workers = min(n_proc or os.cpu_count() or 1, len(wave_files))
    logger.info(f"Create merged wave file {merged_wave_file} from {len(wave_files)} source files "
                f"using {workers} workers")

    start = time.time()
    with h5.File(merged_wave_file, 'w') as f_out:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            file_iter = iter(wave_files)

            # Seed the pool with the first `workers` files
            pending = {executor.submit(_read_wave_file, wf): wf
                       for wf in itertools.islice(file_iter, workers)}

            while pending:
                # Block until at least one read finishes
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    del pending[future]
                    source_file, event_data = future.result()
                    logger.info(f"Adding waveforms of {len(event_data)} events from {source_file}")
                    for event_id, event_info in event_data.items():
                        grp = f_out.create_group(event_id)
                        for attr_key, attr_val in event_info['attrs'].items():
                            grp.attrs[attr_key] = attr_val
                        for ds_name, ds_info in event_info['datasets'].items():
                            ds = grp.create_dataset(ds_name, data=ds_info['data'])
                            for attr_key, attr_val in ds_info['attrs'].items():
                                ds.attrs[attr_key] = attr_val
                    # Only submit the next file after this result has been written
                    try:
                        wf = next(file_iter)
                        pending[executor.submit(_read_wave_file, wf)] = wf
                    except StopIteration:
                        pass
    
    logger.info(f"Finished merging: {merged_wave_file}")
    logger.info(f"Elapsed time: {time.time() - start:.2f} s")

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

    progress_files = glob.glob(f"{working_dir}/{catalog_dir}/fragment/progress_*{Catalog.DEFAULT_EXTENSION}")
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
