"""
Project setup utilities for PycWB configuration repository.

Handles initialization of job working directories with proper configuration
and data quality files.
"""

import shutil
import logging
from pathlib import Path
from typing import Dict, Optional
from jinja2 import Template
import click

from .config_repo_parser import parse_project, get_ifo_list, get_data_settings

logger = logging.getLogger(__name__)


def setup_project(work_dir: str, config_base_path: str = "./", data_type: str = "gwosc", dry_run: bool = False) -> Dict:
    """
    Setup a project working directory with configuration and DQ files.
    
    This function:
    1. Parses the project name from the working directory path
    2. Creates the working directory
    3. Copies user_parameters.yaml from the config repository
    4. Copies DQ files for all detectors
    5. Creates input directory structure
    6. Updates GPS times and chunk ID in user_parameters.yaml using Jinja2 templates
    7. Configures data source (gwdatafind for remote, frFiles for local)
    
    Directory naming convention:
        Parent directory base name is treated as the project name.
        Example: /work/O4_K02_C00_BurstLF_LH_BKG_standard/ 
                 → Project name: O4_K02_C00_BurstLF_LH_BKG_standard
    
    Args:
        work_dir: Absolute or relative path to the working directory
                 (will be created if it doesn't exist)
        config_base_path: Base path to the configuration repository (default: "./")
        data_type: Data source type from settings.yaml (default: "gwosc")
                  Options: "igwn-osg", "cit-local", "gwosc"
        dry_run: If True, only show what would be done without creating files (default: False)
    
    Returns:
        Dictionary containing:
        - work_dir: Full path to working directory (Path object)
        - project_name: Parsed project identifier
        - parsed: Project parsing results
        - gps_times: GPS time boundaries
        - dq_files: List of copied DQ files with paths
        - config_file: Path to user_parameters.yaml in work directory
        - input_dir: Path to input directory
        - data_type: Selected data source type
        - data_settings: Data source configuration used
        
    Raises:
        ValueError: If project name cannot be parsed
        FileNotFoundError: If required source files don't exist
        PermissionError: If unable to create directories or copy files
        
    Example:
        >>> result = setup_project("/work/O4_K02_C00_BurstLF_LH_BKG_standard", data_type="gwosc")
        >>> print(f"Setup complete: {result['work_dir']}")
        >>> print(f"Data source: {result['data_type']}")
    """
    work_path = Path(work_dir).resolve()
    # ask user to confirm if work_path exists and is not empty
    if work_path.exists() and any(work_path.iterdir()):
        if not dry_run:
            click.confirm(f"Working directory {work_path} already exists and is not empty. Continue?", abort=True)
        else:
            logger.info(f"[DRY RUN] Working directory {work_path} already exists and is not empty.")
    
    project_name = work_path.name
    
    logger.info(f"Setting up project: {project_name}")
    logger.info(f"Working directory: {work_path}")
    
    # Parse the project to get all information
    logger.debug(f"Parsing project name: {project_name}")
    project_info = parse_project(project_name, config_base_path)
    
    parsed = project_info['parsed']
    gps_times = project_info['gps_times']
    dq_files = project_info['dq_files']
    ifo_list = project_info['ifo']
    
    if not parsed['config_found']:
        raise FileNotFoundError(f"Configuration not found for {project_name}")
    
    # Create work directory structure
    _create_directory_structure(work_path, dry_run)
    
    # Extract IFO site names from network string (e.g., 'LH' -> ['L', 'H'])
    ifo_site_name = list(parsed['network']) if parsed['network'] else []
    
    # Get data source settings
    logger.debug(f"Getting data settings for type: {data_type}, dq: {parsed['dq']}")
    data_settings = get_data_settings(data_type, parsed['dq'], config_base_path)
    
    # Copy and setup configuration files
    config_source = Path(parsed['full_path']) / "user_parameters.yaml"
    config_dest = work_path / "config/user_parameters.yaml"
    
    logger.info(f"Copying configuration from {config_source}")
    _copy_and_template_config(
        config_source,
        config_dest,
        gps_times['start'],
        gps_times['stop'],
        parsed['chunk_id'],
        ifo_list,
        ifo_site_name,
        dq_files,
        data_settings,
        dry_run
    )

    # user defined input files
    input_files_source = Path(parsed['full_path']) / "input"
    if input_files_source.exists() and input_files_source.is_dir():
        for item in input_files_source.iterdir():
            dest_item = work_path / "input" / item.name
            if dry_run:
                logger.info(f"[DRY RUN] Would copy input file {item} to {dest_item}")
            else:
                if item.is_file():
                    shutil.copy2(item, dest_item)
                    logger.debug(f"Copied input file: {item.name}")
                elif item.is_dir():
                    shutil.copytree(item, dest_item)
                    logger.debug(f"Copied input directory: {item.name}")
                    
    # Copy frames files for local data sources
    if data_settings['is_local']:
        frames_source_dir = Path(parsed['full_path']) / "frames"
        for ifo in ifo_list:
            frames_file = frames_source_dir / f"{ifo}.frames"
            dest_frames = work_path / "input" / f"{ifo}.frames"
            if dry_run:
                logger.info(f"[DRY RUN] Would copy {frames_file} to {dest_frames}")
            else:
                if not frames_file.exists():
                    raise FileNotFoundError(f"Frames file not found: {frames_file}")
                shutil.copy2(frames_file, dest_frames)
                logger.debug(f"Copied frames file: {frames_file.name}")

    # Copy DQ files directly to input directory
    input_dir = work_path / "input"
    copied_dq_files = _copy_dq_files(dq_files, input_dir, dry_run)
    
    logger.info(f"Setup complete for {project_name}")
    
    return {
        'work_dir': work_path,
        'project_name': project_name,
        'parsed': parsed,
        'gps_times': gps_times,
        'dq_files': copied_dq_files,
        'config_file': config_dest,
        'input_dir': input_dir,
        'data_type': data_type,
        'data_settings': data_settings,
        'dry_run': dry_run
    }


def _create_directory_structure(work_path: Path, dry_run: bool = False) -> None:
    """Create the working directory and subdirectories."""
    dirs_to_create = [
        work_path,
        work_path / "input"
    ]
    
    for dir_path in dirs_to_create:
        if dry_run:
            logger.info(f"[DRY RUN] Would create directory: {dir_path}")
        else:
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {dir_path}")


def _copy_and_template_config(
    source_file: Path,
    dest_file: Path,
    gps_start: int,
    gps_end: int,
    chunk_id: str,
    ifo_list: list,
    ifo_site_name: list,
    dq_files: list,
    data_settings: Dict,
    dry_run: bool = False
) -> None:
    """
    Copy user_parameters.yaml and update GPS times, chunk ID, and data source using Jinja2 templates.
    
    The function looks for Jinja2 template variables in the YAML file:
    - {{ CHUNK_START }}: Will be replaced with GPS start time
    - {{ CHUNK_END }}: Will be replaced with GPS end time
    - {{ CHUNK_ID }}: Will be replaced with chunk ID
    - {{ IFO_SHORT_NAME }}: Will be replaced with IFO array like ['L1', 'H1']
    - {{ REF_IFO_SHORT_NAME }}: Will be replaced with first IFO like 'L1'
    - {{ IFO_SITE_NAME }}: Will be replaced with site names like ['L', 'H'] (from network 'LH')
    - {{ CHANNELNAMES_RAW }}: Will be replaced with raw channel names from data settings
    - {{ DQF }}: Will be replaced with DQ file list generated from metadata
    - {{ GWDATAFIND }}: Will be replaced with gwdatafind config for remote data
    - {{ FRFILES }}: Will be replaced with frFiles list for local data
    
    Args:
        source_file: Source user_parameters.yaml path
        dest_file: Destination path
        gps_start: GPS start time (int)
        gps_end: GPS end time (int)
        chunk_id: Chunk identifier (str)
        ifo_list: List of full IFO names (e.g., ['L1', 'H1'])
        ifo_site_name: List of site names (e.g., ['L', 'H'])
        dq_files: List of DQ file dictionaries from get_dq_files() with metadata
        data_settings: Data source configuration from get_data_settings()
        dry_run: If True, don't actually write files
    """
    if not source_file.exists():
        raise FileNotFoundError(f"Source configuration not found: {source_file}")
    
    # Generate DQF list from dq_files with metadata
    dqf_list = []
    for dq in dq_files:
        # Format: [ifo, path, type, 0., inverted, column4]
        dqf_entry = [
            dq['ifo'],
            f"input/{Path(dq['filename']).name}",
            dq['type'],
            0.,
            dq['inverted'],
            dq['column4']
        ]
        dqf_list.append(dqf_entry)
    
    # Build gwdatafind or frFiles config based on data source type
    gwdatafind_config = None
    frfiles_list = None
    
    if data_settings['is_local']:
        # For local data, use frFiles
        frfiles_list = []
        for ifo in ifo_list:
            frfiles_list.append(f"input/{ifo}.frames")
    else:
        # For remote data, use gwdatafind
        gwdatafind_config = {
            'host': data_settings['host'],
            'frametype': [data_settings['frametype'][ifo] for ifo in ifo_list],
            'site': ifo_site_name,
            'urltype': data_settings['urltype']
        }
    
    # Read the source file
    with open(source_file, 'r') as f:
        template_content = f.read()
    
    # Create Jinja2 template and render with string conversions
    template = Template(template_content)
    rendered_content = template.render(
        CHUNK_START=str(gps_start),
        CHUNK_END=str(gps_end),
        CHUNK_ID=str(chunk_id),
        IFO_SHORT_NAME=ifo_list,
        REF_IFO_SHORT_NAME=ifo_list[0] if ifo_list else "",
        IFO_SITE_NAME=ifo_site_name,
        DQF=dqf_list,
        GWDATAFIND=gwdatafind_config,
        FRFILES=frfiles_list,
        CHANNELNAMES_RAW=[data_settings.get('channelNamesRaw')[ifo] for ifo in ifo_list] if data_settings.get('channelNamesRaw') else [],
        DATA_TYPE=data_settings['type']
    )
    
    if dry_run:
        logger.info(f"[DRY RUN] Would write configuration to: {dest_file}")
        logger.debug(f"Template variables: CHUNK_START={gps_start}, CHUNK_END={gps_end}, CHUNK_ID={chunk_id}")
        logger.debug(f"IFO_SHORT_NAME={ifo_list}, REF_IFO_SHORT_NAME={ifo_list[0] if ifo_list else ''}")
        logger.debug(f"IFO_SITE_NAME={ifo_site_name}")
        logger.debug(f"DQF list entries: {len(dqf_list)}")
        logger.debug(f"Rendered content:\n{rendered_content[:200]}...")
    else:
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_file, 'w') as f:
            f.write(rendered_content)
        logger.info(f"Wrote configuration to: {dest_file}")
        logger.debug(f"Template variables: CHUNK_START={gps_start}, CHUNK_END={gps_end}, CHUNK_ID={chunk_id}")
        logger.debug(f"IFO_SHORT_NAME={ifo_list}, REF_IFO_SHORT_NAME={ifo_list[0] if ifo_list else ''}")
        logger.debug(f"IFO_SITE_NAME={ifo_site_name}")
        logger.debug(f"DQF list entries: {len(dqf_list)}")


def _copy_dq_files(
    dq_files: list,
    dest_dir: Path,
    dry_run: bool = False
) -> Dict[str, str]:
    """
    Copy DQ files to working directory.
    
    Args:
        dq_files: List of DQ file dictionaries from get_dq_files()
        dest_dir: Destination directory for DQ files
        dry_run: If True, don't actually copy files
        
    Returns:
        Dictionary mapping source paths to destination paths
    """
    if not dq_files:
        logger.warning("No DQ files found to copy")
        return {}
    
    copied_files = {}
    
    if not dry_run:
        dest_dir.mkdir(parents=True, exist_ok=True)
    
    for dq_file in dq_files:
        source = Path(dq_file['filename'])
        dest = dest_dir / source.name
        
        if dry_run:
            logger.info(f"[DRY RUN] Would copy {source.name} to {dest_dir}")
        else:
            if not source.exists():
                logger.warning(f"Source DQ file not found: {source}")
                continue
            
            shutil.copy2(source, dest)
            logger.debug(f"Copied DQ file: {source.name}")
        
        copied_files[str(source)] = str(dest)
    
    logger.info(f"Copied {len(copied_files)} DQ files to {dest_dir}")
    return copied_files


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    try:
        result = setup_project("/tmp/test_O4_K02_C00_BurstLF_LH_BKG_standard", dry_run=False)
        print(f"\nSetup result:")
        for key, value in result.items():
            if key != 'dq_files':
                print(f"  {key}: {value}")
    except Exception as e:
        logger.error(f"Setup failed: {e}")
