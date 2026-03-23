"""
Configuration parser module for PycWB project management.

This module provides utilities to parse project names and retrieve configuration
parameters for gravitational wave burst searches.

Features:
- Parse project names into structured components
- Extract GPS times from chunk definition files
- Find and parse data quality files
"""

import os
import glob
import re
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def parse_project_name(project_name: str, config_base_path: str = "./") -> Dict[str, str]:
    """
    Parse project name into required components with intelligent path matching.
    
    Format: {OBS}_K{CHUNK_ID}_{DQ}_{PATH}_{label}
    Example: O4_K01_C00_BurstHF_LH_SIM_NSGlitch_Set1_InjBuiltin_run4
    
    The path is determined by finding the directory that contains user_parameters.yaml.
    Handles directory names with underscores (e.g., NSGlitch_Set1) by checking actual filesystem.
    Everything after that directory is considered the label.
    
    Args:
        project_name: The project name string to parse
        config_base_path: Base path to search for directory matches (default: "./")
    
    Returns:
        Dictionary containing:
        - obs_chunk: e.g., "O4_K01"
        - obs: e.g., "O4"
        - chunk_id: e.g., "01"
        - dq: e.g., "C00"
        - path: e.g., "BurstHF/LH/SIM/NSGlitch_Set1" (dir with user_parameters.yaml)
        - search: e.g., "BurstHF"
        - label: e.g., "InjBuiltin_run4"
        - full_path: Complete matched directory path
        - config_found: True if user_parameters.yaml exists, False otherwise
    """
    parts = project_name.split('_')
    
    if len(parts) < 3:
        raise ValueError(f"Project name '{project_name}' must have at least 3 parts")
    
    # Extract fixed parts
    obs = parts[0]  # e.g., "O4"
    chunk_id = parts[1].replace('K', '')  # e.g., "01" from "K01"
    obs_chunk = f"{obs}_K{chunk_id}"
    dq = parts[2]  # e.g., "C00"
    
    # The remaining parts need to be parsed for path and label
    remaining_parts = parts[3:]
    
    # Find the longest matching directory path with user_parameters.yaml
    path_info = _find_config_path(remaining_parts, config_base_path)
    
    # Extract search type (first component of path)
    search = path_info['path_components'][0] if path_info['path_components'] else ""
    
    # Extract network (second component of path, e.g., "LH", "HLV")
    network = path_info['path_components'][1] if len(path_info['path_components']) > 1 else ""
    
    # Extract label (everything after the matched path)
    label_parts = remaining_parts[path_info['matched_count']:]
    label = "_".join(label_parts) if label_parts else ""
    
    return {
        'obs_chunk': obs_chunk,
        'obs': obs,
        'chunk_id': chunk_id,
        'dq': dq,
        'path': path_info['relative_path'],
        'search': search,
        'network': network,
        'label': label,
        'full_path': path_info['full_path'],
        'config_found': path_info['config_found']
    }


def get_ifo_list(network: str, config_base_path: str = "./") -> List[str]:
    """
    Get full IFO names from network short code using settings.yaml.
    
    Reads settings.yaml from config base path to map network short names to full IFO lists.
    
    Args:
        network: Network short code (e.g., "LH", "HLV")
        config_base_path: Base path to config repository (default: "./")
    
    Returns:
        List of full IFO names (e.g., ["L1", "H1"] for "LH")
        
    Raises:
        FileNotFoundError: If settings.yaml not found
        ValueError: If network code not found in settings.yaml
        
    Example:
        >>> ifos = get_ifo_list("LH")
        >>> print(ifos)  # ["L1", "H1"]
    """
    settings_file = Path(config_base_path) / "settings.yaml"
    
    if not settings_file.exists():
        raise FileNotFoundError(f"Settings file not found: {settings_file}")
    
    with open(settings_file, 'r') as f:
        settings = yaml.safe_load(f)
    
    if 'networks' not in settings:
        raise ValueError("'networks' section not found in settings.yaml")
    
    networks = settings['networks']
    if network not in networks:
        raise ValueError(f"Network '{network}' not found in settings.yaml. Available: {list(networks.keys())}")
    
    ifo_list = networks[network]
    if isinstance(ifo_list, str):
        ifo_list = [ifo_list]
    
    return ifo_list


def get_machine_settings(config_base_path: str = "./") -> Dict:
    """
    Load machine-specific settings from config/machine/<machine>.yaml.

    Reads the ``machine`` key from ``settings.yaml`` to determine which profile
    to load, then parses ``machine/<machine>.yaml`` and returns its contents.

    Args:
        config_base_path: Base path to the config repository (default: "./")

    Returns:
        Dictionary of machine settings, e.g.::

            {
                'cluster': 'condor',
                'container_image': '...',
                'accounting_group': '...',
                'job_per_worker': 1,
                'job_memory': '8GB',
                'job_disk': '10GB',
                'data_source': 'cit-local',
            }

    Raises:
        FileNotFoundError: If settings.yaml or the machine yaml file is not found.
        ValueError: If the ``machine`` key is missing from settings.yaml.
    """
    settings_file = Path(config_base_path) / "settings.yaml"

    if not settings_file.exists():
        raise FileNotFoundError(f"Settings file not found: {settings_file}")

    with open(settings_file, 'r') as f:
        settings = yaml.safe_load(f)

    machine = settings.get('machine')
    if not machine:
        raise ValueError("'machine' key not found in settings.yaml")

    machine_file = Path(config_base_path) / "machine" / f"{machine}.yaml"
    if not machine_file.exists():
        raise FileNotFoundError(f"Machine config not found: {machine_file}")

    with open(machine_file, 'r') as f:
        return yaml.safe_load(f) or {}


def get_data_settings(data_type: str, dq: str, config_base_path: str = "./") -> Dict:
    """
    Get data source settings from settings.yaml.
    
    Reads settings.yaml to get data source configuration including host, frametype, 
    channelNamesRaw, and urltype.
    
    Args:
        data_type: Data source type (e.g., "igwn-osg", "cit-local", "gwosc")
        dq: Data quality category (e.g., "C00") used to find frametype and channelNamesRaw
        config_base_path: Base path to config repository (default: "./")
    
    Returns:
        Dictionary containing:
        - type: The data type (e.g., "igwn-osg")
        - is_local: Boolean indicating if this is a local file source
        - host: Data host URL
        - frametype: Dict mapping IFO to frame type string
        - channelNamesRaw: List of raw channel names or dict mapping IFO to channel names
        - urltype: URL type (e.g., "osdf", "file")
        
    Raises:
        FileNotFoundError: If settings.yaml not found
        ValueError: If data_type or dq not found in settings
        
    Example:
        >>> settings = get_data_settings("gwosc", "C00")
        >>> print(settings['host'])  # https://datafind.gwosc.org
    """
    settings_file = Path(config_base_path) / "settings.yaml"
    
    if not settings_file.exists():
        raise FileNotFoundError(f"Settings file not found: {settings_file}")
    
    with open(settings_file, 'r') as f:
        settings = yaml.safe_load(f)
    
    if 'data' not in settings:
        raise ValueError("'data' section not found in settings.yaml")
    
    data_sources = settings['data']
    if data_type not in data_sources:
        raise ValueError(f"Data type '{data_type}' not found in settings.yaml. Available: {list(data_sources.keys())}")
    
    data_source = data_sources[data_type]
    
    # Check if it's a local file source
    is_local = data_type == "local"
    
    # Get channelNamesRaw - try to get by dq first, then overall setting
    channel_names_raw = None
    if 'channelNamesRaw' in data_source:
        raw_config = data_source['channelNamesRaw']
        # If it's a dict with dq as key, get the dq entry
        if isinstance(raw_config, dict) and dq in raw_config:
            channel_names_raw = raw_config[dq]
        # If it's a dict without dq structure, use as is
        elif isinstance(raw_config, dict):
            channel_names_raw = raw_config
        # If it's a list, use as is
        else:
            channel_names_raw = raw_config
    
    return {
        'type': data_type,
        'is_local': is_local,
        'host': data_source.get('host'),
        'frametype': data_source.get('frametype', {}).get(dq, {}),
        'channelNamesRaw': channel_names_raw,
        'urltype': data_source.get('urltype')
    }


def get_gps_time_from_chunk(obs: str, chunk_id: str, search: str, config_base_path: str = "./") -> Dict[str, int]:
    """
    Find start and stop GPS time for a given obs and chunk_id from {search}_chunk.lst file.
    
    Args:
        obs: Observatory identifier (e.g., "O4")
        chunk_id: Chunk ID as string (e.g., "01")
        search: Search type (e.g., "BurstHF")
        config_base_path: Base path where chunk files are located (default: "./")
    
    Returns:
        Dictionary containing:
        - start: Start GPS time (int)
        - stop: Stop GPS time (int)
        - obs: The observation name
        - chunk_id: The chunk ID
        
    Raises:
        FileNotFoundError: If {search}_chunk.lst file not found
        ValueError: If obs/chunk_id combination not found in file
    """
    chunk_file = Path(config_base_path) / f"{search}_chunk.lst"
    
    if not chunk_file.exists():
        raise FileNotFoundError(f"Chunk file not found: {chunk_file}")
    
    with open(chunk_file, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Parse the line: OBS CHUNK_ID START STOP ...
            parts = line.split()
            if len(parts) < 4:
                continue
            
            file_obs = parts[0]
            file_chunk_id = parts[1]
            
            # Normalize chunk_id by stripping leading zeros from numeric prefix,
            # preserving any trailing suffix characters (e.g., "01a" -> "1a", "01" -> "1")
            def _normalize_chunk_id(s):
                return re.sub(r'^0+(\w)', r'\1', s)

            if file_obs == obs and _normalize_chunk_id(file_chunk_id) == _normalize_chunk_id(chunk_id):
                try:
                    start = int(parts[2])
                    stop = int(parts[3])
                    return {
                        'start': start,
                        'stop': stop,
                        'obs': obs,
                        'chunk_id': chunk_id
                    }
                except ValueError:
                    continue
    
    raise ValueError(f"obs={obs}, chunk_id={chunk_id} not found in {chunk_file}")


def get_dq_files(dq: str, search: str, ifo: List[str], config_base_path: str = "./") -> List[Dict[str, str]]:
    """
    Find DQ files for given dq and search parameters with metadata.
    
    Checks if DQ/{dq}/{search} directory exists and finds all {ifo}_{cat}.txt files
    for the provided IFO list. Reads metadata.yaml to get type, inverted, and column4 info.
    
    Args:
        dq: Data quality category (e.g., "C00")
        search: Search type (e.g., "BurstLF")
        ifo: List of interferometer names (e.g., ["L1", "H1"])
        config_base_path: Base path where DQ directory is located (default: "./")
    
    Returns:
        List of dictionaries, each containing:
        - filename: Full path to the DQ file
        - ifo: Interferometer name (e.g., "L1")
        - cat: Category number (e.g., "0")
        - type: DQ type constant (e.g., "CWB_CAT0")
        - inverted: Boolean indicating if DQ is inverted
        - column4: Boolean indicating if file has 4 columns
        
    Raises:
        FileNotFoundError: If DQ/{dq}/{search} directory doesn't exist
        ValueError: If no matching DQ files found
    """
    dq_dir = Path(config_base_path) / "DQ" / dq / search
    
    if not dq_dir.exists():
        raise FileNotFoundError(f"DQ directory not found: {dq_dir}")
    
    # Read metadata.yaml
    metadata_file = dq_dir / "metadata.yaml"
    metadata = {}
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = yaml.safe_load(f) or {}
    
    # Ensure ifo is a list
    if isinstance(ifo, str):
        ifo = [ifo]
    
    ifos = ifo
    
    # Find all matching DQ files
    dq_files = []
    for ifo in ifos:
        # Find all files matching {ifo}_cat*.txt pattern
        pattern = str(dq_dir / f"{ifo}_cat*.txt")
        for filepath in glob.glob(pattern):
            # Extract category from filename (e.g., "L1_cat0.txt" -> "0")
            filename = Path(filepath).name
            # Parse pattern: {ifo}_cat{cat}.txt
            cat = filename.split('_cat')[1].replace('.txt', '')
            
            # Get metadata for this file (try full name first, then generic cat name)
            file_key = f"{ifo}_cat{cat}"
            file_metadata = metadata.get(file_key, metadata.get(f"cat{cat}", {}))
            
            dq_files.append({
                'filename': filepath,
                'ifo': ifo,
                'cat': cat,
                'type': file_metadata.get('type', f'CWB_CAT{cat}'),
                'inverted': file_metadata.get('inverted', False),
                'column4': file_metadata.get('column4', False)
            })
    
    # Sort by ifo, then by cat
    dq_files.sort(key=lambda x: (x['ifo'], x['cat']))
    
    return dq_files


def parse_project(project_name: str, config_base_path: str = "./") -> Dict:
    """
    Complete project parsing in one call: parse name, get GPS times, and get DQ files.
    
    This is a convenience function that combines parse_project_name, get_gps_time_from_chunk,
    get_ifo_list, and get_dq_files into a single operation.
    
    Args:
        project_name: The project name string to parse
        config_base_path: Base path for configuration search (default: "./")
    
    Returns:
        Dictionary containing:
        - parsed: Result from parse_project_name() with all project info
        - gps_times: Result from get_gps_time_from_chunk() with start/stop times
        - ifo: List of full IFO names from get_ifo_list()
        - dq_files: Result from get_dq_files() with list of veto files
        
    Raises:
        ValueError: If project name format is invalid
        FileNotFoundError: If required files (chunk.lst, settings.yaml, DQ files) not found
        
    Example:
        >>> result = parse_project("O4_K02_C00_BurstLF_LH_BKG_standard")
        >>> print(f"Processing {result['gps_times']['start']} to {result['gps_times']['stop']}")
        >>> print(f"IFOs: {result['ifo']}")
        >>> print(f"DQ files: {result['dq_files']}")
    """
    # Parse the project name
    parsed = parse_project_name(project_name, config_base_path)
    
    # Get GPS times
    gps_times = get_gps_time_from_chunk(
        parsed['obs'],
        parsed['chunk_id'],
        parsed['search'],
        config_base_path
    )
    
    # Get IFO list from network code
    ifo = get_ifo_list(parsed['network'], config_base_path)
    
    # Get DQ files
    dq_files = get_dq_files(
        parsed['dq'],
        parsed['search'],
        ifo,
        config_base_path
    )
    
    return {
        'parsed': parsed,
        'gps_times': gps_times,
        'ifo': ifo,
        'dq_files': dq_files
    }


# ============================================================================
# Helper functions (private)
# ============================================================================

def _get_all_config_dirs(base_path: str) -> List[Tuple[Path, List[str]]]:
    """
    Scan filesystem and return all directories containing user_parameters.yaml.
    
    Returns list of tuples: (absolute_path, path_components_list)
    """
    base = Path(base_path).resolve()
    config_dirs = []
    
    for root, dirs, files in os.walk(base):
        if "user_parameters.yaml" in files:
            rel_path = Path(root).relative_to(base)
            # Get path components - preserves underscores in directory names
            components = rel_path.parts if rel_path != Path('.') else []
            config_dirs.append((Path(root), list(components)))
    
    # Sort by length descending (longest paths first)
    config_dirs.sort(key=lambda x: len(x[1]), reverse=True)
    return config_dirs


def _match_path_components(target_parts: List[str], actual_components: List[str]) -> Optional[int]:
    """
    Try to match actual directory components against the underscore-split parts.
    
    Returns the number of target_parts consumed if match is found, None otherwise.
    
    Example:
        target_parts = ["BurstHF", "LH", "SIM", "NSGlitch", "Set1", "extra"]
        actual_components = ["BurstHF", "LH", "SIM", "NSGlitch_Set1"]
        Returns: 5 (BurstHF + LH + SIM + NSGlitch_Set1 consumed 5 parts when reconstructed with underscores)
    """
    if not actual_components:
        return None
    
    # Try to match actual_components sequentially against target_parts
    target_idx = 0
    
    for actual_comp in actual_components:
        # Split actual component by underscores to see how many target parts it uses
        actual_split = actual_comp.split('_')
        
        # Check if target_parts starting at target_idx match this component
        needed_parts = len(actual_split)
        if target_idx + needed_parts > len(target_parts):
            return None  # Not enough target parts left
        
        # Check if the parts match
        if target_parts[target_idx:target_idx + needed_parts] == actual_split:
            target_idx += needed_parts
        else:
            return None  # Mismatch
    
    return target_idx


def _find_config_path(parts: List[str], base_path: str) -> Dict[str, any]:
    """
    Find the directory path that contains user_parameters.yaml, handling underscores in names.
    
    Scans actual filesystem to find directories with user_parameters.yaml, then tries to match
    them against the project name parts. Handles directory names containing underscores.
    
    Args:
        parts: List of path components from project name (split by underscores)
        base_path: Base directory to search in
    
    Returns:
        Dictionary with:
        - relative_path: Matched path relative to base (e.g., "BurstHF/LH/SIM/NSGlitch_Set1")
        - full_path: Absolute path to matched directory
        - matched_count: Number of parts that were matched
        - path_components: List of matched path components
        - config_found: True if user_parameters.yaml exists in the matched directory
    """
    base = Path(base_path).resolve()
    
    # Get all valid config directories, sorted by depth (longest first)
    config_dirs = _get_all_config_dirs(base_path)
    
    # Try to match each config directory against the parts
    for full_path, path_components in config_dirs:
        matched_count = _match_path_components(parts, path_components)
        if matched_count is not None:
            # Found a match!
            relative_path = str(Path(*path_components)) if path_components else ""
            return {
                'relative_path': relative_path,
                'full_path': str(full_path),
                'matched_count': matched_count,
                'path_components': path_components,
                'config_found': True
            }
    
    # If no config found, return warning with first part as search type
    return {
        'relative_path': parts[0] if parts else "",
        'full_path': str(base / parts[0]) if parts else str(base),
        'matched_count': 1 if parts else 0,
        'path_components': [parts[0]] if parts else [],
        'config_found': False
    }


if __name__ == "__main__":
    # Example usage
    test_names = [
        "O4_K01_C00_BurstHF_LH_SIM_NSGlitch_Set1_InjBuiltin_run4",
        "O4_K02_C00_BurstLF_LH_BKG_standard",
        "O4_K05_C01_BurstHF_LH_SIM_STDINJs_Set1_test1",
    ]
    
    for name in test_names:
        print(f"\nParsing: {name}")
        try:
            result = parse_project_name(name, config_base_path="./")
            for key, value in result.items():
                print(f"  {key}: {value}")
            if not result['config_found']:
                print(f"  WARNING: user_parameters.yaml not found in {result['full_path']}")
        except Exception as e:
            print(f"  Error: {e}")
