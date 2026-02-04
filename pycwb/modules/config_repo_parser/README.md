# Configuration Repository Parser

Parser for the PycWB configuration repository structure, enabling automated access to search configurations and related metadata.

## Overview

The configuration repository parser provides utilities to navigate and extract information from the standardized PycWB search configuration directory structure. It handles:

- **Project name parsing**: Extracts observation, chunk, data quality, search type, and configuration path from structured project identifiers
- **Segment metadata**: Retrieves GPS time boundaries from chunk definition files
- **Data quality files**: Locates and parses detector-specific veto files
- **Intelligent path matching**: Handles complex directory names with underscores by scanning the actual filesystem

## Project Name Format

PycWB project names follow a structured format:

```
{OBS}_K{CHUNK_ID}_{DQ}_{PATH}_{label}
```

**Example**: `O4_K01_C00_BurstHF_LH_SIM_NSGlitch_Set1_InjBuiltin_run4`

### Components

- **`{OBS}_K{CHUNK_ID}`** (e.g., `O4_K01`)
  - Observatory run identifier and chunk number
  - Used to look up GPS time boundaries in `{search}_chunk.lst`

- **`{DQ}`** (e.g., `C00`)
  - Data quality flag version
  - Locates veto files in `DQ/{DQ}/{search}/`

- **`{PATH}`** (e.g., `BurstHF/LH/SIM/NSGlitch_Set1`)
  - Directory path to configuration containing `user_parameters.yaml`
  - First component is the search type (BurstHF, BurstLF, etc.)
  - Handles directory names with underscores intelligently

- **`{label}`** (e.g., `InjBuiltin_run4`)
  - Optional descriptive label for the run variant
  - Everything after the configuration directory

## Configuration Repository Structure

```
config/
├── BurstHF_chunk.lst              # Segment definitions for BurstHF search
├── BurstLF_chunk.lst              # Segment definitions for BurstLF search
├── DQ/
│   └── C00/
│       ├── BurstHF/
│       │   ├── H1_cat0.txt        # Vetoed times for H1
│       │   ├── H1_cat1.txt
│       │   └── L1_cat*.txt
│       └── BurstLF/
│           └── ...
├── BurstHF/
│   └── LH/
│       ├── BKG/
│       │   └── user_parameters.yaml
│       ├── SIM/
│       │   ├── NSGlitch/
│       │   │   └── user_parameters.yaml
│       │   └── NSGlitch_Set1/
│       │       └── user_parameters.yaml
│       └── XGB/
│           └── user_parameters.yaml
└── BurstLF/
    └── ...
```

## API Reference

### `parse_project_name(project_name, config_base_path="./")` → `Dict`

Parse a project name string into structured components.

**Parameters:**
- `project_name` (str): Project identifier string
- `config_base_path` (str): Base directory for configuration search

**Returns:** Dictionary with keys:
- `obs_chunk`: Full observation/chunk identifier (e.g., "O4_K01")
- `obs`: Observatory run (e.g., "O4")
- `chunk_id`: Chunk identifier (e.g., "01")
- `dq`: Data quality version (e.g., "C00")
- `path`: Relative config path (e.g., "BurstHF/LH/SIM/NSGlitch_Set1")
- `search`: Search type (e.g., "BurstHF")
- `label`: Run variant label (e.g., "InjBuiltin_run4")
- `full_path`: Absolute path to config directory
- `config_found`: Whether `user_parameters.yaml` was found

**Example:**
```python
from pycwb.modules.config_repo_parser import parse_project_name

parsed = parse_project_name("O4_K02_C00_BurstLF_LH_BKG_standard")
print(parsed['full_path'])  # /path/to/BurstLF/LH/BKG
```

### `get_gps_time_from_chunk(obs, chunk_id, search, config_base_path="./")` → `Dict`

Retrieve GPS time boundaries for a specific observation and chunk.

**Parameters:**
- `obs` (str): Observatory identifier (e.g., "O4")
- `chunk_id` (str): Chunk ID (e.g., "01")
- `search` (str): Search type (e.g., "BurstLF")
- `config_base_path` (str): Base directory containing `{search}_chunk.lst`

**Returns:** Dictionary with:
- `start`: Start GPS time (int)
- `stop`: Stop GPS time (int)
- `obs`: Observatory identifier
- `chunk_id`: Chunk identifier

**Example:**
```python
gps_times = get_gps_time_from_chunk("O4", "01", "BurstLF")
print(f"Processing segment {gps_times['start']}-{gps_times['stop']}")
```

### `get_dq_files(dq, search, config_full_path, config_base_path="./")` → `List[Dict]`

Find all data quality veto files for a given configuration.

**Parameters:**
- `dq` (str): Data quality flag version (e.g., "C00")
- `search` (str): Search type (e.g., "BurstLF")
- `config_full_path` (str): Full path to config directory (from `parse_project_name`)
- `config_base_path` (str): Base directory containing `DQ/` folder

**Returns:** List of dictionaries, each with:
- `filename`: Full path to veto file
- `ifo`: Interferometer name (e.g., "L1")
- `cat`: Veto category (e.g., "0")

**Example:**
```python
parsed = parse_project_name("O4_K02_C00_BurstLF_LH_BKG_standard")
dq_files = get_dq_files(parsed['dq'], parsed['search'], parsed['full_path'])
for dq_file in dq_files:
    print(f"{dq_file['ifo']} category {dq_file['cat']}: {dq_file['filename']}")
```

## Usage Examples

### Complete Workflow

```python
from pycwb.modules.config_repo_parser import (
    parse_project_name,
    get_gps_time_from_chunk,
    get_dq_files
)

# Parse project identifier
project_id = "O4_K02_C00_BurstLF_LH_BKG_standard"
parsed = parse_project_name(project_id)

# Get segment boundaries
gps = get_gps_time_from_chunk(parsed['obs'], parsed['chunk_id'], parsed['search'])
print(f"Processing {gps['start']} to {gps['stop']}")

# Get data quality files for all detectors
dq_files = get_dq_files(parsed['dq'], parsed['search'], parsed['full_path'])

# Load configuration
import yaml
with open(f"{parsed['full_path']}/user_parameters.yaml") as f:
    config = yaml.safe_load(f)
print(f"Detectors: {config['ifo']}")
```

## Features

### Intelligent Path Matching

The parser handles directory names containing underscores by:
1. Scanning the filesystem to find all directories with `user_parameters.yaml`
2. Matching actual directory components against underscore-split project name parts
3. Prioritizing longer/more specific paths over shorter ones

Example:
- If `SIM/STDINJ/` and `SIM/STDINJ_Set1/` both exist
- Project name `O4_K01_C00_BurstHF_LH_SIM_STDINJ_Set1_variant`
- Parser correctly matches `SIM/STDINJ_Set1/` (not `SIM/STDINJ/`)

### Data Quality File Discovery

Automatically locates veto files matching:
- Detectors defined in `user_parameters.yaml`
- Available veto categories in the DQ directory
- Patterns: `{IFO}_cat{N}.txt`

## Error Handling

All functions raise informative exceptions:

- `FileNotFoundError`: Missing configuration or segment definition files
- `ValueError`: Invalid project name format or missing required parameters
- YAML parsing errors: Invalid configuration file syntax

## Integration with PycWB

This module is designed to support:
- Job initialization and validation
- Configuration loading and validation
- Segment processing workflows
- Data quality flag application
- Workflow orchestration (HTCondor, SLURM, etc.)
