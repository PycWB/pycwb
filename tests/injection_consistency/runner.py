"""runner.py — Programmatic invocation of the pycWB e2e pipeline for testing.

Provides :func:`run_pipeline` which sets up a working directory, copies
required asset files (wdmXTalk catalog, noise PSDs), and invokes the
native ``process_job_segment()`` for a single segment.
"""
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Optional

from pycwb.config import Config
from pycwb.workflow.subflow.prepare_job_runs import prepare_job_runs
from pycwb.workflow.subflow.process_job_segment_native import process_job_segment

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths relative to the test package
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "config" / "user_parameters.yaml"
ASSET_DIRS = [
    "input",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_pipeline(
    tmp_path: str | Path,
    config_path: str | Path | None = None,
    copy_assets: bool = True,
) -> Path:
    """Run the pycWB e2e pipeline for a single segment and return the catalog path.

    Parameters
    ----------
    tmp_path : str or Path
        Temporary working directory (e.g. pytest's ``tmp_path``).
    config_path : str or Path, optional
        Path to the YAML config file.  Defaults to
        ``tests/injection_consistency/config/user_parameters.yaml``.
    copy_assets : bool
        If True (default), copy ``wdmXTalk/`` and ``input/`` directories
        from the test package into *tmp_path* so relative paths in the
        config resolve correctly.

    Returns
    -------
    Path
        Absolute path to the generated ``catalog/catalog.parquet``.
    """
    # Save current working directory to restore after pipeline
    _saved_cwd = os.getcwd()

    tmp = Path(tmp_path)
    config_p = Path(config_path) if config_path else DEFAULT_CONFIG

    # ------------------------------------------------------------------
    # 1. Copy asset directories so relative paths in the YAML work
    # ------------------------------------------------------------------
    if copy_assets:
        for asset_dir in ASSET_DIRS:
            src = HERE / asset_dir
            dst = tmp / asset_dir
            if src.exists() and not dst.exists():
                shutil.copytree(src, dst)

    # ------------------------------------------------------------------
    # 2. Copy config into tmp_path so the test is self-contained
    #     (prepare_job_runs also copies it, but we do it early for clarity)
    # ------------------------------------------------------------------
    config_in_tmp = tmp / config_p.name
    if not config_in_tmp.exists():
        shutil.copy2(config_p, config_in_tmp)

    try:
        # ------------------------------------------------------------------
        # 3. Prepare job runs — creates dirs, catalog, job segments
        #     NOTE: prepare_job_runs changes cwd to working_dir internally
        # ------------------------------------------------------------------
        job_segments, config, working_dir = prepare_job_runs(
            working_dir=str(tmp),
            config_file=str(config_in_tmp),
            n_proc=1,
        )

        if len(job_segments) == 0:
            raise RuntimeError("No job segments generated from config")

        # ------------------------------------------------------------------
        # 4. Run the pipeline for the first (only) job segment
        #     Construct the catalog path the same way search() in run.py does
        #     (absolute path) so _catalog_path() returns it as-is without
        #     double-prepending config.catalog_dir.
        # ------------------------------------------------------------------
        from pycwb.modules.catalog.catalog import Catalog
        catalog_file = os.path.join(str(tmp), config.catalog_dir, Catalog.DEFAULT_FILENAME)

        logger.info("Running process_job_segment for job segment %s", job_segments[0].index)
        process_job_segment(
            working_dir=str(tmp),
            config=config,
            job_seg=job_segments[0],
            compress_json=False,
            catalog_file=catalog_file,
            queue=None,
            production_mode=False,
            skip_lags=None,
        )

        # ------------------------------------------------------------------
        # 5. Return the path to the generated catalog
        # ------------------------------------------------------------------
        result = Path(catalog_file)
        if not result.exists():
            raise FileNotFoundError(f"Pipeline completed but catalog not found at {result}")

        return result

    finally:
        # Restore original working directory
        os.chdir(_saved_cwd)
