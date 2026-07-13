"""Simulation matching actions for post-production workflows."""

from __future__ import annotations

import logging
import os

from pycwb.modules.catalog.matching import match_simulations_parquet
from pycwb.post_production.action_spec import action_spec

logger = logging.getLogger(__name__)


@action_spec(
    outputs=["output_file"],
    inputs=["catalog_file", "simulation_file"],
    display_name="Match simulations",
    description="Match trigger and simulation-summary parquet catalogs",
    help=(
        "Run the trigger-to-simulation interval matcher inside a "
        "post-production workflow. The simulation_file input should point to "
        "the simulations.parquet summary produced before postproduction."
    ),
)
def match_simulations(
    work_dir: str,
    catalog_file: str,
    simulation_file: str,
    output_file: str,
    how: str = "right",
    window_buffer: float = 0.0,
    **kwargs,
) -> dict:
    """Match a trigger catalog to simulation summary rows and write parquet."""
    catalog_path = _resolve(work_dir, catalog_file)
    simulation_path = _resolve(work_dir, simulation_file)
    output_path = _resolve(work_dir, output_file)

    table = match_simulations_parquet(
        catalog_path,
        simulation_path,
        window_buffer=float(window_buffer),
        how=how,
        output_parquet=output_path,
    )

    logger.info(
        "Matched simulations: %s x %s -> %s (%d rows, how=%s, buffer=%.3fs)",
        catalog_path,
        simulation_path,
        output_path,
        table.num_rows,
        how,
        float(window_buffer),
    )
    return {
        "matched_file": output_file,
        "output_file": output_file,
        "how": how,
        "window_buffer": float(window_buffer),
        "n_rows": int(table.num_rows),
    }


def _resolve(work_dir: str, path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(work_dir, path)
