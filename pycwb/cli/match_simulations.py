"""
CLI entry point for ``pycwb match-simulations``.

Joins a trigger catalog Parquet file against a simulation summary Parquet file
using a DuckDB interval join and writes the matched table to a new Parquet file.
"""

import logging

logger = logging.getLogger(__name__)


def init_parser(parser):
    parser.add_argument(
        'catalog_parquet',
        metavar='catalog.parquet',
        nargs='?',
        default='catalog/catalog.parquet',
        type=str,
        help='path to the trigger catalog Parquet file (default: catalog/catalog.parquet)',
    )

    parser.add_argument(
        'sim_parquet',
        metavar='simulations.parquet',
        nargs='?',
        default='catalog/simulations.parquet',
        type=str,
        help='path to the simulation summary Parquet file (default: catalog/simulations.parquet)',
    )

    parser.add_argument(
        '--output',
        '-o',
        metavar='output.parquet',
        type=str,
        default=None,
        help=(
            'destination path for the matched Parquet file '
            '(default: matched_<how>.parquet in the current directory)'
        ),
    )

    parser.add_argument(
        '--how',
        choices=['inner', 'left', 'right', 'outer'],
        default='right',
        help=(
            'join type: '
            'inner=matched pairs only; '
            'left=all triggers (NULL sim for unmatched); '
            'right=all simulations (NULL trigger for missed, default); '
            'outer=all triggers and simulations'
        ),
    )

    parser.add_argument(
        '--buffer',
        '-b',
        metavar='seconds',
        type=float,
        default=0.0,
        help='time buffer (seconds) added symmetrically to each trigger window (default: 0)',
    )

    parser.add_argument(
        '--extra-sim-columns',
        metavar='COL',
        nargs='+',
        default=None,
        help='(deprecated — all simulation columns are now included automatically with sim_ prefix)',
    )


def command(args):
    from pycwb.modules.logger import logger_init
    from pycwb.workflow.matching import match_simulations_parquet

    logger_init()

    output = args.output or f'matched_{args.how}.parquet'

    logger.info(
        'Matching %s  ×  %s  (how=%s, buffer=%.3fs)',
        args.catalog_parquet, args.sim_parquet, args.how, args.buffer,
    )

    result = match_simulations_parquet(
        args.catalog_parquet,
        args.sim_parquet,
        window_buffer=args.buffer,
        extra_sim_columns=args.extra_sim_columns,
        how=args.how,
        output_parquet=output,
    )

    print(f'Matched rows : {result.num_rows:,}')
    print(f'Output       : {output}')
