"""CLI module for merging catalog and wave files from batch runs."""

import logging
from pycwb.modules.logger import logger_init

logger = logging.getLogger(__name__)

def init_parser(parser):
    """Initialize argument parser for merge command.
    
    Parameters
    ----------
    parser : argparse.ArgumentParser
        The argument parser to configure with merge-related arguments.
    """
    # working dir
    parser.add_argument('--work-dir',
                        '-d',
                        metavar='work_dir',
                        type=str,
                        default='.',
                        help='the working directory')

    parser.add_argument('--wave',
                        '-w',
                        action='store_true',
                        default=False,
                        help='whether to merge the reconstructed waveforms')
    
    parser.add_argument('--catalog-dir',
                        metavar='catalog_dir',
                        type=str,
                        default='catalog',
                        help='the directory containing the catalog files to be merged')

    parser.add_argument('--output-dir',
                        metavar='output_dir',
                        type=str,
                        default='output',
                        help='the directory containing the wave files to be merged')
    
    parser.add_argument('--mlabel',
                        '-m',
                        metavar='mlabel',
                        default=None,
                        help='merge label')

logger = logging.getLogger(__name__)

def command(args):
    """Execute merge command based on parsed arguments.
    
    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments containing work_dir, wave, catalog_dir, output_dir, and mlabel.
    """
    from pycwb.workflow.merge import merge_catalog, merge_wave, merge_progress
    
    logger_init()

    if not args.wave:
        # Default: Run the search function with the specified user parameter file
        merge_catalog(working_dir=args.work_dir, catalog_dir=args.catalog_dir, merge_label=args.mlabel)
        merge_progress(working_dir=args.work_dir, catalog_dir=args.catalog_dir, merge_label=args.mlabel)
    else:
        merge_wave(working_dir=args.work_dir, output_dir=args.output_dir, merge_label=args.mlabel)
    
