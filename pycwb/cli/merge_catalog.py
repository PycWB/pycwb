import os

def init_parser(parser):
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

def command(args):
    from pycwb.workflow.merge import merge_catalog, merge_wave
    if not args.wave:
        # Default: Run the search function with the specified user parameter file
        merge_catalog(working_dir=args.work_dir)
    else:
        merge_wave(working_dir=args.work_dir)
    
