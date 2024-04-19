import os


def init_parser(parser):
    # working dir
    parser.add_argument('--work-dir',
                        '-d',
                        metavar='work_dir',
                        type=str,
                        default='.',
                        help='the working directory')


def command(args):
    from pycwb.workflow.merge import merge_catalog

    # Run the search function with the specified user parameter file
    merge_catalog(working_dir=args.work_dir)
