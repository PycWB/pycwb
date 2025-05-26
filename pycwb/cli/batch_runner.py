import os


def init_parser(parser):
    # Add the arguments
    parser.add_argument('user_parameter_file',
                        metavar='file_path',
                        type=str,
                        help='the path to the user parameter file')

    # working dir
    parser.add_argument('--work-dir',
                        '-d',
                        metavar='work_dir',
                        type=str,
                        default='.',
                        help='the working directory')

    # threads
    parser.add_argument('--n-proc',
                        '-n',
                        metavar='n_proc',
                        type=int,
                        default=1,
                        help='the number of cpu to use, default to 1. If it set to 0, '
                             'it will use the value from the user parameter file.')

    # jobs
    parser.add_argument('--jobs',
                        '-j',
                        metavar='jobs',
                        type=str,
                        default=None,
                        help='the range of jobs to run, e.g., 0-9')

    # compress json
    parser.add_argument('--compress_json',
                        action='store_true',
                        default=True,
                        help='compress the json files, by default True')
    
    # n_workers
    parser.add_argument('--n-workers',
                        '-w',
                        metavar='n_workers',
                        type=int,
                        default=1,
                        help='the number of workers to use, default to 1. If it set to 0, '
                             'it will use the value from the user parameter file.')


def command(args):
    from pycwb.workflow.batch import batch_run

    # Run the search function with the specified user parameter file
    batch_run(args.user_parameter_file, working_dir=args.work_dir,
              jobs=args.jobs, compress_json=args.compress_json, n_proc=args.n_proc, n_workers=args.n_workers)
