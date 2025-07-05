import os


def init_parser(parser):
    # Add the arguments
    parser.add_argument('user_parameter_file',
                        metavar='file_path',
                        type=str,
                        help='the path to the user parameter file')

    # config_vars
    parser.add_argument('--config-vars',
                        metavar='config_vars',
                        type=str,
                        default=None,
                        help='the config variables to overwrite, in the format of key1=value1,key2=value2')

    parser.add_argument('--input-dir',
                        '-i',
                        metavar='input_dir',
                        type=str,
                        default=None,
                        help='the input directory, the input files will be copied to the working directory, such as the data quality files')
    
    # working dir
    parser.add_argument('--work-dir',
                        '-d',
                        metavar='work_dir',
                        type=str,
                        default='.',
                        help='the working directory')

    # force overwrite
    parser.add_argument('--force-overwrite',
                        '-f',
                        action='store_true',
                        default=False,
                        help='overwrite the existing results')

    # threads
    parser.add_argument('--n-proc',
                        '-n',
                        metavar='n_proc',
                        type=int,
                        default=0,
                        help='the number of cpu to use, if it set to 0, '
                             'it will use the value from the user parameter file.')

    # generate plot
    parser.add_argument('--plot',
                        action='store_true',
                        default=None,
                        help='generate the plot, by default False')

    # compress json
    parser.add_argument('--compress_json',
                        action='store_true',
                        default=False,
                        help='compress the json files, by default False')

    # serve name
    parser.add_argument('--name',
                        metavar='name',
                        type=str,
                        default='pycwb',
                        help='the name of the serve')

    # list number of jobs
    parser.add_argument('--list-n-jobs',
                        action='store_true',
                        default=False,
                        help='list number of the jobs in the flow')

    # list jobs
    parser.add_argument('--list-jobs',
                        action='store_true',
                        default=False,
                        help='list all jobs in the flow')


def command(args):
    from pycwb.workflow.run import search

    if args.list_n_jobs or args.list_jobs:
        jobs = search(args.user_parameter_file, working_dir=args.work_dir, n_proc=1, dry_run=True)

        print(f"Number of jobs: {len(jobs)}")

        if args.list_n_jobs:
            print(f"To list all jobs, use --list-jobs option.")

        # list all jobs
        if args.list_jobs:
            for job in jobs:
                print(job)

        return 0

    # Run the search function with the specified user parameter file
    search(args.user_parameter_file, 
           input_dir=args.input_dir,
           working_dir=args.work_dir, 
           n_proc=args.n_proc,
           overwrite=args.force_overwrite, 
           plot=args.plot, 
           compress_json=args.compress_json, 
           config_vars=args.config_vars)
