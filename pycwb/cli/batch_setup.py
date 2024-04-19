import os


def init_parser(parser):
    # Add the arguments
    parser.add_argument('user_parameter_file',
                        metavar='file_path',
                        type=str,
                        help='the path to the user parameter file')

    parser.add_argument('--cluster',
                        '-c',
                        metavar='job_submission_system',
                        type=str,
                        choices=['condor', 'slurm'],
                        default='condor',
                        help='the submit option, the available options are condor and slurm')

    parser.add_argument('--submit',
                        '-s',
                        action='store_true',
                        default=False,
                        help='submit the jobs to the job submission system')

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

    # conda env, default is current env
    parser.add_argument('--conda-env',
                        '-e',
                        metavar='conda_env',
                        type=str,
                        default=os.environ.get('CONDA_DEFAULT_ENV'),
                        help='the conda environment')

    # additional init
    parser.add_argument('--additional-init',
                        '-a',
                        metavar='additional_init',
                        type=str,
                        help='additional initialization commands')

    # accounting_group
    parser.add_argument('--accounting-group',
                        '-g',
                        metavar='accounting_group',
                        type=str,
                        help='the condor accounting group')

    # threads
    parser.add_argument('--n-proc',
                        '-n',
                        metavar='n_proc',
                        type=int,
                        default=1,
                        help='the number of cpu to use, default to 1. If it set to 0, '
                             'it will use the value from the user parameter file.')

    # job_per_worker
    parser.add_argument('--job-per-worker',
                        '-j',
                        metavar='job_per_worker',
                        type=int,
                        default=5,
                        help='the number of jobs per worker')

    # compress json
    parser.add_argument('--compress_json',
                        action='store_true',
                        default=True,
                        help='compress the json files, by default True')

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
    from pycwb.workflow.batch import batch_setup

    if args.list_n_jobs or args.list_jobs:
        jobs = batch_setup(args.user_parameter_file, working_dir=args.work_dir, n_proc=1, dry_run=True)

        print(f"Number of jobs: {len(jobs)}")

        if args.list_n_jobs:
            print(f"To list all jobs, use --list-jobs option.")

        # list all jobs
        if args.list_jobs:
            for job in jobs:
                print(job)

        return 0

    if args.cluster is 'slurm':
        print("Slurm is not supported yet.")
        return 1

    # Run the search function with the specified user parameter file
    batch_setup(args.user_parameter_file, working_dir=args.work_dir,
                compress_json=args.compress_json,
                cluster=args.cluster,
                conda_env=args.conda_env,
                additional_init=args.additional_init,
                n_proc=args.n_proc,
                accounting_group=args.accounting_group,
                job_per_worker=args.job_per_worker,
                submit=args.submit,
                overwrite=args.force_overwrite, )
