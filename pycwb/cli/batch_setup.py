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
    
    parser.add_argument('--cluster',
                        '-c',
                        metavar='job_submission_system',
                        type=str,
                        choices=['condor', 'slurm'],
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
                        default=1,
                        help='the number of jobs per worker')
    
    # container_image
    parser.add_argument('--container-image',
                        '--image',
                        metavar='container_image',
                        type=str,
                        help='the URI to the container image')

    # should_transfer_files
    parser.add_argument('--should-transfer-files',
                        action='store_true',
                        default=False,
                        help='whether transfer files to computing node, set this if the shared file system does not exist')   

    # memory
    parser.add_argument('--memory',
                        '-m',
                        metavar='memory',
                        type=str,
                        help='the memory for each job')

    # disk
    parser.add_argument('--disk',
                        '-k',
                        metavar='disk',
                        type=str,
                        help='the disk space for each job')

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
                input_dir=args.input_dir,
                compress_json=args.compress_json,
                cluster=args.cluster,
                conda_env=args.conda_env,
                additional_init=args.additional_init,
                n_proc=args.n_proc,
                memory=args.memory,
                disk=args.disk,
                accounting_group=args.accounting_group,
                job_per_worker=args.job_per_worker,
                should_transfer_files=args.should_transfer_files,
                container_image=args.container_image,
                submit=args.submit,
                overwrite=args.force_overwrite, 
                config_vars=args.config_vars)
