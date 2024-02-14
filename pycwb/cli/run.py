import os


def init_parser(parser):
    # Add the arguments
    parser.add_argument('user_parameter_file',
                        metavar='file_path',
                        type=str,
                        help='the path to the user parameter file')

    parser.add_argument('--submit',
                        '-s',
                        metavar='job_submission_system',
                        type=str,
                        choices=['condor', 'slurm'],
                        help='the submit option, the available options are condor and slurm')

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

    # accounting_group
    parser.add_argument('--accounting-group',
                        '-g',
                        metavar='accounting_group',
                        type=str,
                        default='ligo.sim.o4.burst.allsky.cwboffline',
                        help='the condor accounting group')

    # threads
    parser.add_argument('--threads',
                        '-n',
                        metavar='threads',
                        type=int,
                        default=0,
                        help='the number of threads, if it set to 0, it will use the value from the user parameter file')

    # no subprocess
    parser.add_argument('--no-subprocess',
                        action='store_true',
                        default=False,
                        help='run the search in the main process, by default False (Set to True for macOS development)')

    parser.add_argument('--compress_json',
                        action='store_true',
                        default=False,
                        help='compress the json files, by default False')


def command(args):
    from pycwb.search import search
    from pycwb.modules.condor.condor import generate_job_script, generate_condor_sub, submit

    if args.submit:
        print(f"Submitting the search on {args.submit}.")
        # create a dictionary with the submit option
        if args.submit == 'condor':
            generate_job_script(args.user_parameter_file, args.conda_env, args.work_dir, args.threads)
            generate_condor_sub(args.work_dir, args.accounting_group, args.threads)
            submit(args.work_dir)
        elif args.submit == 'slurm':
            print("Not implemented yet.")

    else:
        print("Running the search locally.")
        # Run the search function with the specified user parameter file
        search(args.user_parameter_file, working_dir=args.work_dir, no_subprocess=args.no_subprocess,
               overwrite=args.force_overwrite, nproc=args.threads, compress_json=args.compress_json)
