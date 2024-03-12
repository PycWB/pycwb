import os
import asyncio

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
                        default=False,
                        help='generate the plot, by default False')

    # compress json
    parser.add_argument('--compress_json',
                        action='store_true',
                        default=False,
                        help='compress the json files, by default False')

    # serve
    parser.add_argument('--serve',
                        action='store_true',
                        default=False,
                        help='Run prefect flow in serve mode')

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
    from pycwb.prefect_flow.pycwb_flow import search

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
    
    if args.serve:
        search.serve(name=args.name)
    else:
        # Run the search function with the specified user parameter file
        asyncio.run(search(args.user_parameter_file, working_dir=args.work_dir, n_proc=args.n_proc, submit=args.submit,
               overwrite=args.force_overwrite, plot=args.plot, compress_json=args.compress_json))
