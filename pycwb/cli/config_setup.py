"""
CLI command for setting up configuration and batch job submission.

This command combines the config repository parser setup with batch job setup,
allowing users to quickly initialize a working directory and optionally submit
the job to a batch system.
"""

import os
import logging

logger = logging.getLogger(__name__)


def init_parser(parser):
    """Initialize argument parser for config-setup command."""
    # Working directory name (project name)
    parser.add_argument('workdir',
                        metavar='workdir',
                        type=str,
                        help='Working directory name (will be created), e.g., O4_K02_C00_BurstLF_LH_BKG_standard')
    
    # Data type
    parser.add_argument('--datatype',
                        '-d',
                        metavar='datatype',
                        type=str,
                        default=None,
                        help='Data source type (e.g. igwn-osg, cit-local, gwosc, local). '
                             'Overrides the data_source defined in the machine profile. '
                             'Valid values are those configured in settings.yaml.')

    # Machine name
    parser.add_argument('--machine',
                        metavar='machine',
                        type=str,
                        default=None,
                        help='Machine profile name (overrides the machine key in settings.yaml). '
                             'Loads machine/<machine>.yaml from the config repository.')
    
    # Config base path
    parser.add_argument('--config-base-path',
                        '-c',
                        metavar='config_base_path',
                        type=str,
                        default='./prototypes/config',
                        help='Base path to configuration repository (default: ./prototypes/config)')
    
    # Cluster type for batch submission
    parser.add_argument('--cluster',
                        metavar='job_submission_system',
                        type=str,
                        choices=['condor', 'slurm'],
                        help='Job submission system to use for batch setup (e.g., condor, slurm)')
    
    # Submit flag
    parser.add_argument('--submit',
                        '-s',
                        action='store_true',
                        default=False,
                        help='Submit the job to the batch system after setup')
    
    # Dry run
    parser.add_argument('--dry-run',
                        action='store_true',
                        default=False,
                        help='Show what would be done without creating files')
    
    # Number of processors
    parser.add_argument('--n-proc',
                        '-n',
                        metavar='n_proc',
                        type=int,
                        help='The number of cpu to use for each condor/slurm job')
    
    # Memory
    parser.add_argument('--memory',
                        '-m',
                        metavar='memory',
                        type=str,
                        help='Memory per job (e.g., 4GB)')
    
    # Disk
    parser.add_argument('--disk',
                        '-k',
                        metavar='disk',
                        type=str,
                        help='Disk space per job (e.g., 8GB)')
    
    # Accounting group
    parser.add_argument('--accounting-group',
                        '-g',
                        metavar='accounting_group',
                        type=str,
                        help='Condor accounting group')
    
    # Container image
    parser.add_argument('--container-image',
                        '--image',
                        metavar='container_image',
                        type=str,
                        help='Container image URI')
    
    # Force overwrite
    parser.add_argument('--force-overwrite',
                        '-f',
                        action='store_true',
                        default=False,
                        help='Overwrite existing results')
    
    # List only
    parser.add_argument('--list-n-jobs',
                        action='store_true',
                        default=False,
                        help='List number of jobs without submitting')

    # SLURM-specific options
    parser.add_argument('--walltime',
                        metavar='walltime',
                        type=str,
                        default=None,
                        help='SLURM wall-clock time limit per job (e.g. 72:00:00, default from config)')

    parser.add_argument('--slurm-constraint',
                        metavar='slurm_constraint',
                        type=str,
                        default=None,
                        help='SLURM node feature constraint (e.g. cal)')

    parser.add_argument('--slurm-partition',
                        metavar='slurm_partition',
                        type=str,
                        default=None,
                        help='SLURM partition to submit to')

    parser.add_argument('--n-retries',
                        metavar='n_retries',
                        type=int,
                        default=5,
                        help='Application-level retries on failure (SLURM only, default: 5)')


def command(args):
    """Execute the config-setup command."""
    from pycwb.modules.config_repo_parser import setup_project
    from pycwb.workflow.batch import batch_setup as batch_setup_func
    
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Step 1: Setup the project directory
        logger.info(f"Setting up project: {args.workdir}")
        
        setup_result = setup_project(
            work_dir=args.workdir,
            config_base_path=args.config_base_path,
            machine=args.machine,
            data_type=args.datatype,
            dry_run=args.dry_run
        )
        
        if args.dry_run:
            logger.info("Dry run completed. No files were created.")
            return 0
        
        logger.info(f"Project setup complete at: {setup_result['work_dir']}")
        logger.info(f"  Config file: {setup_result['config_file']}")
        logger.info(f"  Data type: {setup_result['data_type']}")
        logger.info(f"  GPS time: {setup_result['gps_times']['start']} - {setup_result['gps_times']['stop']}")
        
        # Step 2: Run batch setup to prepare for job submission
        config_file = str(setup_result['config_file'])
        
        if args.list_n_jobs:
            # Just list the number of jobs
            jobs = batch_setup_func(
                config_file,
                working_dir=str(setup_result['work_dir']),
                n_proc=1,
                dry_run=True
            )
            logger.info(f"Number of jobs: {len(jobs)}")
            return 0
        
        logger.info(f"Running batch setup for {config_file}")
        batch_setup_func(
            config_file,
            working_dir=str(setup_result['work_dir']),
            n_proc=args.n_proc,
            memory=args.memory,
            disk=args.disk,
            cluster=args.cluster,
            accounting_group=args.accounting_group,
            container_image=args.container_image,
            walltime=args.walltime,
            slurm_constraint=args.slurm_constraint,
            slurm_partition=args.slurm_partition,
            n_retries=args.n_retries,
            submit=args.submit,
            overwrite=args.force_overwrite
        )
        
        if args.submit:
            logger.info("Job submitted successfully!")
        else:
            logger.info("Batch setup complete. Use --submit flag to submit the job.")
        
        return 0
    
    except Exception as e:
        import traceback
        logger.error(f"Error: {e}\n{traceback.format_exc()}")
        return 1
