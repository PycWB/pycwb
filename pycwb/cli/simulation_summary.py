"""
CLI entry point for ``pycwb simulation-summary <config.yaml>``.

Builds a per-simulation summary Parquet file that describes every simulated
signal recorded in the job segments: waveform extent (real_start / real_end),
the containing segment, and CAT0 / CAT1 / CAT2 veto flags.
"""


def init_parser(parser):
    parser.add_argument(
        'user_parameter_file',
        metavar='config.yaml',
        nargs='?',
        default=None,
        help='config path relative to the caller (default: <work-dir>/config/user_parameters.yaml)',
    )

    parser.add_argument(
        '--output',
        '-o',
        metavar='output.parquet',
        type=str,
        default=None,
        help=(
            'destination path for the Parquet summary file '
            '(default: <work_dir>/catalog/simulations.parquet)'
        ),
    )

    parser.add_argument(
        '--work-dir',
        '-d',
        metavar='work_dir',
        type=str,
        default='.',
        help='working directory used to resolve relative paths (default: .)',
    )

    parser.add_argument(
        '--config-vars',
        metavar='key=value,...',
        type=str,
        default=None,
        help='comma-separated key=value pairs to override config fields',
    )

    parser.add_argument(
        '--log-level',
        metavar='level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='logging level (default: INFO)',
    )


def command(args):
    import logging
    import os

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )
    logger = logging.getLogger(__name__)

    from pycwb.config import Config
    from pycwb.modules.job_segment import create_job_segment_from_config
    from pycwb.workflow.subflow.simulation_summary import build_simulation_summary

    # Explicit CLI paths are relative to the caller; paths inside the config
    # are relative to the production directory, as in batch-setup/batch-runner.
    working_dir = os.path.abspath(args.work_dir)
    config_file = (os.path.abspath(args.user_parameter_file) if args.user_parameter_file
                   else os.path.join(working_dir, 'config', 'user_parameters.yaml'))
    output_file = (os.path.abspath(args.output) if args.output else
                   os.path.join(working_dir, 'catalog', 'simulations.parquet'))
    previous_dir = os.getcwd()
    try:
        os.chdir(working_dir)
        # ── Load configuration ────────────────────────────────────────────────
        # config_vars are applied as Jinja2 template substitutions before YAML
        # parsing — the same approach used by prepare_job_runs.py.
        if args.config_vars:
            from pycwb.workflow.subflow.prepare_job_runs import generate_config
            config_file = generate_config(config_file, args.config_vars)

        config = Config()
        config.load_from_yaml(config_file)

        if not config.injection:
            logger.error(
                "No 'injection' block found in %s — nothing to summarise.",
                config_file,
            )
            raise SystemExit(1)

        # ── Build job segments ────────────────────────────────────────────────
        logger.info("Building job segments from config …")
        job_segments = create_job_segment_from_config(config)
        logger.info("%d job segment(s) created.", len(job_segments))

        # ── Run summary ───────────────────────────────────────────────────────
        df = build_simulation_summary(config, job_segments, output_file=output_file)

    finally:
        os.chdir(previous_dir)

    logger.info(
        "Simulation summary complete: %d row(s) written to %s",
        len(df), output_file,
    )
