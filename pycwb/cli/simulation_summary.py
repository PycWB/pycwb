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
        default='config/user_parameters.yaml',
        help='path to the pycwb YAML configuration file (default: config/user_parameters.yaml)',
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

    # ── Load configuration ────────────────────────────────────────────────
    # config_vars are applied as Jinja2 template substitutions before YAML
    # parsing — the same approach used by prepare_job_runs.py.
    config_file = args.user_parameter_file
    if args.config_vars:
        from pycwb.workflow.subflow.prepare_job_runs import generate_config
        config_file = generate_config(config_file, args.config_vars)

    config = Config()
    config.load_from_yaml(config_file)

    if not config.injection:
        logger.error(
            "No 'injection' block found in %s — nothing to summarise.",
            args.user_parameter_file,
        )
        raise SystemExit(1)

    # ── Build job segments ────────────────────────────────────────────────
    logger.info("Building job segments from config …")
    job_segments = create_job_segment_from_config(config)
    logger.info("%d job segment(s) created.", len(job_segments))

    # ── Resolve output path ───────────────────────────────────────────────
    output_file = args.output
    if output_file is None:
        output_file = os.path.join(args.work_dir, 'catalog', 'simulations.parquet')

    # ── Run summary ───────────────────────────────────────────────────────
    df = build_simulation_summary(config, job_segments, output_file=output_file)

    logger.info(
        "Simulation summary complete: %d row(s) written to %s",
        len(df), output_file,
    )
