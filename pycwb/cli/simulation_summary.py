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
        type=str,
        help='path to the pycwb YAML configuration file',
    )

    parser.add_argument(
        '--output',
        '-o',
        metavar='output.parquet',
        type=str,
        default=None,
        help=(
            'destination path for the Parquet summary file '
            '(default: <work_dir>/simulation_summary.parquet)'
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
    config_vars = {}
    if args.config_vars:
        for pair in args.config_vars.split(','):
            k, _, v = pair.partition('=')
            config_vars[k.strip()] = v.strip()

    config = Config.load_from_yaml(args.user_parameter_file, config_vars=config_vars or None)

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
        output_file = os.path.join(args.work_dir, 'simulation_summary.parquet')

    # ── Run summary ───────────────────────────────────────────────────────
    df = build_simulation_summary(config, job_segments, output_file=output_file)

    logger.info(
        "Simulation summary complete: %d row(s) written to %s",
        len(df), output_file,
    )
