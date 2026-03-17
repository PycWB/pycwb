"""
CLI entry point for ``pycwb online <config.yaml>``.
"""


def init_parser(parser):
    parser.add_argument(
        "config",
        metavar="config_file",
        type=str,
        help="Path to the YAML configuration file (with online extension)",
    )
    parser.add_argument(
        "--work-dir", "-d",
        metavar="work_dir",
        type=str,
        default=".",
        help="Working directory for output (default: current directory)",
    )
    parser.add_argument(
        "--n-workers", "-n",
        metavar="N",
        type=int,
        default=None,
        help="Override online_n_workers from config",
    )
    parser.add_argument(
        "--log-level",
        metavar="LEVEL",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )


def command(args):
    from pycwb.workflow.online import OnlineSearchManager

    manager = OnlineSearchManager(
        config_file=args.config,
        working_dir=args.work_dir,
        log_level=args.log_level,
    )

    # CLI override for n_workers
    if args.n_workers is not None:
        manager.n_workers = args.n_workers
        manager.executor._max_workers = args.n_workers

    manager.run()
