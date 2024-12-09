import os


def init_parser(parser):
    # Add the arguments
    parser.add_argument('workflow_file',
                        metavar='file_path',
                        type=str,
                        help='the path to the workflow file')


def command(args):
    from pycwb.post_production.workflow import run_workflow

    run_workflow(args.workflow_file)