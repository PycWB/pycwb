import os

from pycwb.modules.xtalk.monster import load_catalog


def init_parser(parser):
    # Add the arguments
    parser.add_argument('xtalk_file',
                        metavar='file_path',
                        type=str,
                        help='the path to the user parameter file')

    parser.add_argument('--output_dir',
                        '-o',
                        metavar='output_dir',
                        type=str,
                        default='.',
                        help='the output directory')


def command(args):
    # Create the output directory if it does not exist
    if not os.path.exists(args.output_dir):
        print(f"Creating output directory: {args.output_dir}")
        os.makedirs(args.output_dir)
    else:
        print(f"Output directory already exists: {args.output_dir}")

    # change to the output directory
    print(f"Changing to output directory: {args.output_dir}")
    os.chdir(args.output_dir)

    # Load the xtalk file and dump it to current directory
    print(f"Loading xtalk file: {args.xtalk_file}")
    load_catalog(args.xtalk_file, dump=True)
    print(f"Xtalk file loaded and dumped to current directory: {args.output_dir}")