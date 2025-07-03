import os
from math import floor, ceil
from pycwb.config import Config
from pycwb.modules.gwosc.utils import download_frames_files, get_dq_files


def init_parser(parser):
    parser.add_argument('user_parameter_file',
                        metavar='file_path',
                        type=str,
                        help='the path to the user parameter file')

    # working dir
    parser.add_argument('--work-dir',
                        '-d',
                        metavar='work_dir',
                        type=str,
                        default='.',
                        help='the working directory')


def command(args):
    config = Config()
    config.load_from_yaml(args.user_parameter_file)

    sample_rate = config.inRate
    seg_edge = config.segEdge
    gps_start = floor(config.gps_start)
    gps_end = ceil(config.gps_end)
    data_start = gps_start - seg_edge
    data_end = gps_end + seg_edge

    ifos = config.ifo
    work_dir = args.work_dir

    download_frames_files(os.path.join(work_dir, 'input'), 
                          detectors=ifos,
                          gps_start=data_start,
                          gps_end=data_end,
                          sample_rate=sample_rate)
    
    get_dq_files(os.path.join(work_dir, 'input'),
                 detectors=ifos,
                 gps_start=data_start,
                 gps_end=data_end)
    