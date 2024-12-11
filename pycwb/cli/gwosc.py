import os
import shutil
import pycwb
from pycwb.modules.gwosc.gwosc import download_frames_files
from pycwb.modules.gwosc.gwosc import get_cat_files
from pycwb.modules.gwosc.gwosc import analysis_period

def init_parser(parser):

    # Select the GW event
    parser.add_argument('event_name',
                        metavar='event_name',  # Corrected metavar
                        type=str,
                        help='The name of the GW event you want (e.g., "GW150914").')

    # Time left for cWB analysis
    parser.add_argument('--time_left',
                        metavar='time_left',
                        type=float,
                        default=610, 
                        help='The pycWB analysis interval is '
                             '[event_gps_time - time_left; event_gps_time + time_right]. Default: 610 seconds.')

    # Time right for cWB analysis
    parser.add_argument('--time_right',
                        metavar='time_right',
                        type=float,
                        default=610, 
                        help='The pycWB analysis interval is: '
                             '[event_gps_time - time_left; event_gps_time + time_right]. Default: 610 seconds.')

    # List of detectors
    parser.add_argument('--ifos',
                        metavar='ifos',
                        type=str,
                        nargs='+', 
                        default=['H1', 'L1'], 
                        help='List of the detectors you want data from. For example: --ifos H1 L1. The default is H1 L1.')
    
    parser.add_argument('--user_parameters_path',
                        metavar='user_parameters_path',
                        type=str,
                        default=None, 
                        help='Path of the user_parameters file, default pycwb/vendor/template/gwosc/')
    

def copy_user_parameters(user_parameters_path):

    
    file_to_copy = os.path.join(user_parameters_path, 'user_parameters.yaml')  

    # Check if the file exists
    if not os.path.exists(file_to_copy):
        raise FileNotFoundError(f"The file {file_to_copy} does not exist. The default file is {file_to_copy}")
    
    # Define the destination directory for the event
    destination_dir = os.path.join(".")
    os.makedirs(destination_dir, exist_ok=True)  # Ensure the directory exists

    # Define the destination file path
    destination_file = os.path.join(destination_dir, os.path.basename(file_to_copy))
    
    # Copy the file
    shutil.copy(file_to_copy, destination_file)
    print(f"Copied {file_to_copy} to {destination_file}")


def command(args):

    if not all(ifo in ["H1", "L1"] for ifo in args.ifos) and not args.user_parameters_path:
        raise ValueError("Only H1 and L1 are supported in ifos with the default user_parameters file, "
                         "please provide a custom user_parameters file")

    output = './input'
    
    if args.user_parameters_path is None:
        package_abs_path = os.path.dirname(os.path.abspath(pycwb.__file__))
        args.user_parameters_path = os.path.join(package_abs_path, 'vendor/template/gwosc')
        print(f'{args.user_parameters_path}')

    download_frames_files(args.event_name, output, args.ifos)
    get_cat_files(args.event_name, output, args.ifos)
    analysis_period(args.event_name, output, args.time_left, args.time_right, args.ifos)
    copy_user_parameters(args.user_parameters_path)