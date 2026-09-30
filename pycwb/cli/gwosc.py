import os
import shutil
from collections.abc import Sequence
from pathlib import Path

import yaml

import pycwb


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


def configure_downloaded_frames(
    parameter_file: str | os.PathLike[str],
    input_dir: str | os.PathLike[str],
    ifos: Sequence[str],
) -> None:
    """Match the bundled template to the actual 4 kHz GWOSC frame release."""
    from gwpy.io.gwf import get_channel_names

    input_dir = Path(input_dir).resolve()
    channels = []
    for ifo in ifos:
        frame_list = input_dir / f"{ifo}_frames.in"
        frames = frame_list.read_text().splitlines()
        if not frames:
            raise ValueError(f"No downloaded frames for {ifo}")
        strain = [name for name in get_channel_names(frames[0])
                  if name.startswith(f"{ifo}:") and name.endswith("STRAIN")]
        if len(strain) != 1:
            raise ValueError(f"Expected one strain channel for {ifo}, found {strain}")
        channels.append(strain[0])

    path = Path(parameter_file)
    params = yaml.safe_load(path.read_text())
    params.update(ifo=list(ifos), refIFO=ifos[0], inRate=4096, levelR=1,
                  channelNamesRaw=channels,
                  frFiles=[str(input_dir / f"{ifo}_frames.in") for ifo in ifos])
    params["DQF"] = [
        [ifo, str(input_dir / f"{ifo}_cat{category}.txt"), f"CWB_CAT{category}", 0., False, False]
        for category in range(3) for ifo in ifos
    ] + [[ifo, str(input_dir / "cwb_period.txt"), "CWB_CAT0", 0., False, False]
         for ifo in ifos]
    path.write_text(yaml.safe_dump(params, sort_keys=False))


def command(args):
    from pycwb.modules.gwosc.gwosc import (
        analysis_period,
        download_frames_files,
        event_info,
        get_cat_files,
    )

    if not all(ifo in ["H1", "L1"] for ifo in args.ifos) and not args.user_parameters_path:
        raise ValueError("Only H1 and L1 are supported in ifos with the default user_parameters file, "
                         "please provide a custom user_parameters file")

    output = './input'
    
    use_template = args.user_parameters_path is None
    if use_template:
        package_abs_path = os.path.dirname(os.path.abspath(pycwb.__file__))
        args.user_parameters_path = os.path.join(package_abs_path, 'vendor/template/gwosc')
        print(f'{args.user_parameters_path}')

    download_frames_files(args.event_name, output, args.ifos, args.time_left, args.time_right)
    get_cat_files(args.event_name, output, args.ifos, args.time_left, args.time_right)
    analysis_period(args.event_name, output, args.time_left, args.time_right, args.ifos)
    copy_user_parameters(args.user_parameters_path)
    if use_template:
        detectors, _, _, _ = event_info(args.event_name, args.ifos)
        configure_downloaded_frames("user_parameters.yaml", output, detectors)
