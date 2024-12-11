from gwosc.datasets import event_gps, event_detectors, event_segment
from gwosc.locate import get_urls
from gwosc.timeline import get_segments
import os
import requests

def event_info(event_name, ifos):
    """
    Retrieve key information about the specified gravitational wave event.

    Parameters:
    -----------
    event_name : str
        The name of the gravitational wave event (e.g., "GW150914").

    Returns:
    --------
    tuple :
        - A list of detectors that observed the event, filtered to include only allowed detectors.
        - The GPS time of the event.
        - The start and end GPS time of the event segment.
    """
    allowed_cWB_detectors = ifos
    detectors = event_detectors(event_name)  
    detectors = [detector for detector in detectors if detector in allowed_cWB_detectors]
    event_gps_time = event_gps(event_name)
    start_time, end_time = event_segment(event_name)

    return detectors, event_gps_time, start_time, end_time

def download_frames_files(event_name, output_dir, ifos):
    """
    Download frame files for the given gravitational wave event and save a list of their paths.

    Parameters:
    -----------
    event_name : str
        The name of the gravitational wave event (e.g., "GW150914").
    output_dir : str
        The directory where frame files and associated lists will be saved.

    Returns:
    --------
    None
    """
    os.makedirs(output_dir, exist_ok=True)

    detectors, event_gps_time, start_time, end_time = event_info(event_name, ifos)

    for detector in detectors:
        urls = get_urls(detector=detector, start=start_time, end=end_time, dataset=event_name, format='gwf', sample_rate=4096)
        
        # Exclude files with '-32.gwf'
        urls = [url for url in urls if "-32.gwf" not in url] 
        
        frame_dir = os.path.join(f"{output_dir}/frames/", f"{detector}_frames")
        os.makedirs(frame_dir, exist_ok=True)

        local_paths = []

        # Download files
        for url in urls:
            frame_file_path = os.path.join(frame_dir, os.path.basename(url))
            print(f"Downloading {frame_file_path} from {url}...")
            response = requests.get(url)
            with open(frame_file_path, 'wb') as frame_file:
                frame_file.write(response.content)
            local_paths.append(frame_file_path)

        # Save *_frames.in file
        frame_list_file = os.path.join(output_dir, f"{detector}_frames.in")
        with open(frame_list_file, 'w') as f:
            for path in local_paths:
                f.write(path + "\n")

        print(f"Frame list saved to {frame_list_file}")

def get_cat_files(event_name, output_dir, ifos):
    """
    Generate files containing Data Quality (DQ) segments for the given gravitational wave event.

    Parameters:
    -----------
    event_name : str
        The name of the gravitational wave event (e.g., "GW150914").
    output_dir : str
        The directory where DQ segment files will be saved.

    Returns:
    --------
    None
    """
    os.makedirs(output_dir, exist_ok=True)

    detectors, event_gps_time, start_time, end_time = event_info(event_name, ifos)

    for detector in detectors:
        for category, cat_name in [
            (f'{detector}_DATA', f'{detector}_cat0.txt'),
            (f'{detector}_BURST_CAT1', f'{detector}_cat1.txt'),
            (f'{detector}_BURST_CAT2', f'{detector}_cat2.txt'),
        ]:
            segments = get_segments(category, start_time, end_time)
            cat_file_path = os.path.join(output_dir, cat_name)

            with open(cat_file_path, 'w') as cat_file:
                for segment in segments:
                    cat_file.write(f"{segment[0]} {segment[1]}\n")

            print(f"Segments saved in {cat_file_path}")

def analysis_period(event_name, output_dir, time_left, time_right, ifos):
    """
    Calculate the analysis period for cWB based on the event GPS time and save it to a file.

    This function calculates the start and end times for the cWB analysis period by subtracting
    and adding the specified offsets (`time_left` and `time_right`) to the GPS time of the 
    gravitational wave event. The resulting period is saved in a file named `cwb_period.txt`.

    Parameters:
    -----------
    event_name : str
        The name of the gravitational wave event (e.g., "GW150914").
    time_left : int, optional
        Time in seconds to subtract from the event GPS time to determine the start of the analysis period
        (default is 610 seconds).
    time_right : int, optional
        Time in seconds to add to the event GPS time to determine the end of the analysis period
        (default is 610 seconds).

    Returns:
    --------
    None
    """
    _, event_gps_time, _, _ = event_info(event_name, ifos)
    print(f"Event gps time: {event_gps_time}")
    analysis_period = [event_gps_time - time_left, event_gps_time + time_right]

    cwb_period_file_path = os.path.join(output_dir, "cwb_period.txt")
    with open(cwb_period_file_path, 'w') as cwb_period_file:
        cwb_period_file.write(f"{analysis_period[0]} {analysis_period[1]}\n")
    print(f"cWB analysis period saved in {cwb_period_file_path}")