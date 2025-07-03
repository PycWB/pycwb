from gwosc.datasets import event_gps, event_detectors, event_segment
from gwosc.locate import get_urls
from gwosc.timeline import get_segments
import os
import requests

def download_frames_files(output_dir, detectors, gps_start, gps_end, sample_rate=16384):
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


    for detector in detectors:
        urls = get_urls(detector=detector, start=gps_start, end=gps_end, format='gwf', sample_rate=sample_rate)
        
        # Exclude files with '-32.gwf'
        urls = [url for url in urls if "-32.gwf" not in url] 
        
        frame_dir = os.path.join(f"{output_dir}/frames/", f"{detector}_frames")
        os.makedirs(frame_dir, exist_ok=True)

        local_paths = []

        # Download files
        for url in urls:
            frame_file_path = os.path.join(frame_dir, os.path.basename(url))
            if os.path.exists(frame_file_path):
                print(f"File {frame_file_path} already exists, skipping download.")
                local_paths.append(frame_file_path)
                continue
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


def get_dq_files(output_dir, detectors, gps_start, gps_end):
    """
    Generate files containing Data Quality (DQ) segments for the given start and end times.

    Parameters:
    -----------
    output_dir : str
        The directory where the category files will be saved.
    detectors : list of str
        List of detector names (e.g., ['H1', 'L1']).
    gps_start : int
        The GPS start time for the segments.
    gps_end : int
        The GPS end time for the segments.

    Returns:
    --------
    None
    """
    os.makedirs(output_dir, exist_ok=True)

    for detector in detectors:
        for category, cat_name in [
            (f'{detector}_DATA', f'{detector}_cat0.txt'),
            (f'{detector}_BURST_CAT1', f'{detector}_cat1.txt'),
            (f'{detector}_BURST_CAT2', f'{detector}_cat2.txt'),
        ]:
            segments = get_segments(category, gps_start, gps_end)
            cat_file_path = os.path.join(output_dir, cat_name)

            with open(cat_file_path, 'w') as cat_file:
                for segment in segments:
                    cat_file.write(f"{segment[0]} {segment[1]}\n")

            print(f"Segments saved in {cat_file_path}")