import requests
import pathlib
from pycwb.constants.project_constants import XTALK_DATA_URL, XTALK_FILE_LIST_URL


def fetch_txt(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises an exception for 4XX or 5XX errors
        return response.text.splitlines()  # Returns the list of file names
    except requests.RequestException as e:
        return f"An error occurred: {e}"


def download_file(url, file_name):
    #try:
    response = requests.get(url)
    response.raise_for_status()  # Raises an exception for 4XX or 5XX errors
    with open(file_name, "wb") as f:
        f.write(response.content)
    #     print(f"File {file_name} has been downloaded successfully.")
    # except requests.RequestException as e:
    #     print(f"An error occurred: {e}")


def check_and_download_xtalk_data(file_name: str, output_dir: str = ".")\
        -> bool:
    # Fetch and print the file list
    file_list = fetch_txt(XTALK_FILE_LIST_URL)

    # replace any file extension (.bin, .xbin) with .npz
    # new_file_name = pathlib.Path(file_name).with_suffix(".npz").name
    file_name = pathlib.Path(file_name).name

    # Check if the file is in the list
    if file_name in file_list:
        # make output_dir if it does not exist
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Download the file
        print(f"File {file_name} is available on {XTALK_DATA_URL}.")
        download_url = f"{XTALK_DATA_URL}/{file_name}"
        print(f"Downloading {download_url}...")
        download_file(download_url, f"{output_dir}/{file_name}")
        print(f"File saved as {output_dir}/{file_name}")
        return True
    else:
        print(f"File {file_name} is not available on {XTALK_DATA_URL}.")
        return False