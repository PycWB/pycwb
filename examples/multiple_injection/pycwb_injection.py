from pathlib import Path

from pycwb.workflow.run import search


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    search(str(directory / "user_parameters_injection.yaml"), working_dir=str(directory))
