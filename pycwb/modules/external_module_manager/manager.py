import os
import subprocess
import shutil
import tempfile
import logging
from ...utils.yaml_helper import load_yaml
from .config_schema import schema


logger = logging.getLogger(__name__)

def load_config(external_module_config, config_schema=schema):
    """
    Load the external module configuration from a file.

    Args:
        external_module_config (str): Path to the external module configuration YAML file.

    Returns:
        dict: Loaded configuration as a dictionary.
    """
    params = load_yaml(external_module_config, config_schema)

    target_dir = params.get("target_dir")
    modules = params.get("modules", [])
    return target_dir, modules


def check_module_existence(module_name, target_dir=None):
    """
    Check if the specified module exists in the the pycwb installation directory.

    Args:
        module_name (str): Name of the module to check.

    Returns:
        bool: True if the module exists, False otherwise.
    """
    if target_dir is None:
        import pycwb
        target_dir = os.path.join(os.path.dirname(pycwb.__file__), 'modules')

    module_path = os.path.join(target_dir, module_name)
    return os.path.exists(module_path)


def pull_external_module(module_name, module_path, repo_url, target_dir=None):
    """
    Pull the specified external module from the repository.

    Args:
        module_name (str): Name of the module to pull.
        module_path (str): Path to the module in the repository.
        repo_url (str): URL of the repository to pull from.
        target_dir (str, optional): Directory to check for the module. Defaults to None.

    Returns:
        bool: True if the module was pulled successfully, False otherwise.
    """
    if target_dir is None:
        import pycwb
        target_dir = os.path.join(os.path.dirname(pycwb.__file__), 'modules')

    dest_path = os.path.join(target_dir, module_name)

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger.info(f"Cloning {repo_url} into {tmp_dir}")
            subprocess.run(['git', 'clone', '--depth', '1', repo_url, tmp_dir], check=True)

            source_path = os.path.join(tmp_dir, module_path)
            if not os.path.exists(source_path):
                logger.error(f"Module path '{module_path}' does not exist in the repo.")
                return False

            if os.path.exists(dest_path):
                logger.info(f"Removing existing directory at {dest_path}")
                shutil.rmtree(dest_path)

            shutil.copytree(source_path, dest_path)
            logger.info(f"Module copied to {dest_path}")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"Git clone failed: {e}")
    except Exception as e:
        logger.error(f"Error while copying module: {e}")

    return False
    

    