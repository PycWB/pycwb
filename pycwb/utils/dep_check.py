import os
import yaml
import pycwb
from importlib.metadata import distribution, PackageNotFoundError


def is_package_installed(package_name):
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False


def check_dependencies(module_list):
    """Check if the dependencies of the given modules are installed

    Parameters
    ----------
    module_list : list
        List of modules to check

    Returns
    -------
    missing_packages : list
        List of missing packages
    """
    print("Checking dependencies ...")

    base_dir = f'{os.path.dirname(os.path.abspath(pycwb.__file__))}/modules'

    req_packages = []

    for module in module_list:
        req_packages += check_deps_for_module(base_dir, module)

    req_packages = list(set(req_packages))

    missing_packages = []
    for dep in req_packages:
        try:
            distribution(dep)
        except PackageNotFoundError:
            # test non-pip packages
            if not is_package_installed(dep):
                missing_packages.append(dep)

    if missing_packages:
        # Ask user to install missing packages
        print("The following packages are missing:")
        for package in missing_packages:
            print(package)
        print("Please install them and try again.")
    else:
        print("All dependencies exist.")

    return missing_packages


def check_deps_for_module(base_dir, module):
    req_packages = []
    file_path = os.path.join(base_dir, module, "module.yaml")

    # check if the yaml file exists
    if not os.path.isfile(file_path):
        print(f"No module.yaml found for module {module}")
        return []

    # load yaml
    with open(file_path, 'r') as f:
        module_info = yaml.safe_load(f)

    # check if 'dependencies' key exists
    if 'dependencies' not in module_info:
        print(f"No dependencies found for module {module}")
        return []

    # check if the dependencies are installed
    dependencies = module_info['dependencies']

    for dep in dependencies:
        # if dep start with pycwb., it is a pycwb module, skip
        if dep.startswith('@pycwb.'):
            continue

        # if dep start with @, it is a module, check recursively
        if dep.startswith('@'):
            req_packages += check_deps_for_module(base_dir, dep[1:])
            continue

        req_packages.append(dep)

    return req_packages
