import importlib
import importlib.util
import sys
import os
import logging

logger = logging.getLogger(__name__)


def import_helper(module_str, module_name):
    if module_str.endswith('.py'):
        spec = importlib.util.spec_from_file_location(module_name, module_str)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_str)

    return module


def import_function(func_str):
    """
    Import a function based on the given function string which can be a file path or a module path.

    :param func_str: The function string in the format 'path/to/file.func' or 'module.path.func'
    :return: The imported function
    """
    # Split the function string into path_part and function name
    if '.' not in func_str:
        raise ValueError(f"Invalid function string: {func_str}. Expected format 'path/to/file.func' or 'module.path.func'")

    # Split into two parts: everything except the last component, and the last component
    parts = func_str.rsplit('.', 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid function string: {func_str}. Expected exactly one '.' before the function name")

    path_part, func_name = parts

    is_file = False
    file_path = None

    # Check if the path_part resembles a file path (contains slashes or is absolute)
    if any(sep in path_part for sep in ('/', '\\')) or os.path.isabs(path_part):
        # Check possible file paths
        if not path_part.endswith('.py'):
            # Check if adding .py makes the file exist
            py_candidate = path_part + '.py'
            if os.path.isfile(py_candidate):
                file_path = py_candidate
                is_file = True
            elif os.path.isfile(path_part):
                file_path = path_part
                is_file = True
        else:
            if os.path.isfile(path_part):
                file_path = path_part
                is_file = True

    if is_file:
        # Import from file
        module_name = os.path.splitext(os.path.basename(file_path))[0]
        module = import_helper(file_path, module_name)
    else:
        # Attempt to import as a module
        try:
            module = import_helper(path_part, path_part)
        except ImportError as e:
            # If the import failed, try prepending 'pycwb.modules' if not already a pycwb module
            if not path_part.startswith('pycwb.'):
                new_module_name = f'pycwb.modules.{path_part}'
                try:
                    module = import_helper(new_module_name, new_module_name)
                except ImportError:
                    raise ImportError(
                        f"Could not import module '{path_part}' or '{new_module_name}'. Please check the path.") from e
            else:
                # Re-raise the original error if it's already a pycwb module
                raise

    # Retrieve the function from the module
    func = getattr(module, func_name, None)
    if func is None:
        raise AttributeError(f"Function '{func_name}' not found in module '{module.__name__}'")

    logger.info(f"Successfully imported function '{func_name}' from '{path_part}'")
    return func


def import_function_from_file(file, func_name):
    module_name = file.split('/')[-1].split('.')[0]
    module = import_helper(file, module_name)

    func = getattr(module, func_name)
    logger.info(f"Imported function {func_name} from {file}")
    return func
