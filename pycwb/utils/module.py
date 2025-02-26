import importlib
import importlib.util
import sys


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
    func_name = func_str.split('.')[-1]
    module_name = '.'.join(func_str.split('.')[:-1])
    if not module_name.startswith('pycwb'):
        module_name = f"pycwb.modules.{module_name}"
    module = import_helper(module_name, module_name)
    func = getattr(module, func_name)
    return func