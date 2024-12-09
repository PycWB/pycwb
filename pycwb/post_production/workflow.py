import yaml
import copy
from pycwb.utils.module import import_helper


def run_workflow(workflow_file):
    with open(workflow_file, 'r') as f:
        workflow = yaml.safe_load(f)

    # the global arguments will be inserted into each step,
    # the output of each step will be stored in the global arguments
    # if the output_alias if given, the output will be stored in the
    # global arguments with the key of output_alias
    global_args = workflow['global']
    # iterate through each step
    for step in workflow['steps']:
        # get the function, this will be replaced with a module loader
        func_name = step['action'].split('.')[-1]
        module_name = '.'.join(step['action'].split('.')[:-1])
        if not module_name.startswith('pycwb'):
            module_name = f"pycwb.modules.{module_name}"
        module = import_helper(module_name, module_name)
        func = getattr(module, func_name)
        # combine global_args and step['args']
        args = copy.deepcopy(global_args)
        args.update(step['args'])

        print("-"*50)
        print(f"Running action {step['action']} with args {list(args.keys())}")
        result = func(**args)
        if 'output_alias' in step:
            global_args[step['output_alias']] = result
            print(f"Output stored with key: {step['output_alias']}")
        # if result is a dict, add to results
        elif isinstance(result, dict):
            global_args.update(result)