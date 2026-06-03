"""Workflow runner for pycwb post-production YAML pipelines.

Parses a YAML workflow configuration, resolves action functions dynamically,
and executes them sequentially with a persistent global context.

Diagram generation
------------------
If *generate_diagram* is True (the default), a Mermaid (``.mmd``) and
Graphviz (``.png``) DAG diagram is generated before execution begins.
The diagram shows each step as a node and file/alias dependencies as edges.
Set ``generate_diagram=False`` or use ``--no-diagram`` in the CLI to skip.
"""

import copy
import logging
import os

import yaml

from pycwb.utils.module import import_helper

logger = logging.getLogger(__name__)


def run_workflow(workflow_file: str, generate_diagram: bool = True):
    """Execute a YAML-defined post-production workflow.

    Parameters
    ----------
    workflow_file : str
        Path to the YAML workflow file.
    generate_diagram : bool
        If True, generate a DAG diagram (``.mmd`` + ``.png``) before
        executing the steps.  The diagram is written to
        ``<work_dir>/workflow_diagram.*``.
    """
    with open(workflow_file, 'r') as f:
        workflow = yaml.safe_load(f)

    # the global arguments will be inserted into each step,
    # the output of each step will be stored in the global arguments
    # if the output_alias if given, the output will be stored in the
    # global arguments with the key of output_alias
    global_args = workflow['global']

    # ── Generate DAG diagram ────────────────────────────────────────────
    if generate_diagram:
        try:
            from pycwb.post_production.diagram import generate_workflow_diagram
            work_dir = global_args.get('work_dir', '.')
            output_prefix = os.path.join(work_dir, 'workflow_diagram')
            result = generate_workflow_diagram(
                workflow_file,
                output_prefix=output_prefix,
                generate_png=True,
            )
            dag = result.get('dag', {})
            n_nodes = len(dag.get('nodes', []))
            n_edges = len(dag.get('edges', []))
            logger.info(
                'Workflow diagram: %d nodes, %d edges → %s',
                n_nodes, n_edges,
                ', '.join(str(v) for v in result.values() if v),
            )
        except Exception:
            logger.warning(
                'Failed to generate workflow diagram — continuing without it',
                exc_info=True,
            )

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