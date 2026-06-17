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

import logging
import os
import shutil

from pycwb.post_production.workflow_config import (
    load_workflow,
    prepare_step_args,
    store_result,
    workflow_base_context,
    workflow_runtime,
)
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
    workflow = load_workflow(workflow_file)

    # the global arguments will be inserted into each step,
    # the output of each step will be stored in the global arguments
    # if the output_alias if given, the output will be stored in the
    # global arguments with the key of output_alias
    global_args = workflow_base_context(workflow)
    runtime = workflow_runtime(workflow, global_args)

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
    cleanup_mode = str(runtime.get("cleanup_tmp", "never")).lower()
    completed = False
    try:
        for step in workflow['steps']:
            func = _resolve_action(step['action'])
            args = prepare_step_args(step, global_args, runtime)

            print("-"*50)
            label = step.get("name") or step.get("id") or step["action"]
            print(f"Running action {label} ({step['action']}) with args {list(args.keys())}")
            result = func(**args)
            store_result(global_args, step, result)
            if step.get("id"):
                print(f"Output stored with id: {step['id']}")
            if 'output_alias' in step:
                print(f"Output stored with alias: {step['output_alias']}")
        completed = True
    finally:
        if cleanup_mode == "always" or (cleanup_mode == "on_success" and completed):
            _cleanup_tmp(runtime, global_args)


def _resolve_action(action: str):
    """Import and return the callable referenced by *action*."""
    func_name = action.split('.')[-1]
    module_name = '.'.join(action.split('.')[:-1])
    if not module_name.startswith('pycwb'):
        module_name = f"pycwb.modules.{module_name}"
    module = import_helper(module_name, module_name)
    return getattr(module, func_name)


def _cleanup_tmp(runtime: dict, context: dict) -> None:
    """Remove the configured temporary directory after execution."""
    tmp_dir = os.path.abspath(str(runtime.get("tmp_dir", "")))
    work_dir = os.path.abspath(str(context.get("work_dir", ".")))
    if not tmp_dir or tmp_dir in {"/", os.path.expanduser("~"), work_dir}:
        logger.warning("Refusing to cleanup unsafe tmp_dir: %s", tmp_dir)
        return
    if not os.path.isdir(tmp_dir):
        return
    shutil.rmtree(tmp_dir)
    logger.info("Temporary workflow directory removed: %s", tmp_dir)
