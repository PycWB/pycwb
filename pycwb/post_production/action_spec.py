"""Action specification decorator for workflow diagram generation.

Declares which function parameters are **inputs** (data consumed from
earlier steps) and **outputs** (data produced for later steps).  The
decorator stores this metadata as ``func.__action_spec__`` so that the
DAG builder in `pycwb.post_production.diagram` can discover dependencies
without executing the workflow.

Usage
-----
>>> from pycwb.post_production.action_spec import action_spec
>>>
>>> @action_spec(
...     outputs=['output_file'],
...     inputs=['progress_file'],
...     description='Select a random subset of jobs by live time',
... )
... def select_jobs_by_livetime(
...     work_dir, progress_file, output_file, fraction=0.10,
...     exclude_zero_lag=True, seed=150914, **kwargs,
... ) -> dict:
...     ...

Notes
-----
- Only **data-flow** parameters should be listed.  Configuration
  parameters such as ``work_dir``, ``seed``, ``fraction``, ``search``,
  ``nifo``, and ``**kwargs`` are intentionally omitted.
- The ``description`` field is used as a human-readable label in the
  generated diagram.  If omitted, the function's docstring first line is
  used as a fallback.
"""

from __future__ import annotations

from typing import Callable, Optional


def action_spec(
    outputs: Optional[list[str]] = None,
    inputs: Optional[list[str]] = None,
    description: Optional[str] = None,
    display_name: Optional[str] = None,
    help: Optional[str] = None,
    args_schema: Optional[dict] = None,
    composite: bool = False,
) -> Callable:
    """Decorator that annotates a workflow action with I/O metadata.

    Parameters
    ----------
    outputs : list of str, optional
        Names of parameters that produce files or values consumed by
        downstream steps (e.g. ``['output_file', 'model_file']``).
    inputs : list of str, optional
        Names of parameters that consume files or values from upstream
        steps (e.g. ``['progress_file', 'catalog_file']``).
    description : str, optional
        Human-readable label shown in the diagram node.  Falls back
        to the first line of the function's docstring.
    display_name : str, optional
        Short user-facing action name for diagrams.
    help : str, optional
        Longer help text for interactive diagrams and developer tooling.
    args_schema : dict, optional
        Lightweight argument metadata for UI/help rendering.  This is not a
        validation schema yet; it is intentionally permissive.
    composite : bool
        True if this action intentionally wraps smaller reusable actions.

    Returns
    -------
    Callable
        The decorated function, with ``__action_spec__`` attached.
    """
    def decorator(func: Callable) -> Callable:
        func.__action_spec__ = {
            'outputs': list(outputs) if outputs else [],
            'inputs': list(inputs) if inputs else [],
            'description': description or (
                func.__doc__.strip().split('\n')[0].strip()
                if func.__doc__ else func.__name__
            ),
            'display_name': display_name or func.__name__,
            'help': help or '',
            'args_schema': dict(args_schema) if args_schema else {},
            'composite': bool(composite),
        }
        return func
    return decorator


def get_action_spec(func: Callable) -> dict:
    """Return the ``__action_spec__`` dict for *func*, or an empty spec.

    Parameters
    ----------
    func : Callable
        A workflow action function (may or may not be decorated).

    Returns
    -------
    dict
        With keys ``'outputs'`` (list), ``'inputs'`` (list),
        ``'description'`` (str).
    """
    return getattr(func, '__action_spec__', {
        'outputs': [],
        'inputs': [],
        'description': (
            func.__doc__.strip().split('\n')[0].strip()
            if func.__doc__ else func.__name__
        ),
        'display_name': func.__name__,
        'help': '',
        'args_schema': {},
        'composite': False,
    })
