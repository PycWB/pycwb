"""Small input helpers shared by the executable teaching notebooks.

Search stages remain in the notebooks. These helpers only locate the checkout,
prepare a compact configuration.
"""

import os
from pathlib import Path

import yaml

from pycwb.config import Config


REPOSITORY = Path(__file__).resolve().parents[1]


def prepare_injection_config(name, *, parameters=None, updates=None):
    """Write a fresh educational configuration based on the maintained demo.

    ``PYCWB_EXAMPLES_WORK_DIR`` selects an output parent for notebook execution.
    ``PYCWB_EXAMPLES_XTALK`` optionally selects an existing cross-talk catalog.
    Existing configurations are never overwritten: choose a new output parent
    when repeating a notebook.
    """
    parent = Path(os.environ.get("PYCWB_EXAMPLES_WORK_DIR", "notebook-runs")).resolve()
    work = parent / name
    work.mkdir(parents=True, exist_ok=False)
    params = yaml.safe_load((REPOSITORY / "examples/demo/user_parameters.yaml").read_text())
    params.update(plot_waveform=False, plot_sky_map=False)
    if parameters is not None:
        params["injection"]["parameters"] = parameters
        params["injection"]["generator"] = "pycwb.modules.injection.gwsignal_waveform.get_td_waveform"
    params.update(updates or {})
    xtalk = os.environ.get("PYCWB_EXAMPLES_XTALK")
    if xtalk:
        catalog = Path(xtalk).expanduser().resolve(strict=True)
        params.update(filter_dir=str(catalog.parent), wdmXTalk=catalog.name)
    else:
        cache = Path.home() / ".cache/pycwb/examples"
        cache.mkdir(parents=True, exist_ok=True)
        params["filter_dir"] = str(cache)
    path = work / "user_parameters.yaml"
    path.write_text(yaml.safe_dump(params, sort_keys=False))
    config = Config()
    config.load_from_yaml(path)
    return work, path, config
