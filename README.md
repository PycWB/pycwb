# PycWB

[![Documentation](https://readthedocs.org/projects/pycwb/badge/?version=latest)](https://docs.pycwb.org)
[![Build Status](https://git.ligo.org/yumeng.xu/pycwb/badges/main/pipeline.svg)](https://git.ligo.org/yumeng.xu/pycwb/-/pipelines)
[![Coverage](https://git.ligo.org/yumeng.xu/pycwb/badges/main/coverage.svg)](https://git.ligo.org/yumeng.xu/pycwb/-/pipelines)
[![Releases](https://git.ligo.org/yumeng.xu/pycwb/-/badges/release.svg)](https://git.ligo.org/yumeng.xu/pycwb/-/releases)
[![PyPI version](https://badge.fury.io/py/pycWB.svg)](https://badge.fury.io/py/pycWB)
[![License](https://img.shields.io/badge/license-GPLv3-blue)](https://git.ligo.org/yumeng.xu/pycwb/-/blob/main/LICENSE)

PycWB is a modular Python implementation of the coherent WaveBurst
(cWB/cWB-2G) algorithms for gravitational-wave burst searches.
The documentation can be found at [docs.pycwb.org](https://docs.pycwb.org).

## Get started

This checkout contains the new `validate` command. Until a
release containing it is published, install this source checkout:

```bash
conda create -n pycwb -c conda-forge python=3.13 pip nds2-client python-nds2-client lalsuite python-ligo-lw
conda activate pycwb
git clone https://git.ligo.org/yumeng.xu/pycwb.git
cd pycwb
python -m pip install .
pycwb --version
pycwb validate examples/demo/user_parameters.yaml
pycwb run examples/demo/user_parameters.yaml --work-dir my_first_search
pycwb progress --work-dir my_first_search
```

The CLI example in `examples/demo/` generates synthetic data containing a loud
injected burst and writes the normal pipeline outputs. It needs no detector-data account. The first run may
download a roughly 53 MiB cross-talk catalog and compile numerical kernels.
Use a fresh directory for each run. See [Your First Search](https://docs.pycwb.org/en/latest/start_here.html)
for expected outputs and [troubleshooting](docs/source/troubleshooting.rst).

## Choose the matching version

A normal `python -m pip install pycwb` selects a stable release; `--pre` allows
prereleases. Older releases have different dependencies and may require ROOT.
Match `pycwb --version` to the documentation version. Development documentation
can describe features not yet available on PyPI.

The current native path requires Python >=3.11 and does not require ROOT.
Linux x86_64/Python 3.13 is the current CI environment. Other platform coverage
and optional ROOT/PyCBC/XGBoost setup are described in the
[installation guide](docs/source/install.rst).

## Run your own analysis

```bash
pycwb validate user_parameters.yaml
pycwb run user_parameters.yaml
```

```python
from pycwb.workflow.run import search

search("user_parameters.yaml")
```

For production configuration templates, follow
[Setup Config Templates](https://docs.pycwb.org/en/latest/config_repository.html).
Read [Understanding Your Results](docs/source/understanding_results.rst),
[Reproducibility](docs/source/reproducibility.rst), and
[Validation Scope](docs/source/validation_status.rst) before interpreting an analysis.

## Help and contributions

- [Support and bug reports](SUPPORT.md), including an email route without a GitLab account.
- [Contribution instructions](CONTRIBUTING.md) and [changes](CHANGES.md).
- [Citation metadata](CITATION.cff) and [scientific citation guidance](docs/source/credit.rst).
- [Release and compatibility process](docs/source/release_policy.rst).

Legacy notebooks are available in [examples](examples); their installation
cells may target older releases. The CLI [synthetic example](examples/demo/README.md) is the maintained beginner path.
