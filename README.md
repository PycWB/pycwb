# PycWB

[![Documentation](https://readthedocs.org/projects/pycwb/badge/?version=latest)](https://pycwb.readthedocs.io)
[![Build Status](https://git.ligo.org/yumeng.xu/pycwb/badges/main/pipeline.svg)](https://git.ligo.org/yumeng.xu/pycwb/-/pipelines)
[![Releases](https://git.ligo.org/yumeng.xu/pycwb/-/badges/release.svg)](https://git.ligo.org/yumeng.xu/pycwb/-/releases)
[![PyPI version](https://badge.fury.io/py/pycWB.svg)](https://badge.fury.io/py/pycWB)
[![License](https://img.shields.io/badge/license-GPLv3-blue)](https://git.ligo.org/yumeng.xu/pycwb/-/blob/main/LICENSE)

PycWB is a modularized Python package for gravitational wave burst search based on the core functions of cWB.
The documentation can be found at [pycwb.readthedocs.io](https://pycwb.readthedocs.io).

## Installation

### Install PycWB with pip

PycWB is available on [PyPI](https://pypi.org/project/pycWB/). You can install it with pip.
Some dependencies are required before installing `pycwb` with pip.
The easiest way is to install them with conda.

> Python requirement: `>=3.10`

```bash
conda create -n pycwb python=3.13
conda activate pycwb
conda install -c conda-forge root=6 healpix_cxx=3 nds2-client python-nds2-client lalsuite python-ligo-lw setuptools_scm cmake pkg-config
python3 -m pip install pycwb
```

Currently, the ROOT-enabled `pycwb` build is only available for `x86_64` architecture. You can install the pure Python version of `pycwb` without installing ROOT:

```bash
conda create -n pycwb python=3.13
conda activate pycwb
conda install -c conda-forge nds2-client python-nds2-client lalsuite python-ligo-lw setuptools_scm cmake pkg-config
python3 -m pip install pycwb
```

For Apple Silicon users, if you need the ROOT version, install dependencies with the following commands:

```bash
# make sure rosetta is installed
softwareupdate --install-rosetta --agree-to-license
# Optional: export CONDA_BUILD=1
conda create -n pycwb_x64
conda activate pycwb_x64
conda config --env --set subdir osx-64
conda install python==3.11 root=6.28 healpix_cxx=3 nds2-client python-nds2-client lalsuite python-ligo-lw setuptools_scm cmake pkg-config ruamel.yaml htcondor
python3 -m pip install pycwb
```

### Install PycWB from source

```bash
conda create -n pycwb python
conda activate pycwb
conda install -c conda-forge root=6 healpix_cxx=3 nds2-client python-nds2-client lalsuite python-ligo-lw setuptools_scm cmake pkg-config
git clone git@git.ligo.org:yumeng.xu/pycwb.git
cd pycwb
python -m pip install .
```

> Again, for Apple Silicon users, if you don't need the ROOT version, remove `root=6` and `healpix_cxx=3` from the dependencies in the command above. The installation process will automatically install the pure Python version of `pycwb`.

## Usage

Example project can be found in [examples](https://git.ligo.org/yumeng.xu/pycwb/-/tree/main/examples)

```python
from pycwb.workflow.run import search

search('./user_parameters.yaml')
```

or run with command line

```bash
pycwb run ./user_parameters.yaml
```

### Verify installation

```bash
pycwb --version
pycwb --help
```

### Quick start for config setup

For one-command project setup and optional job submission, see [QUICKSTART_CONFIG_SETUP.md](./QUICKSTART_CONFIG_SETUP.md).

## Interactive tutorial

 - Google Colab tutorial: [pycWB_GW150914.ipynb](https://colab.research.google.com/github/PycWB/pycwb/blob/main/examples/colab/pycWB_GW150914.ipynb)
