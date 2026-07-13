# PycWB

[![Documentation](https://readthedocs.org/projects/pycwb/badge/?version=latest)](https://pycwb.readthedocs.io)
[![Build Status](https://git.ligo.org/yumeng.xu/pycwb/badges/main/pipeline.svg)](https://git.ligo.org/yumeng.xu/pycwb/-/pipelines)
[![Coverage](https://git.ligo.org/yumeng.xu/pycwb/badges/main/coverage.svg)](https://git.ligo.org/yumeng.xu/pycwb/-/pipelines)
[![Releases](https://git.ligo.org/yumeng.xu/pycwb/-/badges/release.svg)](https://git.ligo.org/yumeng.xu/pycwb/-/releases)
[![PyPI version](https://badge.fury.io/py/pycWB.svg)](https://badge.fury.io/py/pycWB)
[![License](https://img.shields.io/badge/license-GPLv3-blue)](https://git.ligo.org/yumeng.xu/pycwb/-/blob/main/LICENSE)

PycWB is a modular Python implementation of the coherent WaveBurst
(cWB/cWB-2G) algorithms for gravitational-wave burst searches.
The documentation can be found at [pycwb.readthedocs.io](https://pycwb.readthedocs.io).

## Installation

### Install PycWB with pip

PycWB is available on [PyPI](https://pypi.org/project/pycWB/). You can install it with pip.
Some dependencies are required before installing `pycwb` with pip.
The easiest way is to install them with conda. For regular use, install the pure Python path first.
ROOT is optional and is only needed when testing ROOT-backed components or comparing against ROOT/C++ cWB behavior.

> Python requirement: `>=3.10`

```bash
conda create -n pycwb python=3.13
conda activate pycwb
conda install -c conda-forge nds2-client python-nds2-client lalsuite python-ligo-lw setuptools_scm cmake pkg-config
python3 -m pip install pycwb
```

### Optional ROOT-backed testing environment

Use this only if you explicitly want to test the ROOT-backed extension or ROOT/C++ interoperability paths. Currently, the ROOT-enabled `pycwb` build is available on `x86_64` platforms.

```bash
conda create -n pycwb python=3.13
conda activate pycwb
conda install -c conda-forge root=6 healpix_cxx=3 nds2-client python-nds2-client lalsuite python-ligo-lw setuptools_scm cmake pkg-config
python3 -m pip install pycwb
```

For Apple Silicon users, if you need this optional ROOT-enabled environment, install dependencies with the following commands:

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

The default source install does not require ROOT.

```bash
conda create -n pycwb python
conda activate pycwb
conda install -c conda-forge nds2-client python-nds2-client lalsuite python-ligo-lw setuptools_scm cmake pkg-config
git clone git@git.ligo.org:yumeng.xu/pycwb.git
cd pycwb
python -m pip install .
```

To test the optional ROOT-backed extension from source, install `root=6` and `healpix_cxx=3` in the same conda environment before running `python -m pip install .`. If ROOT is not available, setup skips the C++ wavelet extension and installs the native Python path.

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

 - Google Colab tutorial: [GW150914.ipynb](https://colab.research.google.com/github/PycWB/pycwb/blob/main/examples/colab/GW150914.ipynb)
