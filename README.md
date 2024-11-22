# PycWB

[![Documentations](https://readthedocs.org/projects/pycwb/badge/?version=latest)](https://pycwb.readthedocs.io)
[![Build Status](https://git.ligo.org/yumeng.xu/pycwb/badges/main/pipeline.svg)](https://git.ligo.org/yumeng.xu/pycwb/-/pipelines)
[![Releases](https://git.ligo.org/yumeng.xu/pycwb/-/badges/release.svg)](https://git.ligo.org/yumeng.xu/pycwb/-/releases)
[![PyPI version](https://badge.fury.io/py/pycWB.svg)](https://badge.fury.io/py/pycWB)
[![License](https://img.shields.io/badge/license-GPLv3-blue)](https://git.ligo.org/yumeng.xu/pycwb/-/blob/main/LICENSE)

PycWB is a modularized Python package for gravitational wave burst search based on the core function of cWB.
The documentation can be found at [pycwb.readthedocs.io](https://pycwb.readthedocs.io).

## Installation

### Install PycWB with pip

PycWB is available on [PyPI](https://pypi.org/project/pycWB/). You can install it with pip.
Some dependencies are required to be installed before installing pycWB with pip. 
The easiest way is to install them with conda.

```bash
conda create -n pycwb "python>=3.9,<3.11"
conda activate pycwb
conda install -c conda-forge root=6.26.10 healpix_cxx=3.81 nds2-client python-nds2-client lalsuite setuptools_scm cmake pkg-config
python3 -m pip install pycwb
```

Currently, pycWB is only available for x64 architecture.
For Apple Silicon users, you can install the dependencies with the following commands:

```bash
# make sure rosetta is installed
softwareupdate --install-rosetta --agree-to-license
# Optional: export CONDA_BUILD=1
conda create -n pycwb
conda activate pycwb
conda config --env --set subdir osx-64
conda install -c conda-forge "python>=3.9,<3.11" root=6.26.10 healpix_cxx=3.81 nds2-client python-nds2-client lalsuite setuptools_scm cmake pkg-config ruamel.yaml htcondor
```

### Install pycWB from source

```bash
conda create -n pycwb python
conda activate pycwb
conda install -c conda-forge root=6.26.10 healpix_cxx=3.81 nds2-client python-nds2-client lalsuite setuptools_scm cmake pkg-config
git clone git@git.ligo.org:yumeng.xu/pycwb.git
cd pycwb
make install
```

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

## Interactive tutorial

 - Google Colab tutorial: [pycWB_GW150914.ipynb](https://colab.research.google.com/github/PycWB/pycwb/blob/main/examples/colab/pycWB_GW150914.ipynb)
