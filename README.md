# PycWB

[![Build Status](https://git.ligo.org/yumeng.xu/pycwb/badges/main/pipeline.svg)](https://git.ligo.org/yumeng.xu/pycwb/-/pipelines)
[![Releases](https://git.ligo.org/yumeng.xu/pycwb/-/badges/release.svg)](https://git.ligo.org/yumeng.xu/pycwb/-/releases)
[![PyPI version](https://badge.fury.io/py/pycWB.svg)](https://badge.fury.io/py/pycWB)
[![License](https://img.shields.io/badge/license-GPLv3-blue)](https://git.ligo.org/yumeng.xu/pycwb/-/blob/main/LICENSE)

PycWB is a modularized Python package for gravitational wave burst search based on the core function of cWB.

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
from pycwb.search import search

search('./user_parameters.yaml')
```

or run with command line

```bash
pycwb_search ./user_parameters.yaml
```

## Documentation

Documentation can be found in [https://yumeng.xu.docs.ligo.org/pycwb](https://yumeng.xu.docs.ligo.org/pycwb)