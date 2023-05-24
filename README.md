# pycWB

[![Build Status](https://git.ligo.org/yumeng.xu/pycwb/badges/main/pipeline.svg)](https://git.ligo.org/yumeng.xu/pycwb/-/pipelines)
[![Releases](https://git.ligo.org/yumeng.xu/pycwb/-/badges/release.svg)](https://git.ligo.org/yumeng.xu/pycwb/-/releases)
[![License](https://img.shields.io/badge/license-GPLv3-blue)](https://git.ligo.org/yumeng.xu/pycwb/-/blob/main/LICENSE)

pycWB is a modularized Python package for gravitational wave burst search based on the core function of cWB.

## Installation

### Install pycWB with pip

pycWB is available on [TestPyPI](https://test.pypi.org/project/pycwb/). You can install it with pip.
Some dependencies are required to be installed before installing pycWB with pip. 
The easiest way is to install them with conda.

```bash
conda create -n pycwb "python>=3.9,<3.11"
conda activate pycwb
conda install -c conda-forge root=6.26.10 healpix_cxx=3.81 nds2-client python-nds2-client lalsuite setuptools_scm cmake pkg-config
python3 -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ --no-deps pycwb
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

Example project can be found in [examples](./examples)

```python
from pycwb.search import search

search('./user_parameters.yaml')
```

[//]: # (# pycWB)

[//]: # ()
[//]: # (This is python wrapper of `cWB`)

[//]: # ()
[//]: # (## What does this package do)

[//]: # ()
[//]: # (- [x] Generate `ini` and `yaml` configuration file with python script)

[//]: # (- [x] Initialize `ROOT` and `cwb` with `ini` file &#40;replacing `root_logon.c` and bash files&#41;)

[//]: # (- [x] Run `inet2G` job with `yaml` file &#40;replacing `user_parameters.c`&#41;)

[//]: # ()
[//]: # (## Install cWB)

[//]: # ()
[//]: # (Check [installation guide]&#40;./docs/0.installation_guide.md&#41; to simply install `cWB` with conda)

[//]: # ()
[//]: # (## Generate config files)

[//]: # ()
[//]: # (Run the following script to generate `config.ini` and the sample `user_parameters.yaml`)

[//]: # (in your working directory)

[//]: # ()
[//]: # (```bash)

[//]: # (pyburst_gen_config --cwb_install <path to cwb install> --cwb_source <path to cwb source> --work_dir <path to work dir>)

[//]: # (```)

[//]: # ()
[//]: # (edit these two files to fit your environment and your job)

[//]: # ()
[//]: # (## Initialize pycWB)

[//]: # ()
[//]: # (The [initialisation guide]&#40;./docs/1.initialisation_guide.md&#41; can help you understand the detail of the environment setup)

[//]: # (and library loading with python. This processing is coded in the class `pycWB`. If you are not interested in the detail,)

[//]: # (you can directly initialize the `cWB` with)

[//]: # ()
[//]: # (```python)

[//]: # (from pycwb import pycWB)

[//]: # ()
[//]: # (cwb = pycWB&#40;'./config.ini'&#41;  # config file path)

[//]: # (ROOT = cwb.ROOT)

[//]: # (gROOT = cwb.gROOT)

[//]: # (```)

[//]: # ()
[//]: # (Required directories will be automatically created unless you initialize)

[//]: # (with `pycWB&#40;'./config.ini', create_dirs=False&#41;`)

[//]: # ()
[//]: # (## Run analysis)

[//]: # ()
[//]: # (The project can be setup with original `.c` file as well as `.yaml` config file,)

[//]: # (see [example]&#40;./examples/MultiStages2G/user_parameters.yaml&#41;.)

[//]: # ()
[//]: # (> The compatibility of `ROOT TBroswer` with macos still need to be fixed)

[//]: # (> This project is tested with macos, linux should be fine in princple.)

[//]: # ()
[//]: # (### with `.c` config file)

[//]: # ()
[//]: # (The [Example : interactive multistages 2G analysis]&#40;./docs/2.test_interactive_multistages_2G_analysis.md&#41; contains a)

[//]: # (full example to run the `pycWB`)

[//]: # ()
[//]: # (### with `.yaml` config file &#40;recommended&#41;)

[//]: # ()
[//]: # (If you don't want to setup a cwb run with c file `user_parameters.c`,)

[//]: # (you can setup an analysis with `yaml` config file.)

[//]: # ()
[//]: # (#### A quick example)

[//]: # ()
[//]: # (```python)

[//]: # (from pycwb import pycWB, tools)

[//]: # ()
[//]: # (cwb = pycWB&#40;'./config.ini'&#41;  # config file path)

[//]: # (ROOT = cwb.ROOT)

[//]: # (gROOT = cwb.gROOT)

[//]: # ()
[//]: # (# create frame file)

[//]: # ()
[//]: # (tools.create_frame_noise&#40;gROOT, ROOT&#41;)

[//]: # (tools.setup_sim_data&#40;['H1','L1','V1']&#41;)

[//]: # ()
[//]: # (# run full `cwb_inet2G` analysis)

[//]: # ()
[//]: # (job_id = 1)

[//]: # (job_stage = 'FULL')

[//]: # (job_file = './user_parameters.yaml')

[//]: # (inet_option = '--tool emax --level 8  --draw true')

[//]: # (cwb.cwb_inet2G&#40;job_id, job_file, job_stage, inet_option=inet_option&#41;)

[//]: # (```)

[//]: # ()
[//]: # (> The reason to choose `yaml` is that it can support more complicated types compare to `ini` and)

[//]: # (> much close to python compare to `json`)

[//]: # (>)

[//]: # (> "YAML" will be checked by `jsonschema` with file `config/user_parameters_schema.py`)

[//]: # (> and converted to C code to run with `pyROOT`)
