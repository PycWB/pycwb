# pycWB

This is a project to simplify the installation of `cWB` and run `cWB` with python.

## Installation

### Install cWB

Check [installation guide](./docs/0.installation_guide.md) to simply install `cWB` with conda

### Install pycWB from git

```bash
git clone git@git.ligo.org:yumeng.xu/pycwb.git
cd pycWB
python setup.py sdist && pip install dist/*.tar.gz
```

## Generate config files

Run the following script to generate `config.ini` and the sample `user_parameters.yaml`
in your working directory

```bash
pycwb_gen_config --cwb_install <path to cwb install> --cwb_source <path to cwb source> --work_dir <path to work dir>
```

edit these two files to fit your environment and your job

## Initialize pycWB

The [initialisation guide](./docs/1.initialisation_guide.md) can help you understand the detail of the environment setup
and library loading with python. This processing is coded in the class `pycWB`. If you are not interested in the detail,
you can directly initialize the `cWB` with

```python
from pycWB import pycWB

cwb = pycWB('./config.ini')  # config file path
ROOT = cwb.ROOT
gROOT = cwb.gROOT
```

Required directories will be automatically created unless you initialize
with `pycWB('./config.ini', create_dirs=False)`

## Run analysis

The project can be setup with original `.c` file as well as `.yaml` config file,
see [example](./examples/MultiStages2G/user_parameters.yaml).

> The compatibility of `ROOT TBroswer` with macos still need to be fixed
> This project is tested with macos, linux should be fine in princple.

### with `.c` config file

The [Example : interactive multistages 2G analysis](./docs/2.test_interactive_multistages_2G_analysis.md) contains a
full example to run the `pycWB`

### with `.yaml` config file (recommended)

If you don't want to setup a cwb run with c file `user_parameters.c`,
you can setup an analysis with `yaml` config file.

#### A quick example

see detail in [YAML Example : interactive multistages 2G analysis](./docs/3.run_pycwb_with_yaml_config.md) to

```python
from pycWB import pycWB, sim

cwb = pycWB('./config.ini')  # config file path
ROOT = cwb.ROOT
gROOT = cwb.gROOT

# create frame file

sim.create_frame_noise(gROOT, ROOT)
sim.setup_sim_data(['H1','L1','V1'])

# run full `cwb_inet2G` analysis

job_id = 1
job_stage = 'FULL'
job_file = './user_parameters.yaml'
inet_option = '--tool emax --level 8  --draw true'
cwb.cwb_inet2G(job_id, job_file, job_stage, inet_option=inet_option)
```

> The reason to choose `yaml` is that it can support more complicated types compare to `ini` and
> much close to python compare to `json`
>
> "YAML" will be checked by `jsonschema` with file `config/user_parameters_schema.py`
> and converted to C code to run with `pyROOT`
