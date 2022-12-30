# pycWB

This is a project to simplify the installation of `cWB` and run `cWB` with python.

## Installation

### Install cWB

Check [installation guide](./docs/0.installation_guide.md) to simply install `cWB` with conda

### Install pycWB from git

```bash
git clone git@git.ligo.org:yumeng.xu/pycwb.git
pip install ./pycWB
```

## Initialise pycWB

The [initialisation guide](./docs/1.initialisation_guide.md) can help you understand the detail of the environment setup and library loading with python. This processing is coded in the class `pycWB`.  If you are not interested in the detail, you can directly initialize the `cWB` with

```python
import sys
sys.path.append('..') # add pycWB parent path
from pycWB import pycWB

cwb = pycWB('../pycWB/config/config.ini') # config file path
ROOT = cwb.ROOT
gROOT = cwb.gROOT
```

Required directories will be automatically created unless you initialise 
with `pycWB('config.ini', create_dirs=False)`

## Run analysis

The project can be setup with original `.c` file as well as `.yaml` config file, see [example](./examples/MultiStages2G/user_parameters.yaml).

> The compatibility of `ROOT TBroswer` with macos still need to be fixed
> This project is tested with macos, linux should be fine in princple.


### with `.c` config file
The [Example : interactive multistages 2G analysis](./docs/2.test_interactive_multistages_2G_analysis.md) contains a full example to run the `pycWB`


### with `.yaml` config file (recommended)

If you don't want to setup a cwb run with c file `user_parameters.c`, 
you can follow [YAML Example : interactive multistages 2G analysis](./docs/3.run_pycwb_with_yaml_config.md) to setup
an analysis with `yaml` config file.

```python
cwb.cwb_inet2G(job_id,'user_parameters.yaml', 'FULL')
```

> The reason to choose `yaml` is that it can support more complicated types compare to `ini` and 
> much close to python compare to `json`
> 
> "YAML" will be checked by `jsonschema` with file `config/user_parameters_schema.py`
> and converted to C code to run with `pyROOT`
