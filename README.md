# pycWB

This is a project to simplify the installation of `cWB` and run `cWB` with python.

Check [installation guide](./0.installation_guide.md) to simply install `cWB` with conda

The [initialisation guide](1.initialisation_guide.md) can help you understand the detail of the environment setup and library loading with python. This processing is coded in the class `pycWB`.  If you are not interested in the detail, you can directly initialize the `cWB` with

```python
import sys
sys.path.append('..') # add pycWB parent path
from pycWB import pycWB

cwb = pycWB('../pycWB/config/config.ini')
ROOT = cwb.ROOT
gROOT = cwb.gROOT
```

The [Example : interactive multistages 2G analysis](./2.test_interactive_multistages_2G_analysis.md) contains a full example to run the `pycWB`

> The compatibility of `ROOT TBroswer` with macos still need to be fixed
> This project is tested with macos, linux should be fine in princple.

If you don't want to setup a cwb run with c file `user_parameters.c`, 
you can follow [YAML Example : interactive multistages 2G analysis](./3.run_pycwb_with_yaml_config.md) to setup
an analysis with `yaml` config file.

> The reason to choose `yaml` is that it can support more complicated types compare to `ini` and 
> much close to python compare to `json`