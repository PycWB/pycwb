.. _dev_setup:

Development Setup
=================

How to set up a pycWB development environment from source.

.. contents:: Table of Contents
   :depth: 2
   :local:


Prerequisites
-------------

- Python 3.10+
- conda or mamba
- Git
- C++ compiler (for ``cwb-core/``; optional for pure-Python development)


Quick Start
-----------

.. code-block:: bash

   # Clone the repo
   git clone git@git.ligo.org:yumeng.xu/pycwb.git
   cd pycwb

   # Create conda environment
   conda create -n pycwb python=3.13
   conda activate pycwb

   # Install core scientific dependencies
   conda install -c conda-forge healpix_cxx=3 lalsuite setuptools_scm cmake pkg-config

   # Optional: ROOT (for C++ wavelet extension)
   conda install -c conda-forge root=6

   # Editable install
   pip install -e .


Building the C++ Core (optional)
--------------------------------

The C++ wavelet core is compiled via CMake:

.. code-block:: bash

   make build_cwb
   # or: python setup.py build_cwb

This produces the ROOT/PyROOT bindings. ROOT is optional—the pure-Python
analysis path works without it. All new development should target the
pure-Python path.


Verifying the Installation
--------------------------

.. code-block:: bash

   # Basic import check
   python -c "import pycwb; print(pycwb.__version__)"

   # CLI check
   pycwb --help

   # JAX check (recommended for performance)
   python -c "import jax; print('JAX:', jax.__version__)"

   # ROOT check (optional)
   python -c "from pycwb.utils.check_ROOT import has_ROOT; print('ROOT:', has_ROOT())"


IDE Setup
---------

**VS Code** (recommended):
- Python extension with conda environment selected
- Pylance for type checking
- Copilot instructions at ``.github/copilot-instructions.md`` provide
  project-specific context

**Editor settings** (``.vscode/settings.json``):

.. code-block:: json

   {
     "python.defaultInterpreterPath": "~/miniconda3/envs/pycwb/bin/python",
     "python.testing.pytestEnabled": true,
     "python.testing.unittestEnabled": false,
     "python.analysis.typeCheckingMode": "basic"
   }


conda Environment Reference
---------------------------

Minimal environment (pure-Python only):

.. code-block:: yaml

   # environment.yml
   name: pycwb
   channels:
     - conda-forge
   dependencies:
     - python=3.13
     - numpy
     - scipy
     - jax
     - numba
     - pyarrow
     - pandas
     - pyyaml
     - healpix_cxx=3
     - lalsuite
     - pip
     - pip:
       - pycwb  # or: -e . for editable

Full environment (with ROOT and NDS2):

.. code-block:: yaml

   # environment-full.yml
   name: pycwb
   channels:
     - conda-forge
   dependencies:
     - python=3.13
     - root=6
     - nds2-client
     - python-nds2-client
     - python-ligo-lw
     - gwpy
     - ...  # plus all pure-Python deps


macOS / Apple Silicon Notes
---------------------------

- ROOT may not build natively on arm64 Macs. Use the pure-Python path.
- JAX runs on CPU via accelerated Metal; install ``jax-metal`` for GPU support.
- C++ core requires x86_64 or Rosetta.
