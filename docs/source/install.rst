.. _installing_pycwb:

####################
Installation Guide
####################


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Installation with Conda/Pip
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

The project is available on PyPI. PycWB requires Python 3.10 or newer.
Some dependencies are easiest to install from conda-forge before installing
``pycwb`` with pip. For regular use, install the pure Python path first.
ROOT is optional and is only needed when testing ROOT-backed components or
comparing against ROOT/C++ cWB behavior.

.. code-block:: bash

   conda create -n pycwb python=3.13
   conda activate pycwb
   conda install -c conda-forge nds2-client python-nds2-client lalsuite python-ligo-lw setuptools_scm cmake pkg-config
   python3 -m pip install pycwb

=============================================
Optional ROOT-backed testing environment
=============================================

Use this only if you explicitly want to test the ROOT-backed extension or
ROOT/C++ interoperability paths. Currently, the ROOT-enabled build is available
on ``x86_64`` platforms.

.. code-block:: bash

   conda create -n pycwb python=3.13
   conda activate pycwb
   conda install -c conda-forge root=6 healpix_cxx=3 nds2-client python-nds2-client lalsuite python-ligo-lw setuptools_scm cmake pkg-config
   python3 -m pip install pycwb

Apple Silicon users who need this optional ROOT-enabled environment can create
an ``osx-64`` conda environment under Rosetta:

.. code-block:: bash

   softwareupdate --install-rosetta --agree-to-license
   conda create -n pycwb_x64
   conda activate pycwb_x64
   conda config --env --set subdir osx-64
   conda install python==3.11 root=6.28 healpix_cxx=3 nds2-client python-nds2-client lalsuite python-ligo-lw setuptools_scm cmake pkg-config ruamel.yaml htcondor
   python3 -m pip install pycwb

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Installing from Source with Conda and Pip
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

We recommend installing PycWB from source inside a conda environment because
some dependencies are easier to resolve from conda-forge. Install the package
with pip from the repository root. This default source install does not require
ROOT.

.. code-block:: bash

    conda create -n pycwb python
    conda activate pycwb
    conda install -c conda-forge nds2-client python-nds2-client lalsuite python-ligo-lw setuptools_scm cmake pkg-config
    git clone git@git.ligo.org:yumeng.xu/pycwb.git
    cd pycwb
    python -m pip install .

If you want to test the optional ROOT-backed extension from source, install
``root=6`` and ``healpix_cxx=3`` in the same conda environment before running
``python -m pip install .``. If ROOT is not available, setup skips the C++
wavelet extension and installs the native Python path.

=====================================
Verify installation
=====================================

.. code-block:: bash

    pycwb --version
    pycwb --help

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Other scenarios
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

=====================================
Building the Documentation
=====================================

To build the documentation, you will need to install the following packages and
then run the ``make doc`` command.

.. code-block:: bash

    pip install sphinx sphinxawesome-theme
    make doc

The documentation will be built in the ``docs/build/html`` directory.


.. caution::

    The rst files in the ``docs/source`` with name ``pycwb.*`` and ``modules.rst`` will be deleted when you run
    ``make doc`` for preventing caches. So please do not edit them manually or name any of your rst files with the same name.
