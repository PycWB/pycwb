.. _installing_pycwb:

####################
Installation Guide
####################


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Installation with Conda/Pip
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

The project is available on PyPI

.. code-block:: bash

   conda create -n pycwb "python>=3.9,<3.11"
   conda activate pycwb
   conda install -c conda-forge root=6.26.10 healpix_cxx=3.81 nds2-client python-nds2-client lalsuite setuptools_scm cmake pkg-config
   python3 -m pip install pycwb

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Installing from Source with Conda and Pip
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

We recommend installing pycwb from source with conda environment,
because all the dependencies can be installed with conda and some of them are not available in pip.
pycWB can be installed with pip from source.

The command `make install` will help you pack the source code and install
it with pip.

.. code-block:: bash

    conda create -n pycwb python
    conda activate pycwb
    conda install -c conda-forge root=6.26.10 healpix_cxx=3.81 nds2-client python-nds2-client lalsuite setuptools_scm
    git clone git@git.ligo.org:yumeng.xu/pycwb.git
    cd pycwb
    make install


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Other scenarios
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

=====================================
Building the Documentation
=====================================

To build the documentation, you will need to install the following packages and
then run the ``make doc`` command.

.. code-block:: bash

    pip install "sphinx<7.0.0" sphinx_rtd_theme
    make doc

The documentation will be built in the ``docs/build/html`` directory.


.. caution::

    The rst files in the ``docs/source`` with name ``pycwb.*`` and ``modules.rst`` will be deleted when you run
    ``make doc`` for preventing caches. So please do not edit them manually or name any of your rst files with the same name.