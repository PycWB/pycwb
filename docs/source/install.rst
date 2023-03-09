.. _installing_pyburst:

####################
Installation Guide
####################


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Installation with Conda/Pip
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

The installation with conda/pip is not done yet.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Installing from Source with Conda
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


.. code-block:: bash

    conda create -n pyburst python
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

    pip install sphinx sphinx_rtd_theme
    make doc

The documentation will be built in the ``docs/_build/html`` directory.


.. caution::

    The rst files in the ``docs/source`` with name ``pyburst.*`` and ``modules.rst`` will be deleted when you run
    ``make doc`` for preventing caches. So please do not edit them manually or name any of your rst files with the same name.