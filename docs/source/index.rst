.. PyBurst documentation master file, created by
   sphinx-quickstart on Wed Mar  8 13:16:45 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PyBurst's documentation!
===================================

PyBurst is a Python package for gravitational wave burst search based on the core function of cWB.

...

.. toctree::
   :hidden:
   :maxdepth: 2

   install
   credit
   Modules <modules>
   genindex

.. toctree::
   :hidden:
   :caption: User Guides
   :maxdepth: 1

   tutorials


Getting Started
===============

Installation
------------

.. code-block:: bash

    conda create -n pyburst python
    conda install -c conda-forge root=6.26.10 healpix_cxx=3.81 nds2-client python-nds2-client lalsuite setuptools_scm
    git clone git@git.ligo.org:yumeng.xu/pycwb.git
    cd pycwb
    make install


Run your first burst search
---------------------------

Generate a user parameter file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    pyburst_gen_config -o user_parameters.yaml

Start searching!
~~~~~~~~~~~~~~~~

.. code-block:: python

   from pyburst.search import search

   search('./user_parameters.yaml')


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
