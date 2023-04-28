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

    conda create -n pycwb python
    conda activate pycwb
    conda install -c conda-forge root=6.26.10 healpix_cxx=3.81 nds2-client python-nds2-client lalsuite setuptools_scm
    git clone git@git.ligo.org:yumeng.xu/pycwb.git
    cd pycwb
    make install


Run your first burst search
---------------------------

Copy the example configuration file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    mkdir my_search
    cp [path_to_source_code]/examples/injection/user_parameters_injection.yaml user_parameters.yaml

Start searching!
~~~~~~~~~~~~~~~~

.. code-block:: python

   from pycwb.search import search

   search('./user_parameters.yaml')

Go deeper into PyBurst.search
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to know more about the search process, please refer to
:ref:`tutorial_search`

Step by step injection!
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to know more about the injection process step by step, please refer to
:ref:`tutorial_injection`

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
