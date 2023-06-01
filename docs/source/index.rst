.. pycWB documentation master file, created by
   sphinx-quickstart on Wed Mar  8 13:16:45 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pycWB's documentation!
===================================

pycWB is a modularized Python package for gravitational wave burst search based on the core function of cWB.

...

.. toctree::
   :hidden:
   :maxdepth: 5

   install
   credit
   pycWB <modules>
   genindex

.. toctree::
   :hidden:
   :caption: User Guides
   :maxdepth: 1

   tutorials
   schema

.. toctree::
   :hidden:
   :caption: Developer Guides
   :maxdepth: 1

   mod_cwb


Getting Started
===============

Installation
------------

.. code-block:: bash

   conda create -n pycwb "python>=3.9,<3.11"
   conda activate pycwb
   conda install -c conda-forge root=6.26.10 healpix_cxx=3.81 nds2-client python-nds2-client lalsuite setuptools_scm cmake pkg-config
   python3 -m pip install pycwb


Run your first burst search
---------------------------

Copy the example configuration folder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    cp -r [path_to_source_code]/examples/injection my_search

Start searching!
~~~~~~~~~~~~~~~~

You can directly run the example search script in the example folder

.. code-block:: bash

    cd my_search
    pycwb_search user_parameters_injection.yaml

If you are on a cluster, you can submit the job to the cluster

.. code-block:: bash

    pycwb_search user_parameters_injection.yaml --submit condor

Or you can open the juptyer notebook `pycwb_injection.ipynb` and run the search step by step

Go deeper into pycWB.search
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to know more about the search process, please refer to
:ref:`tutorial_search`

Step by step injection!
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to know more about the injection process step by step, please refer to
:ref:`tutorial_injection` or the juptyer notebook `pycwb_injection.ipynb`

Basic Workflow
==============
.. image:: workflow.png
   :width: 100%
   :align: center

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
