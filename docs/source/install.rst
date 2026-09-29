.. _installing_pycwb:
.. _installation-and-versions:

Installation
============

Python 3.13 is recommended; Python 3.11 or newer is required.
ROOT is not required for PycWB >= 1. For ROOT-based analyses, use the original
cWB software; see :doc:`cwb_heritage`.

.. _source-installation-for-this-development-guide:

Install from source
-------------------

Create a separate environment, then install PycWB:

.. code-block:: bash

   conda create -n pycwb -c conda-forge python=3.13 pip \
       nds2-client python-nds2-client lalsuite python-ligo-lw
   conda activate pycwb
   git clone https://git.ligo.org/yumeng.xu/pycwb.git
   cd pycwb
   python -m pip install .
   pycwb --version

To reproduce an existing analysis, check out its recorded tag or commit before
installing. For an editable development install, see :doc:`dev_setup`.
NDS2 client libraries are supplied through conda-forge.

.. _optional-components:

Postproduction and waveform extras
----------------------------------

XGBoost is required for the standard postproduction workflow's model training
and scoring. Install it from the same source checkout:

.. code-block:: bash

   python -m pip install '.[xgboost]'

For PyCBC waveform integrations, also install:

.. code-block:: bash

   python -m pip install '.[pycbc]'

For a released-package installation, the corresponding extras are
``pycwb[xgboost]`` and ``pycwb[pycbc]``.

Next steps
----------

Continue with :doc:`start_here` or :doc:`postproduction`.

.. _choose-a-release-channel:

* :ref:`installation_release_channels`: stable and prerelease installation
  commands. Match the documentation version to ``pycwb --version``.

.. _platform-coverage:

* :ref:`platform_coverage`: tested platforms and current limitations.

.. _build-the-documentation:

* :ref:`building_documentation`: build this documentation locally.
