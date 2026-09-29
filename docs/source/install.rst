.. _installing_pycwb:
.. _installation-and-versions:

Installation
============

Python 3.13 is recommended; Python 3.11 or newer is required.
ROOT is not required for PycWB >= 1.

Install from PyPI
-----------------

Create a separate environment, then install PycWB:

.. code-block:: bash

   conda create -n pycwb -c conda-forge python=3.13 pip \
       nds2-client python-nds2-client lalsuite python-ligo-lw
   conda activate pycwb
   python -m pip install --pre pycwb
   pycwb --version

The ``--pre`` flag selects the current 1.x prereleases from PyPI.
NDS2 client libraries are supplied through conda-forge.

.. _optional-components:

Postproduction and waveform extras
----------------------------------

XGBoost is required for the standard postproduction workflow's model training
and scoring:

.. code-block:: bash

   python -m pip install --pre 'pycwb[xgboost]'

For PyCBC waveform integrations, also install:

.. code-block:: bash

   python -m pip install --pre 'pycwb[pycbc]'

.. _source-installation-for-this-development-guide:

Install from source
-------------------

For unreleased changes, activate the environment created above and install
the source checkout:

.. code-block:: bash

   git clone https://git.ligo.org/yumeng.xu/pycwb.git
   cd pycwb
   python -m pip install .

From this checkout, use ``'.[xgboost]'`` or ``'.[pycbc]'`` to install extras.
For an editable development install, see :doc:`dev_setup`.

Next steps
----------

Continue with :doc:`start_here` or :doc:`postproduction`.

.. _choose-a-release-channel:

* :ref:`installation_release_channels`: stable and prerelease installation options.

.. _platform-coverage:


.. _build-the-documentation:

For development and documentation builds, see :doc:`dev_setup`.
