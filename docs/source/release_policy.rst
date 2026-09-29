.. _release_policy:

Releases and Compatibility
==========================

Release versions
----------------

.. raw:: html

   <span id="match-software-examples-and-documentation"></span>

The ``latest`` documentation follows development. For a released version,
select its documentation in the version menu. Available releases are listed
in the `PyPI release history <https://pypi.org/project/PycWB/#history>`_.

.. _release-checklist-for-maintainers:

Maintainers: follow :doc:`dev_release` for the release checklist, publication
steps, and hosted documentation settings.

.. _installation_release_channels:

Choose a release channel
------------------------

**Stable release:** install the latest stable package from PyPI.

.. code-block:: bash

   python -m pip install pycwb
   pycwb --version

**Prerelease:** include ``--pre`` to install alpha and other prereleases.

.. code-block:: bash

   python -m pip install --pre pycwb
   pycwb --version

To install a specific version, use ``python -m pip install pycwb==VERSION``.
See :doc:`install` for environment setup and optional dependencies.

Compatibility changes
---------------------

Read ``CHANGES.md`` for changes to commands, configuration and output formats.
Catalog and job-manifest compatibility is described in :doc:`catalog_format`.

Use the documented interfaces in :doc:`reference` when writing integrations.
