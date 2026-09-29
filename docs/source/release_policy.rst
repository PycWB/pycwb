.. _release_policy:

Releases and Compatibility
==========================

Match software, examples and documentation
------------------------------------------

``latest`` documentation is built from the development checkout. Stable and
prerelease documentation should be built from the corresponding Git tags. The
page title contains the checkout's package version. Until a tagged build is
published, use the source checkout for newly documented features.

``pip install pycwb`` normally selects a stable release. ``--pre`` allows
prereleases; an explicit ``pycwb==VERSION`` selects a recorded version. Consult
`PyPI <https://pypi.org/project/PycWB/#history>`_ for available versions, then
follow that release's requirements. Never infer compatibility from the word
"latest" across PyPI, documentation and container tags.

.. _release-checklist-for-maintainers:

Maintainers: follow :doc:`dev_release` for the release checklist, publication
steps, and hosted documentation settings.

.. _installation_release_channels:

Choose a release channel
------------------------

**Stable release:** install the version appropriate for your analysis, then
select that version in the documentation. The PyPI release description contains
its installation requirements; older ROOT-based releases have different
requirements from the native Python path described below.

.. code-block:: bash

   python -m pip install pycwb
   pycwb --version

**Prerelease:** alpha releases require an explicit version or ``--pre``. An
explicit version is preferable when preserving an analysis environment.

.. code-block:: bash

   python -m pip install --pre pycwb
   pycwb --version

Check the `PyPI release history <https://pypi.org/project/PycWB/#history>`_
for available versions. Do not assume a prerelease contains every feature
shown in the development documentation.

Compatibility changes
---------------------

Document CLI, YAML, Python API and output-format changes separately. For each
breaking change, provide an old/new example and state whether existing runs can
be resumed or must be regenerated. Scientific changes such as detector geometry,
normalization, thresholds or matching conventions need explicit release notes
even when the file format is unchanged.

The project does not yet promise a fixed deprecation-support interval or a
stable interface for every internal module. Prefer documented workflow APIs;
record exact revisions for custom integrations. Existing catalog compatibility
rules are described in :ref:`postproduction`.
