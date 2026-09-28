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

Release checklist for maintainers
---------------------------------

1. Update ``CHANGES.md`` with user-visible changes, known limitations and upgrade
   instructions, including scientific behavior changes.
2. Run the regular tests, strict documentation build and packaged demo. Record
   which numerical reference comparisons were run and their outcome.
3. Verify a clean installation from the built wheel/source distribution,
   including packaged templates. Check version reporting outside the checkout.
4. Create the release tag using the existing release process. The GitLab tag
   pipeline builds and publishes the source distribution; avoid a second manual
   upload of the same version.
5. Enable the tag's documentation build in Read the Docs, retain older release
   documentation, and point the stable alias only at a stable release.
6. Publish release notes and, when available, an immutable software archive/DOI.
   Document container tags and digests that were actually built and tested.

Enabling hosted documentation versions and publishing archives are maintainer
service settings; changing the repository alone does not publish a release.

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
