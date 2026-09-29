.. _dev_release:

Release procedure
=================

Release checklist for maintainers
---------------------------------

1. Update ``CHANGES.md`` with user-visible changes, known limitations and upgrade
   instructions, including scientific behavior changes.
2. Run the regular tests, strict documentation build and standalone recovery example. Record
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

For user-facing version and compatibility guidance, see :doc:`release_policy`.
