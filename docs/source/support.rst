.. _support:

Help and Community
==================

Public source and issues are hosted on
`LIGO GitLab <https://git.ligo.org/yumeng.xu/pycwb>`_. Use an anonymous HTTPS
clone for reading the source; an SSH key is not required for downloading it.
Creating issues or merge requests may require an account.

Ask a question or report a problem
----------------------------------

Use the `issue tracker <https://git.ligo.org/yumeng.xu/pycwb/-/issues>`_ when
you have access. If you cannot access it or do not have a collaboration account,
email the package contact at `yumeng.xu@ligo.org <mailto:yumeng.xu@ligo.org>`_.
You do not need access to LIGO Slack to report a problem. No response-time
commitment is currently advertised.

For a bug report, include:

* the PycWB version or source revision and whether it has local modifications;
* operating system, Python version and ``pycwb doctor`` output;
* the exact command, smallest useful YAML and a public/synthetic reproducer;
* expected behavior, observed behavior and relevant traceback;
* whether the unmodified standalone example in ``examples/demo/`` works.

Keep private data and credentials out of public issues. Send a suspected
security problem privately to the package contact rather than publishing
sensitive details in an ordinary issue.

Contribute
----------

Documentation fixes, reproducible bug reports, installation reports and tests
are useful contributions. See :ref:`dev_contributing`. Start with a small
change and explain the user-facing behavior it improves. Be respectful and
constructive; critique the work rather than the person.

Near-term roadmap
-----------------

The next maturity steps are a published native stable release with matching
documentation, clean-install coverage on more platforms, release-specific
scientific validation reports, and reproducible versioned containers. These are
work items, not claims that those platforms or releases are already validated.
