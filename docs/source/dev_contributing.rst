.. _dev_contributing:

Contributing
============

How to contribute to pycWB—pull request workflow, code style, and review
process.

.. contents:: Table of Contents
   :depth: 2
   :local:


Getting Started
---------------

1. **Fork the repo** on LIGO GitLab.
2. **Set up your dev environment** (:ref:`dev_setup`).
3. **Find an issue** or propose a new feature.
4. **Create a branch**: ``feature/<description>`` or ``fix/<description>``.


Pull Request Workflow
---------------------

.. code-block:: bash

   # 1. Create a branch
   git checkout -b feature/my-new-module

   # 2. Make changes + tests
   # ... edit code ...

   # 3. Run tests locally
   pytest
   python -m unittest discover tests/

   # 4. Commit with descriptive message
   git commit -m "Add my_new_module: <one-line description>

   <Paragraph explaining what, why, and any design decisions>"

   # 5. Push and create MR/PR
   git push origin feature/my-new-module


Code Style
----------

- **Python**: Follow PEP 8. Use type hints for all public functions.
- **Docstrings**: NumPy-style with Parameters/Returns/Raises sections.
- **Naming**: ``snake_case`` for functions/variables, ``CamelCase`` for
  classes, ``UPPER_CASE`` for constants.
- **Imports**: standard library → third-party → pycwb internal, each group
  separated by a blank line.
- **Logging**: use ``logging.getLogger(__name__)``, not ``print()``.
- **Config parameters**: ``snake_case``, descriptive names with units in
  the schema description.


What to Include in a PR
-----------------------

Every pull request should include:

1. **Code changes** — focused, minimal diff.
2. **Unit tests** — covering new functionality and edge cases.
3. **Docstring updates** — if public API changes.
4. **CHANGES.md** entry — brief note under the appropriate version.
5. **Schema update** — if new config parameters are added.

For new modules, also include:
- ``__init__.py`` with public API imports
- ``tests/__init__.py``
- At least one test file


Review Checklist
----------------

Reviewers will check:

- [ ] Tests pass locally and in CI
- [ ] New code has tests
- [ ] Docstrings are complete and accurate
- [ ] No new ROOT dependencies (ROOT is being phased out)
- [ ] No sideways imports between modules
- [ ] Hot-path code uses Numba or JAX, not pure NumPy
- [ ] JAX device buffers are freed after use
- [ ] Config schema updated for new parameters
- [ ] CHANGES.md updated
- [ ] No breaking changes to the YAML config format without migration path


Documentation Maintenance
-------------------------

Every PR that changes user-facing behavior must update the docs:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - If you changed...
     - Update...
   * - A config parameter (new or modified)
     - ``user_parameters_schema.py`` + :ref:`schema`
   * - A pipeline stage or algorithm
     - The relevant :ref:`core_concepts` page + :ref:`pipeline_lifecycle`
   * - A CLI command or workflow
     - :ref:`analysis_recipes` (if a recipe is affected) + :ref:`standard_analysis`
   * - A public Python API
     - Docstring in the source file (auto-documented in :ref:`modules`)
   * - The build or test system
     - :ref:`dev_build_test`
   * - A new term or concept
     - :ref:`glossary`
   * - Anything user-facing
     - :ref:`choose_your_path` (check if paths need updating)

**PR doc checklist** (add to PR description):

.. code-block:: text

   - [ ] Docstring updated (if API changed)
   - [ ] Schema page updated (if new/changed params)
   - [ ] Core Concepts page updated (if algorithm changed)
   - [ ] Recipe updated (if workflow changed)
   - [ ] Tutorial updated (if user flow changed)
   - [ ] Glossary updated (if new terms)
   - [ ] CHANGES.md entry added


Release Process
---------------

Releases are versioned with ``setuptools_scm`` from Git tags.

.. code-block:: bash

   # 1. Update CHANGES.md with release notes
   # 2. Tag the release
   git tag -a v1.1.0 -m "Release v1.1.0"
   git push --tags

   # 3. Build and upload to PyPI
   python -m build
   twine upload dist/*

   # 4. Update conda-forge feedstock (if applicable)

Versioning follows ``MAJOR.MINOR.PATCH``:
- **MAJOR**: breaking config format changes
- **MINOR**: new features, modules, or significant improvements
- **PATCH**: bug fixes, documentation, performance


Where to Ask Questions
----------------------

- **Bug reports / feature requests**: LIGO GitLab issues
- **Development discussion**: LIGO Slack #cwb channel
- **Documentation**: This site (``docs/``)
