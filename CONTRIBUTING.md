# Contributing to PycWB

Start with the [contribution guide](https://docs.pycwb.org/en/latest/dev_contributing.html) or its [source](docs/source/dev_contributing.rst).

Use Python 3.11 or newer. For a development environment, install the checkout with `python -m pip install -e ".[test]"`. Install documentation tools with `python -m pip install -r docs/requirements.txt`, then run `make doc-check`. Run lint and mypy in the separate Python 3.11 environment described in the [build/test guide](docs/source/dev_build_test.rst#lint-and-type-checks).

Explain the problem and resulting behavior, add relevant regression tests, and update documentation and `CHANGES.md` for user-visible changes. The scientific tests may require additional fixtures; see the [build/test guide](docs/source/dev_build_test.rst).

The [GitHub](https://github.com/PycWB/pycwb) and [LIGO GitLab](https://git.ligo.org/yumeng.xu/pycwb) repositories are mirrored. Fork either repository and submit a GitHub pull request or GitLab merge request. Contributors outside LVK can use GitHub without an LVK account.

Questions and bug reports: use [GitHub issues](https://github.com/PycWB/pycwb/issues) or [GitLab issues](https://git.ligo.org/yumeng.xu/pycwb/-/issues), or email yumeng.xu@ligo.org if you cannot access either tracker. See [SUPPORT.md](SUPPORT.md).
