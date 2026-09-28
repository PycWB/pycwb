# Contributing to PycWB

Start with the [contribution guide](https://docs.pycwb.org/en/latest/dev_contributing.html) or its [source](docs/source/dev_contributing.rst).

For a development environment, install the checkout with `python -m pip install -e ".[test]"`. Install documentation tools with `python -m pip install -r docs/requirements.txt`, then run `make doc-check`.

Explain the problem and resulting behavior, add relevant regression tests, and update documentation and `CHANGES.md` for user-visible changes. The scientific tests may require additional fixtures; see the [build/test guide](docs/source/dev_build_test.rst).

Questions and bug reports: use [GitLab issues](https://git.ligo.org/yumeng.xu/pycwb/-/issues), or email yumeng.xu@ligo.org if you cannot access the tracker. See [SUPPORT.md](SUPPORT.md).
