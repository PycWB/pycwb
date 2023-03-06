!#/bin/bash
python setup.py clean && python setup.py build_cwb && python setup.py sdist && pip install dist/*.tar.gz