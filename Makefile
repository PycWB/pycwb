install: clean sdist
	pip install dist/*.tar.gz

clean:
	python setup.py clean

build_cwb:
	python setup.py build_cwb

sdist:
	python setup.py sdist

sdist_clean:
	rm -rf dist

doc: clean_doc
	TZ=UTC sphinx-apidoc -o docs/source pycwb pycwb/vendor/* && cd docs && TZ=UTC make html

clean_doc:
	cd docs && make clean && rm -f source/modules.rst source/pycwb*.rst

quick_update: sdist_clean sdist
	pip install dist/*.tar.gz

install_doc_deps:
	pip install sphinx sphinx_rtd_theme