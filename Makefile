install: clean build_cwb sdist
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
	sphinx-apidoc -o docs/source pyburst pyburst/vendor/* && cd docs && make html

clean_doc:
	cd docs && make clean && rm -f source/modules.rst source/pyburst*.rst

quick_update: sdist_clean sdist
	pip install dist/*.tar.gz