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

doc:
	TZ=UTC python -m sphinx -b html docs/source docs/build/html

doc-check:
	TZ=UTC python -m sphinx -W --keep-going -b html docs/source docs/build/html

clean_doc:
	cd docs && make clean && rm -f source/modules.rst source/pycwb*.rst

quick_update: sdist_clean sdist
	pip install dist/*.tar.gz

install_doc_deps:
	python -m pip install -r docs/requirements.txt
