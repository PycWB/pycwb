import os
from setuptools import setup
from setuptools import find_packages

requires = []
install_requires = [
    "pyyaml",
    "jsonschema",
    "watchfiles",
    "numpy",
    "gwpy",
    "ligo-segments",
    "aiohttp",
    "pycbc",
]


# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


def find_files(dirname, relpath=None):
    def find_paths(dirname):
        items = []
        for fname in os.listdir(dirname):
            path = os.path.join(dirname, fname)
            if os.path.isdir(path):
                items += find_paths(path)
            elif not path.endswith(".py") and not path.endswith(".pyc"):
                items.append(path)
        return items

    items = find_paths(dirname)
    if relpath is None:
        relpath = dirname
    print(items)
    return [os.path.relpath(path, relpath) for path in items]


setup(
    name="pycwb",
    author="Yumeng Xu",
    author_email="xusmailbox@gmail.com",
    description=("This is a project to simplify the installation of `cWB` and run `cWB` with python."),
    long_description=read('README.md'),
    long_description_content_type="text/markdown",
    keywords=['ligo', 'physics', 'gravity', 'signal processing', 'gravitational waves', 'cwb', 'coherent wave burst'],
    url="https://git.ligo.org/yumeng.xu/pycwb",
    install_requires=install_requires,
    scripts=["bin/pycwb_gen_config"],#find_files('bin', relpath='./'),
    packages=find_packages(),
    include_package_data=True,
    python_requires='>=3.8'
)
