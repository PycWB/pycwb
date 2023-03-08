import os, shutil
from setuptools import setup, Command
from setuptools import find_packages
from distutils.errors import DistutilsExecError
from distutils.command.clean import clean as _clean

requires = []
install_requires = [
    "matplotlib<3.7.0",
    "pyyaml",
    "jsonschema",
    "watchfiles",
    "numpy",
    "gwpy",
    "ligo-segments",
    "aiohttp",
    "pycbc",
    "filelock",
    # "nds2-client",
    # "python-nds2-client"
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


class BuildCWB(Command):
    user_options = []
    description = "Build the core functions of cWB"

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        # pwd
        pwd = os.getcwd()
        try:
            # remove dist
            print('removing old dist')
            if os.path.exists('dist'):
                self.spawn(['rm', '-rf', 'dist'])

            print('compiling cwb core')
            # build cwb core
            os.chdir('cwb-core')
            self.spawn(['bash', './build.sh'])
        except DistutilsExecError:
            self.warn('cwb core compilation failed, skipping')
            print('exiting cwb-core directory')
            os.chdir(pwd)
            raise DistutilsExecError
        finally:
            print('exiting cwb-core directory')
            os.chdir(pwd)


class Clean(_clean):
    def finalize_options(self):
        _clean.finalize_options(self)
        self.clean_files = ["pyburst/vendor/lib/libWAT_rdict.pcm", 'pyburst/vendor/lib/libwavelet.so',
                            'pyburst/vendor/lib/wavelet.so', 'pyburst/vendor/lib/wavelet-4x.dylib']
        self.clean_folders = ['cwb-core/build', 'dist', 'pyburst.egg-info']

    def run(self):
        _clean.run(self)
        for f in self.clean_files:
            try:
                os.unlink(f)
                print('removed {0}'.format(f))
            except:
                pass

        for fol in self.clean_folders:
            shutil.rmtree(fol, ignore_errors=True)
            print('removed {0}'.format(fol))


setup(
    name="pyburst",
    author="Yumeng Xu",
    author_email="xusmailbox@gmail.com",
    description=("This is a project to simplify the installation of `cWB` and run `cWB` with python."),
    long_description=read('README.md'),
    long_description_content_type="text/markdown",
    keywords=['ligo', 'physics', 'gravity', 'signal processing', 'gravitational waves', 'cwb', 'coherent wave burst'],
    url="https://git.ligo.org/yumeng.xu/pyburst",
    install_requires=install_requires,
    cmdclass={
        'build_cwb': BuildCWB,
        'clean': Clean
    },
    scripts=["bin/pyburst_gen_config"],  # find_files('bin', relpath='./'),
    packages=find_packages(),
    include_package_data=True,
)
