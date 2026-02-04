import os, shutil
import re
import subprocess, platform
import sys

from setuptools import setup, Command, Extension
from setuptools import find_packages
from setuptools.command.build_ext import build_ext
from distutils.errors import DistutilsExecError
from distutils.command.clean import clean as _clean
from distutils.version import LooseVersion

requires = []
install_requires = [
    "matplotlib",
    "astropy",
    "pyyaml",
    "jsonschema",
    "watchfiles",
    "numpy",
    "numba",
    "gwpy",
    "ligo-segments",
    "ligo-gracedb",
    "aiohttp",
    "pycbc",
    "filelock",
    "scipy<1.14", # required by healpy
    "pillow>=9.0.0",
    "click",
    "orjson",
    "dacite",
    "lalsuite>=7.0.0",
    "prefect",
    "prefect-dask",
    "dask",
    "dask_jobqueue",
    "htcondor",
    "psutil",
    "memspectrum",
    "exceptiongroup>=1.0.0;python_version<'3.11'",  # Backport for Python < 3.11
    "python-ligo-lw<2.0.0",  # error for version > 2: FileNotFoundError: [Errno 2] No such file or directory: 'ligolw/version.py.in'
    "scikit-learn",
    "jinja2",  # for template rendering
    "plotly",  # for plotting
    "healpy",
    "wdm-wavelet",
    "burst-waveform",
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
        self.clean_files = []
        self.clean_folders = ['dist']
        # self.clean_files = ["pycwb/vendor/lib/libWAT_rdict.pcm", 'pycwb/vendor/lib/libwavelet.so',
        #                     'pycwb/vendor/lib/wavelet.so', 'pycwb/vendor/lib/wavelet-4x.dylib']
        # self.clean_folders = ['cwb-core/build', 'dist', 'pycwb.egg-info']

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


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " + ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)', out.decode()).group(1))
            if cmake_version < '3.1.0':
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ['-DCMAKE_INSTALL_PREFIX:PATH=' + extdir]

        build_args = ['--', '-j']#['--config', cfg]

        env = os.environ.copy()
        # env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
        #                                                       self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        # subprocess.check_call(['bash', 'build.sh', extdir], cwd=ext.sourcedir, env=env)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)
        subprocess.check_call(['cmake', '--build', '.', '--target', 'install'] + build_args, cwd=self.build_temp)


setup(
    long_description=read('README.md'),
    long_description_content_type="text/markdown",
    url="https://git.ligo.org/yumeng.xu/pycwb",
    install_requires=install_requires,
    cmdclass={
        'build_cwb': BuildCWB,
        'build_ext': CMakeBuild,
        'clean': Clean
    },
    ext_modules=[CMakeExtension('wavelet', 'cwb-core')],
    scripts=["bin/pycwb", "bin/pycwb_search", "bin/pycwb_show"],  # find_files('bin', relpath='./'),
    packages=find_packages(),
    include_package_data=True,
)
