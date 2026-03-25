import os
import re
import shutil
import subprocess
import platform
import sys
import importlib.util
import warnings

from setuptools import setup, Command, Extension
from setuptools.command.build_ext import build_ext


class BuildCWB(Command):
    user_options = []
    description = "Build the core functions of cWB"

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        pwd = os.getcwd()
        try:
            if os.path.exists('dist'):
                self.spawn(['rm', '-rf', 'dist'])
            os.chdir('cwb-core')
            self.spawn(['bash', './build.sh'])
        except Exception:
            self.warn('cwb core compilation failed, skipping')
            raise
        finally:
            os.chdir(pwd)


class Clean(Command):
    user_options = []
    description = "Remove build artifacts"

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        for folder in ['build', 'dist']:
            shutil.rmtree(folder, ignore_errors=True)
            print(f'removed {folder}')


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            )

        if platform.system() == "Windows":
            from packaging.version import Version
            cmake_version = Version(re.search(r'version\s*([\d.]+)', out.decode()).group(1))
            if cmake_version < Version('3.1.0'):
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [f'-DCMAKE_INSTALL_PREFIX:PATH={extdir}']
        build_args = ['--', '-j']

        os.makedirs(self.build_temp, exist_ok=True)
        env = os.environ.copy()
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)
        subprocess.check_call(['cmake', '--build', '.', '--target', 'install'] + build_args, cwd=self.build_temp)


def has_root():
    if os.environ.get("PYCWB_DISABLE_WAT", "").lower() in {"1", "true", "yes"}:
        return False

    if os.environ.get("PYCWB_FORCE_WAT", "").lower() in {"1", "true", "yes"}:
        return True

    if importlib.util.find_spec("ROOT") is not None:
        return True

    root_config = shutil.which("root-config")
    if root_config:
        try:
            subprocess.check_output([root_config, "--version"], stderr=subprocess.STDOUT)
            return True
        except Exception:
            pass

    root_sys = os.environ.get("ROOTSYS")
    if root_sys:
        root_bin = os.path.join(root_sys, "bin", "root")
        if os.path.exists(root_bin):
            return True

    return False


cmdclass = {
    'build_cwb': BuildCWB,
    'clean': Clean,
}

ext_modules = []
if has_root():
    cmdclass['build_ext'] = CMakeBuild
    ext_modules = [CMakeExtension('wavelet', 'cwb-core')]
else:
    message = "ROOT not found in build environment: skipping C++ wavelet extension build"
    warnings.warn(message)
    sys.stderr.write(message + "\n")


setup(
    cmdclass=cmdclass,
    ext_modules=ext_modules,
)
