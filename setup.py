# -*- coding: utf-8 -*-
import os
import sys
import subprocess
from platform import python_version

from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext


def parse_requirements_file(filename):
    with open(filename, encoding="utf-8") as fid:
        requires = [l.strip() for l in fid.readlines() if l]
    return requires


# https://github.com/pybind/cmake_example/blob/c45488dfdff04eec43fd2e59fcf9d9cd21b83880/setup.py
class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " + ", ".join(e.name for e in self.extensions))
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + extdir,
            "-DPYTHON_EXECUTABLE=" + sys.executable,
            "-DPYTHON_INTERPRETER_VERSION_CALLING_SETUP_SCRIPT=" + str(python_version()),
        ]

        cfg = "Debug" if self.debug else "Release"
        build_args = ["--config", cfg]

        cmake_args += ["-DCMAKE_BUILD_TYPE=" + cfg]
        build_args += ["--", "-j"]

        env = os.environ.copy()
        env["CXXFLAGS"] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get("CXXFLAGS", ""), self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=self.build_temp)


INSTALL_REQUIRES = parse_requirements_file("requirements.txt")
setup(
    name="pyhjs",
    version="1.0.0",
    description='Python wrapper of "Finding the Skeleton of 2D Shape and Contours: Implementation of Hamilton-Jacobi Skeleton"',
    cmdclass={"build_ext": CMakeBuild},
    packages=find_packages(),
    ext_modules=[CMakeExtension("pyhjs")],
    install_requires=INSTALL_REQUIRES,
    zip_safe=False,
)
