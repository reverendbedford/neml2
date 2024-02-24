#! /usr/bin/env python

# Copyright 2023, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: NEML2 -- the New Engineering material Model Library, version 2
# By: Argonne National Laboratory
# OPEN SOURCE LICENSE (MIT)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import os
import sys
import platform
import subprocess
import sysconfig
from pathlib import Path

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            )

        if platform.system() == "Windows":
            raise RuntimeError("Windows is not currently not supported")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        ext_dir = str(Path(self.get_ext_fullpath(ext.name)).parent.resolve())
        torch_dir = str(Path(sysconfig.get_path("purelib")) / "torch")
        Path(self.build_temp).mkdir(parents=True, exist_ok=True)

        configure_args = [
            "cmake",
            ext.sourcedir,
            "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
            "-DCMAKE_UNITY_BUILD=ON",
            "-DBUILD_TESTING=OFF",
            "-DLIBTORCH_DIR={}".format(torch_dir),
            "-DNEML2_UNIT=OFF",
            "-DNEML2_REGRESSION=OFF",
            "-DNEML2_VERIFICATION=OFF",
            "-DNEML2_BENCHMARK=OFF",
            "-DNEML2_PROFILING=OFF",
            "-DNEML2_PYBIND=ON",
            "-DNEML2_DOC=OFF",
            "-DCMAKE_INSTALL_PREFIX=" + ext_dir,
            "-DPYTHON_EXECUTABLE={}".format(sys.executable),
            "-B{}".format(self.build_temp),
        ]

        build_args = [
            "cmake",
            "--build",
            self.build_temp,
            "--config",
            "RelWithDebInfo",
            "--target",
            "install",
        ]

        env = os.environ.copy()

        subprocess.check_call(configure_args, env=env)
        subprocess.check_call(build_args)


setup(
    ext_modules=[CMakeExtension("neml2")],
    cmdclass={"build_ext": CMakeBuild},
)
