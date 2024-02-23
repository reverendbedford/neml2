import os
import sys
import platform
import subprocess
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
        ext_dir = Path(self.get_ext_fullpath(ext.name)).parent.resolve()

        Path(self.build_temp).mkdir(parents=True, exist_ok=True)

        configure_args = [
            "cmake",
            "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
            "-DCMAKE_UNITY_BUILD=ON",
            "-DNEML2_UNIT=OFF",
            "-DNEML2_REGRESSION=OFF",
            "-DNEML2_VERIFICATION=OFF",
            "-DNEML2_BENCHMARK=OFF",
            "-DNEML2_PROFILING=OFF",
            "-DNEML2_PYBIND=ON",
            "-DNEML2_DOC=OFF",
            "-DCMAKE_INSTALL_PREFIX=" + ext_dir,
            "-DPYTHON_EXECUTABLE={}".format(sys.executable),
            "-S{}".format(ext.sourcedir),
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


setup()
