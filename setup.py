import os
import re
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


# A CMakeExtension needs a sourcedir instead of a file list.
# The name must be the _single_ output extension from the CMake build.
# If you need multiple extensions, see scikit-build.
class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:
        # Give up on windows
        if self.compiler.compiler_type == "msvc":
            raise RuntimeError("MSVC not supported")

        # Must be in this form due to bug in .resolve() only fixed in Python 3.10+
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve() / "neml2"

        # Configure arguments
        cmake_args = [
            "-DCMAKE_INSTALL_PREFIX={}".format(extdir),
            "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
            "-DCMAKE_UNITY_BUILD=ON",
            "-DNEML2_PYBIND=OFF",
            "-DNEML2_TESTS=ON",
            "-DNEML2_RUNNER=OFF",
            "-DNEML2_DOC=OFF",
        ]

        # Build arguments
        build_args = ["-j{}".format(os.environ.get("BUILD_JOBS", "1"))]

        # Install arguments
        install_args = []

        if sys.platform.startswith("darwin"):
            # Cross-compile support for macOS - respect ARCHFLAGS if set
            archs = re.findall(r"-arch (\S+)", os.environ.get("ARCHFLAGS", ""))
            if archs:
                cmake_args += ["-DCMAKE_OSX_ARCHITECTURES={}".format(";".join(archs))]

        build_temp = Path(self.build_temp)
        if not build_temp.exists():
            build_temp.mkdir(parents=True)

        subprocess.run(
            ["cmake", ext.sourcedir, *cmake_args], cwd=build_temp, check=True
        )
        subprocess.run(
            ["cmake", "--build", ".", *build_args], cwd=build_temp, check=True
        )
        subprocess.run(
            ["cmake", "--install", ".", *install_args], cwd=build_temp, check=True
        )


setup(
    ext_modules=[CMakeExtension("neml2")],
    cmdclass={"build_ext": CMakeBuild},
)
