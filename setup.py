"""Setup script for building smudgy with C++17 and optional OpenMP support.
Automatically falls back to serial build if OpenMP is unavailable.
"""

import platform
from distutils.errors import CompileError, LinkError

import numpy as np
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import find_packages, setup

system = platform.system()


# -------------------------------------------------
# OpenMP flag configuration per platform
# -------------------------------------------------


def get_openmp_flags():
    if system == "Darwin":
        return (
            ["-std=c++17", "-O3", "-Xpreprocessor", "-fopenmp"],
            ["-lomp"],
        )
    elif system == "Linux":
        return (
            ["-std=c++17", "-O3", "-fopenmp"],
            ["-fopenmp"],
        )
    elif system == "Windows":
        return (
            ["/std:c++17", "/O2", "/openmp"],
            [],
        )
    else:
        raise RuntimeError(f"Unsupported platform: {system}")


def get_serial_flags():
    if system == "Windows":
        return (["/std:c++17", "/O2"], [])
    else:
        return (["-std=c++17", "-O3"], [])


# -------------------------------------------------
# Custom build_ext with automatic fallback
# -------------------------------------------------


class BuildExtWithFallback(build_ext):

    def build_extensions(self):
        openmp_compile, openmp_link = get_openmp_flags()
        serial_compile, serial_link = get_serial_flags()

        # First try with OpenMP
        for ext in self.extensions:
            ext.extra_compile_args = openmp_compile
            ext.extra_link_args = openmp_link

        try:
            print("\nAttempting to build with OpenMP support...\n")
            super().build_extensions()
            print("\nOpenMP successfully enabled.\n")
        except (CompileError, LinkError):
            print("\nOpenMP build failed. Falling back to serial build.\n")

            for ext in self.extensions:
                ext.extra_compile_args = serial_compile
                ext.extra_link_args = serial_link

            super().build_extensions()
            print("\nSerial build completed (OpenMP disabled).\n")


# -------------------------------------------------
# Extension definition
# -------------------------------------------------

ext_modules = [
    Pybind11Extension(
        name="smudgy.core._cpp_functions_ext",
        sources=[
            "smudgy/bindings/_bindings.cpp",
            "smudgy/core/cpp/_functions.cpp",
            "smudgy/core/cpp/_kernels.cpp",
            "smudgy/core/cpp/_integration.cpp",
        ],
        include_dirs=[
            "smudgy/core/cpp",
            np.get_include(),
        ],
        language="c++",
    ),
]


# -------------------------------------------------
# Setup
# -------------------------------------------------

setup(
    name="smudgy",
    version="0.1.2",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtWithFallback},
)
