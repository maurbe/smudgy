"""
Setup script for building smudgy with C++17 and OpenMP support.
"""

import platform
import sys

import numpy as np
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import find_packages, setup

system = platform.system()

# -------------------------
# Compiler flags per OS
# -------------------------

if system == "Darwin":
    # Apple Clang + Homebrew libomp
    extra_compile_args = [
        "-std=c++17",
        "-O3",
        "-Xpreprocessor",
        "-fopenmp",
    ]
    extra_link_args = ["-lomp"]

elif system == "Linux":
    # GCC (default on Linux runners)
    extra_compile_args = [
        "-std=c++17",
        "-O3",
        "-fopenmp",
    ]
    extra_link_args = ["-fopenmp"]

elif system == "Windows":
    extra_compile_args = ["/std:c++17", "/O2", "/openmp"]
    extra_link_args = []

else:
    raise RuntimeError(f"Unsupported platform: {system}")

# -------------------------
# Extension definition
# -------------------------

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
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=[("OPENMP_AVAILABLE", "1")],
        language="c++",
    ),
]

# -------------------------
# Setup
# -------------------------

setup(
    name="smudgy",
    version="0.1.2",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)