"""Setup script for building the library with C++ extensions and OpenMP support."""

import os
import platform
import tempfile
import textwrap
from distutils.ccompiler import new_compiler
from distutils.sysconfig import customize_compiler

import numpy as np
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import find_packages, setup

system = platform.system()
has_openmp = False
extra_compile_args = []
extra_link_args = []


def _supports_openmp(compile_args, link_args) -> bool:
    test_code = textwrap.dedent("""
        #include <omp.h>
        int main() { return 0; }
        """)
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "omp_test.cpp")
        with open(src, "w", encoding="utf-8") as handle:
            handle.write(test_code)
        compiler = new_compiler()
        customize_compiler(compiler)
        try:
            obj_files = compiler.compile([src], extra_postargs=compile_args)
            compiler.link_executable(
                obj_files, os.path.join(tmp, "a.out"), extra_postargs=link_args
            )
            return True
        except Exception:
            return False


if system == "Darwin":
    # Dynamically set CC and CXX for macOS using Homebrew
    gcc_path = os.popen("brew list gcc | grep '/bin/gcc-' | head -1").read().strip()
    gxx_path = os.popen("brew list gcc | grep '/bin/g++-' | head -1").read().strip()
    if gcc_path and gxx_path:
        os.environ["CC"] = gcc_path
        os.environ["CXX"] = gxx_path
    else:
        raise OSError(
            "GCC and G++ paths could not be determined. Ensure GCC is installed via Homebrew."
        )

    extra_compile_args = ["-std=c++17", "-O3", "-fopenmp"]
    extra_link_args = ["-fopenmp"]

elif system == "Linux":
    # Validate GCC and G++ paths for Linux
    gcc_path = os.popen("which gcc").read().strip()
    gxx_path = os.popen("which g++").read().strip()
    if gcc_path and gxx_path:
        os.environ["CC"] = gcc_path
        os.environ["CXX"] = gxx_path
    else:
        raise OSError("GCC and G++ compilers are not installed or not in PATH.")

    extra_compile_args = ["-std=c++17", "-O3", "-fopenmp"]
    extra_link_args = ["-fopenmp"]

else:
    extra_compile_args = ["/std:c++17", "/O2", "/openmp"]
    extra_link_args = []
"""
if system == "Darwin":  # macOS
    extra_compile_args = ["-std=c++17", "-O3", "-Xpreprocessor", "-fopenmp"]
    extra_link_args = ["-fopenmp"]
elif system == "Linux":
    extra_compile_args = ["-std=c++17", "-O3", "-fopenmp"]
    extra_link_args = ["-fopenmp"]
else:
    extra_compile_args = ["/std:c++17", "/O2", "/openmp"]
    extra_link_args = []
"""
if not _supports_openmp(extra_compile_args, extra_link_args):
    extra_compile_args = (
        ["-std=c++17", "-O3"] if system != "Windows" else ["/std:c++17", "/O2"]
    )
    extra_link_args = []
    has_openmp = False
else:
    has_openmp = True


ext_modules = [
    Pybind11Extension(
        name="smudgy.core._cpp_functions_ext",
        sources=[
            "smudgy/bindings/_bindings.cpp",
            "smudgy/core/cpp/_functions.cpp",
            "smudgy/core/cpp/_kernels.cpp",
            "smudgy/core/cpp/_integration.cpp",
        ],
        include_dirs=["smudgy/core/cpp", np.get_include()],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=[("OPENMP_AVAILABLE", "1")] if has_openmp else [],
        language="c++",
    ),
]


setup(
    name="smudgy",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    #long_description=open("README.md").read(),
    #long_description_content_type="text/markdown",
)
