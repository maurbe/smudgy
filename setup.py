import platform
from setuptools import setup, find_packages

import numpy as np
from pybind11.setup_helpers import Pybind11Extension, build_ext


system = platform.system()
extra_compile_args = []
extra_link_args = []

if system == "Darwin":  # macOS
    extra_compile_args = ["-std=c++17", "-O3", "-Xpreprocessor", "-fopenmp"]
    extra_link_args = ["-lomp"]
elif system == "Linux":
    extra_compile_args = ["-std=c++17", "-O3", "-fopenmp"]
    extra_link_args = ["-fopenmp"]
else:
    extra_compile_args = ["/std:c++17", "/O2", "/openmp"]
    extra_link_args = []


ext_modules = [
    Pybind11Extension(
        name="sph_lib.cpp.functions",
        sources=[
            "sph_lib/bindings/bindings.cpp",
            "sph_lib/cpp/functions.cpp",
            "sph_lib/cpp/kernels.cpp",
        ],
        include_dirs=["sph_lib/cpp", np.get_include()],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c++",
    ),
]


setup(
    name="sph_lib",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
