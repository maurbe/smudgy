import platform
import os
from setuptools import setup, find_packages

try:
    import torch
    from torch.utils.cpp_extension import CppExtension, BuildExtension, include_paths
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not found. Attempting to continue...")

system = platform.system()
extra_compile_args = []
extra_link_args = []
ext_modules = []

if TORCH_AVAILABLE:
    torch_lib_dir = os.path.join(torch.__path__[0], "lib")
    
    if system == "Darwin":  # macOS
        extra_compile_args = ["-Xpreprocessor", "-fopenmp"]
        rpath_list = [
            "-lomp",
            f"-Wl,-rpath,{torch_lib_dir}",
        ]
        if "CONDA_PREFIX" in os.environ:
            conda_lib = os.path.join(os.environ['CONDA_PREFIX'], 'lib')
            rpath_list.append(f"-Wl,-rpath,{conda_lib}")
        rpath_list.extend([
            "-Wl,-rpath,@loader_path",
            "-Wl,-rpath,@executable_path/../lib",
        ])
        extra_link_args = rpath_list
    elif system == "Linux":
        extra_compile_args = ["-fopenmp"]
        extra_link_args = ["-fopenmp", f"-Wl,-rpath,{torch_lib_dir}"]
    else:
        extra_compile_args = ["/openmp"]
        extra_link_args = []

    # Define C++ extension
    ext_modules = [
        CppExtension(
            name="sph_lib.cpp.functions",  # this is the module Python will import
            sources=[
                "sph_lib/bindings/bindings.cpp",  # pybind11 bindings
                "sph_lib/cpp/functions.cpp",               # backend numerics
                "sph_lib/cpp/kernels.cpp",                 # backend kernels
            ],
            include_dirs=include_paths() + ["sph_lib/cpp"],  # include path for kernels.h
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        ),
    ]

setup(
    name="sph_lib",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension} if TORCH_AVAILABLE else {},
)
