<br />
<p align="center">
  <img src="src/banner.png" alt="Smudgy Banner" width="100%">
</p>

Welcome to the `smudgy` GitHub page. 
`smudgy` is a python package with a C++ and (optionally) OpenMP backend for point cloud smoothing and interpolation using SPH operations – lightning fast, scalable and memory-efficient. Please refer to the official [docs](https://smudgy.readthedocs.io/en/latest/) for guides and tutorials.

You can install `smudgy` in several ways. OpenMP support is optional but recommended for parallel performance. See the [OpenMP Support](#openmp-support) section for details on how to setup OpenMP **before** installation.

#### 1. Install from PyPI (recommended)

```sh
pip install smudgy
```

#### 2. Build from Source

Clone the repository and install:

```sh
git clone https://github.com/maurbe/smudgy.git
cd smudgy
pip install .
```

### OpenMP Support

To enable parallelization, you must have OpenMP installed on your system **before** building the package. See instructions for your operating system below. If OpenMP is not found, `smudgy` will still work, but parallelization will be disabled.
The following OS-dependent instructions ensure that OpenMP is installed and configured for subsequent `smudgy` installation. If you are unsure whether OpenMP is installed correctly, download the `src/omp_test.cpp` file and test the OpenMP installation as described below.

#### Linux

Most Linux distributions provide OpenMP support out of the box with GCC. To ensure you have it, install the required packages:
```sh
sudo apt-get update
sudo apt-get install -y libomp5 libomp-dev
```
Test the installation:
```sh
g++ -fopenmp omp_test.cpp -o omp_test
./omp_test
```

#### macOS

Apple's Clang does not support OpenMP by default. To enable OpenMP parallelization, follow these steps:

First, install ``libomp`` via Homebrew:
```sh
brew update
brew install libomp
```

Then, export the necessary environment variables:
```sh
export OMP_PREFIX="$(brew --prefix libomp)"
export CPPFLAGS="-I${OMP_PREFIX}/include"
export LDFLAGS="-L${OMP_PREFIX}/lib"
export CFLAGS="-Xpreprocessor -fopenmp ${CPPFLAGS}"
export CXXFLAGS="-Xpreprocessor -fopenmp ${CPPFLAGS}"
```
Test the installation:
```sh
clang++ -Xpreprocessor -fopenmp omp_test.cpp \
        -I${OMP_PREFIX}/include \
        -L${OMP_PREFIX}/lib -lomp \
        -o omp_test
./omp_test
```

#### Windows

**Note:** Windows is currently not officially supported. However, the following instructions are provided for users who wish to attempt installation on Windows systems. On Windows, OpenMP is supported by Microsoft Visual Studio (MSVC) and MinGW compilers. 

**MSVC**: no extra installation is needed; OpenMP is included with MSVC. The build system will automatically enable OpenMP if available.
Compile and test the installation:
```sh
cl /openmp omp_test.cpp
omp_test.exe
```

**MinGW**: ensure you have a recent version with OpenMP support. You may need to add `-fopenmp` to your compiler flags if building manually. 
Compile and test the installation:
```sh
g++ -fopenmp omp_test.cpp -o omp_test.exe
omp_test.exe
```

### Running the Test Suite

After installation, it is highly recommended to run the test suite to verify your installation and check OpenMP support.

**Note**: If you installed `smudgy` via pip, the test suite is not included by default. In this case, clone the repository first:
```sh
git clone https://github.com/youruser/smudgy.git
cd smudgy
```

Afterwards, run the tests:
```sh
pytest -rs
```

The tests will automatically check for OpenMP availability and skip parallelization tests if OpenMP is not enabled or detected. If you see tests being skipped due to missing OpenMP, revisit the [OpenMP Support](#openmp-support) section to ensure your environment is set up correctly and the package was compiled with OpenMP support.

If you have any issues, please consult the documentation or open an issue on GitHub.