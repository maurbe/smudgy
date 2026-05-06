# Installation

You can install `smudgy` in several ways. OpenMP support is optional but recommended for parallel performance. See the {ref}`openmp-support` section for details on how to setup OpenMP **before** installation.

1. Install from PyPI (recommended)

```bash
pip install smudgy
```

2. To build from source, clone the repository and install

```bash
git clone https://github.com/maurbe/smudgy.git
cd smudgy
pip install .
```

(openmp-support)=
## OpenMP Support 
[//]: # (<img src="_static/test.png" alt="Icon" class="inline-title-icon">)

To enable parallelization, you must have OpenMP installed on your system **before** installing `smudgy`.
If OpenMP is not found, `smudgy` will still work, but parallelization will be disabled.
To install it, see instructions for your operating system below. If you are unsure whether OpenMP is installed correctly, download the `src/omp_test.cpp` file from the repository and test the OpenMP installation depending on your OS.

::::{tab-set}
:::{tab-item} Linux
Most Linux distributions provide OpenMP support out of the box with the `gcc` compiler. 

Install the required packages:

```bash
sudo apt-get update
sudo apt-get install -y libomp-dev
```

Test it:

```bash
g++ -fopenmp omp_test.cpp -o omp_test
./omp_test # should print "OpenMP available"
```
:::

:::{tab-item} MacOS
Apple's `Clang` compiler does not support OpenMP by default. To install it OpenMP, follow these steps.

First, install the Xcode Command Line Tools:

```bash
xcode-select --install
```

Then, install `libomp` via Homebrew:

```bash
brew update
brew install libomp
```

Then, export the necessary environment variables:

```bash
export OMP_PREFIX="$(brew --prefix libomp)"
export CPPFLAGS="-I${OMP_PREFIX}/include"
export LDFLAGS="-L${OMP_PREFIX}/lib"
export CFLAGS="-Xpreprocessor -fopenmp ${CPPFLAGS}"
export CXXFLAGS="-Xpreprocessor -fopenmp ${CPPFLAGS}"
```

Test it:

```bash
clang++ -Xpreprocessor -fopenmp omp_test.cpp \
    -I${OMP_PREFIX}/include \
    -L${OMP_PREFIX}/lib -lomp \
    -o omp_test
./omp_test # should print "OpenMP available"
```
:::

:::{tab-item} Windows
```{caution}
Windows is currently not officially supported. However, the following instructions are provided for users who wish to attempt installation on Windows systems.
```

[MSVC](https://visualstudio.microsoft.com/de/vs/features/cplusplus/) is recommended for Windows users as it already includes OpenMP. The build system will automatically enable OpenMP if available. Compile and test it:

```bash
cl /openmp omp_test.cpp
omp_test.exe # should print "OpenMP available"
```
:::
::::

## Running the Test Suite

After installation, it is highly recommended to run the test suite to verify your installation and check OpenMP support.

Identify the `smudgy` installation directory and run `pytest`:

```bash
python -m pytest "$(python -c 'import smudgy, os; print(os.path.dirname(smudgy.__file__))')"
```

The tests will automatically check for OpenMP availability and skip parallelization tests if OpenMP is not enabled or detected.
If you see tests being skipped due to missing OpenMP, revisit the {ref}`openmp-support` section to ensure your environment is set up correctly.

If you have any issues, please consult the documentation or open an issue on [GitHub](https://github.com/maurbe/smudgy/issues).
