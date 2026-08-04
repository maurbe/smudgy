# Installation

You can install `smudgy` in several ways. MPI support is optional but recommended for parallel performance over multiple ranks. See the {ref}`mpi-support` section for details on how to install MPI.

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

(mpi-support)=
## MPI Support

To enable MPI, you must have installed it on your system **before** installing `smudgy`.
If MPI is not found, `smudgy` will still work, but parallelization over ranks will be disabled.
To install it, see instructions for your operating system below.

::::{tab-set}
:::{tab-item} Linux
Install MPI with your local package manager:

```bash
sudo apt install libopenmpi-dev openmpi-bin
```

:::

:::{tab-item} MacOS
The easiest way is to use the homebrew package manager:

```bash
brew install open-mpi
```

:::

:::{tab-item} Windows

For Windows, the easiest way to install MPI is via [Microsoft-MPI](https://learn.microsoft.com/en-us/message-passing-interface/microsoft-mpi).

:::

:::{tab-item} HPC Cluster
On most HPC clusters, MPI is already installed and managed by the system.

Simply load the appropriate MPI module before installing or running `smudgy`, for example:

```bash
module load openmpi
```

Consult your cluster documentation for the correct module name.
:::
::::

If you are unsure whether MPI is installed correctly, run the following command and check its output

```bash
mpirun --version
```

After you installed MPI and `smudgy`, run the following command to check whether `mpi4py` (installed as a dependency) has found your local MPI installation

```bash
python -c "from mpi4py import MPI; print(MPI.Get_library_version())"
```


## Running the Test Suite

After installation, it is recommended to run the test suite to verify your installation.

Identify the `smudgy` installation directory and run `pytest`:

```bash
python -m pytest --pyargs smudgy
```

If you have any issues, please consult the documentation or open an issue on [GitHub](https://github.com/maurbe/smudgy/issues).
