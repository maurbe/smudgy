
Installation
============

You can install smudgy in several ways. OpenMP support is optional but recommended for parallel performance. See below for details on installation and enabling OpenMP.

Install from PyPI (recommended)
-------------------------------
Simply run:

.. code-block:: bash

   pip install smudgy

See the OpenMP Support section if you want parallelization.


Build from Source
-----------------
Clone the repository and install:

.. code-block:: bash

   git clone https://github.com/maurbe/smudgy.git
   cd smudgy
   pip install .

OpenMP Support (Parallelization)
--------------------------------
To enable OpenMP parallelization, you must have OpenMP installed on your system **before** installing smudgy. See instructions for your operating system below. If OpenMP is not found, the package will still work, but parallelization will be disabled.


.. tab-set::
    .. tab-item:: Linux

        Most Linux distributions provide OpenMP support out of the box with GCC. To ensure you have it, install the required packages:

        .. code-block:: bash

            sudo apt-get update
            sudo apt-get install -y libomp-dev

    .. tab-item:: MacOS
        
        Apple's Clang does not support OpenMP by default. To enable OpenMP parallelization, install libomp via homebrew:

        1. Install GCC and libomp via Homebrew:

        .. code-block:: bash

            brew update
            brew install libomp

        2. Export the necessary environment variables:
        This sets the correct paths for GCC and G++ with the version you installed via Homebrew.

        .. code-block:: bash

            export OMP_PREFIX="$(brew --prefix libomp)"
            export CPPFLAGS="-I${OMP_PREFIX}/include"
            export LDFLAGS="-L${OMP_PREFIX}/lib"
            export CFLAGS="-Xpreprocessor -fopenmp ${CPPFLAGS}"
            export CXXFLAGS="-Xpreprocessor -fopenmp ${CPPFLAGS}"

    .. tab-item:: Windows

        .. note::
            Windows is currently not officially supported. However, the following instructions are provided for users who wish to attempt installation on Windows systems.

        On Windows, OpenMP is supported by Microsoft Visual Studio (MSVC) and MinGW compilers. If you use MSVC (the default for most Python distributions):

        - No extra installation is needed; OpenMP is included with MSVC.
        - The build system will automatically enable OpenMP if available.

        If you use MinGW, ensure you have a recent version with OpenMP support. You may need to add `-fopenmp` to your compiler flags if building manually.

Pre-installation: Test OpenMP Availability
------------------------------------------
You can check if OpenMP is available and working by compiling and running the following test program. The instructions differ by OS:

Create a file `omp_test.cpp` with:

.. code-block:: cpp

   #include <omp.h>
   #include <cstdio>

   int main() {
       int n = 0;
       int threads = 1;
   #ifdef _OPENMP
       #pragma omp parallel reduction(+:n)
       {
           n += 1;
           #pragma omp master
           threads = omp_get_num_threads();
       }
       if (threads > 1) {
           std::printf("✅ OpenMP available! Number of threads: %d\n", threads);
       } else {
           std::printf("⚠️  OpenMP available, but only 1 thread detected.\n");
       }
   #else
       std::printf("❌ OpenMP NOT available.\n");
   #endif
       return 0;
   }

Linux
~~~~~
GCC usually supports OpenMP out of the box. Compile and run with:

.. code-block:: bash

   g++ -fopenmp omp_test.cpp -o omp_test
   ./omp_test

macOS
~~~~~
Apple Clang does not support OpenMP by default. Use Homebrew's libomp and dynamically fetch the GCC paths as described in the macOS OpenMP Support section above. Then compile and run:

.. code-block:: bash

   clang++ -Xpreprocessor -fopenmp omp_test.cpp \
        -I${OMP_PREFIX}/include \
        -L${OMP_PREFIX}/lib -lomp \
        -o omp_test

    ./omp_test

Windows
~~~~~~~
If using Microsoft Visual Studio (MSVC):

1. Open the "Developer Command Prompt for VS".
2. Compile and run:
   .. code-block:: bash
   cl /openmp omp_test.cpp
   omp_test.exe

If using MinGW:

.. code-block:: bash

   g++ -fopenmp omp_test.cpp -o omp_test.exe
   ./omp_test.exe

No special environment variables are usually needed on Windows if the compiler supports OpenMP.

You should see a message indicating whether OpenMP is available and how many threads are detected.

Running the Test Suite (Recommended)
------------------------------------
After installation, it is highly recommended to run the test suite to verify your installation and check OpenMP support.


- If you installed smudgy via pip, the test suite is not included by default. To run the full tests, clone the repository:

.. code-block:: bash

    git clone https://github.com/youruser/smudgy.git

- Then run the tests with (this also applies to installations from source):

.. code-block:: bash

    pytest -rs

The tests will automatically check for OpenMP availability and skip parallelization tests if OpenMP is not enabled or detected. If you see tests being skipped due to missing OpenMP, revisit the OpenMP Support section to ensure your environment is set up correctly and the package was compiled with OpenMP support.
Running the tests is the best way to confirm that your installation is working as expected and that you are getting the performance benefits of parallelization.

If you have any issues, please consult the documentation or open an issue on GitHub.
