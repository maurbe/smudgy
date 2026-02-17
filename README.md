
# sph_lib Installation Guide


You can install `sph_lib` in several ways. OpenMP support is optional but recommended for parallel performance. See below for details on installation and enabling OpenMP.


## 1. Install from PyPI (recommended)

Once released on PyPI, simply run:

```sh
pip install sph_lib
```

If a pre-built wheel is available for your platform, no compiler or OpenMP setup is needed. Otherwise, see the [OpenMP Support](#openmp-support-parallelization) section if you want parallelization.


## 2. Install from a Pre-built Wheel

If you have a wheel file (e.g., from GitHub Releases):

```sh
pip install path/to/sph_lib-<version>-<platform>.whl
```

Or directly from a URL:

```sh
pip install https://github.com/youruser/yourrepo/releases/download/vX.Y/sph_lib-<version>-<platform>.whl
```


## 3. Build from Source

Clone the repository and install:

```sh
git clone https://github.com/youruser/sph_lib.git
cd sph_lib
pip install .
```

If you want to enable OpenMP parallelization, see the [OpenMP Support](#openmp-support-parallelization) section before running pip install .


### OpenMP Support (Parallelization)

To enable OpenMP parallelization, you must have OpenMP installed on your system **before** building the package. See instructions for your operating system below. If OpenMP is not found, the package will still work, but parallelization will be disabled.


#### Linux

Most Linux distributions provide OpenMP support out of the box with GCC. To ensure you have it:

```sh
sudo apt-get update  # Debian/Ubuntu
sudo apt-get install build-essential libgomp1
```
GCC automatically links OpenMP when you build the package. No extra flags are usually needed.


#### macOS

Apple's Clang does not support OpenMP by default. Install OpenMP via Homebrew:

```sh
brew update
brew install libomp
```

Then, export the following flags so your compiler and linker can find OpenMP (libomp):

```sh
export OMP_PREFIX="$(brew --prefix libomp)"
export CPPFLAGS="-Xclang -fopenmp -I${OMP_PREFIX}/include ${CPPFLAGS}"
export LDFLAGS="-L${OMP_PREFIX}/lib -lomp ${LDFLAGS}"
export DYLD_LIBRARY_PATH="${OMP_PREFIX}/lib:${DYLD_LIBRARY_PATH}"
```

Now install the package as above. You can reuse these flags for any OpenMP C++ compilation (see below).


#### Windows

On Windows, OpenMP is supported by Microsoft Visual Studio (MSVC) and MinGW compilers. If you use MSVC (the default for most Python distributions):

- No extra installation is needed; OpenMP is included with MSVC.
- The build system will automatically enable OpenMP if available.

If you use MinGW, ensure you have a recent version with OpenMP support. You may need to add `-fopenmp` to your compiler flags if building manually.


## Pre-installation: Test OpenMP Availability

You can check if OpenMP is available and working by compiling and running the following test program. The instructions differ by OS:

Create a file `omp_test.cpp` with:

```cpp
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
```

### Linux

GCC usually supports OpenMP out of the box. Compile and run with:

```sh
g++ -fopenmp omp_test.cpp -o omp_test
./omp_test
```

### macOS

Apple Clang does not support OpenMP by default. Use Homebrew's libomp and set the flags as described in the [macOS OpenMP Support](#macos) section above. Then compile and run:

```sh
clang -Xclang -fopenmp omp_test.cpp -L${OMP_PREFIX}/lib -lomp -I${OMP_PREFIX}/include -o omp_test
./omp_test
```

### Windows

If using Microsoft Visual Studio (MSVC):

1. Open the "Developer Command Prompt for VS".
2. Compile and run:
   ```
   cl /openmp omp_test.cpp
   omp_test.exe
   ```

If using MinGW:

```sh
g++ -fopenmp omp_test.cpp -o omp_test.exe
./omp_test.exe
```

No special environment variables are usually needed on Windows if the compiler supports OpenMP.

You should see a message indicating whether OpenMP is available and how many threads are detected.

---

If you have any issues, please consult the documentation or open an issue on GitHub.