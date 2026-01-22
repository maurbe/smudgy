Install libomp via Homebrew:

brew update
brew install libomp

Export the flags so clang/LLVM can see the headers and dylib (add to your shell profile if you want them persistent):

export OMP_PREFIX="$(brew --prefix libomp)"
export CPPFLAGS="-Xclang -fopenmp -I${OMP_PREFIX}/include ${CPPFLAGS}"
export LDFLAGS="-L${OMP_PREFIX}/lib -lomp ${LDFLAGS}"
export DYLD_LIBRARY_PATH="${OMP_PREFIX}/lib:${DYLD_LIBRARY_PATH}"

Quick sanity check: compile and run a tiny OpenMP program.

Create omp_test.cpp:
#include <omp.h>
#include <cstdio>

int main() {
    int n = 0;
    #pragma omp parallel reduction(+:n)
    n += 1;
    std::printf("threads=%d\n", n);
    return 0;
}

Compile and run:
clang -Xclang -fopenmp omp_test.cpp -L${OMP_PREFIX}/lib -lomp -I${OMP_PREFIX}/include -o omp_test
./omp_test

If OpenMP is linked correctly you should see threads= followed by the number of threads spawned (typically >1 unless you export OMP_NUM_THREADS=1).

Finally, rebuild the sph_lib extension (e.g., pip install -e .) so it picks up the OpenMP-enabled toolchain. When testing, you can toggle the new use_openmp flag to confirm that disabling it forces serial execution even though libomp is installed.