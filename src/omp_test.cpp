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