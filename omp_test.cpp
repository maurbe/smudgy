#include <omp.h>
#include <cstdio>

int main() {
    int n = 0;
    #pragma omp parallel reduction(+:n)
    n += 1;
    std::printf("threads=%d\n", n);
    return 0;
}