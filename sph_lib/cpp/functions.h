#pragma once
#include <vector>
#include <string>

void ngp_2d_cpp(
    const float* pos,
    const float* quantities,
    int N,
    int num_fields,
    const float* boxsizes,
    const int* gridnums,
    const bool* periodic,
    bool use_openmp,
    int omp_threads,
    float* fields,
    float* weights
);

void ngp_3d_cpp(
    const float* pos,
    const float* quantities,
    int N,
    int num_fields,
    const float* boxsizes,
    const int* gridnums,
    const bool* periodic,
    bool use_openmp,
    int omp_threads,
    float* fields,
    float* weights
);

void cic_2d_cpp(
    const float* pos,
    const float* quantities,
    int N,
    int num_fields,
    const float* boxsizes,
    const int* gridnums,
    const bool* periodic,
    bool use_openmp,
    int omp_threads,
    float* fields,
    float* weights
);

void cic_3d_cpp(
    const float* pos,
    const float* quantities,
    int N,
    int num_fields,
    const float* boxsizes,
    const int* gridnums,
    const bool* periodic,
    bool use_openmp,
    int omp_threads,
    float* fields,
    float* weights
);

void cic_2d_adaptive_cpp(
    const float* pos,
    const float* quantities,
    int N,
    int num_fields,
    const float* boxsizes,
    const int* gridnums,
    const bool* periodic,
    const float* pcellsizesHalf,
    bool use_openmp,
    int omp_threads,
    float* fields,
    float* weights
);

void cic_3d_adaptive_cpp(
    const float* pos,
    const float* quantities,
    int N,
    int num_fields,
    const float* boxsizes,
    const int* gridnums,
    const bool* periodic,
    const float* pcellsizesHalf,
    bool use_openmp,
    int omp_threads,
    float* fields,
    float* weights
);

void tsc_2d_cpp(
    const float* pos,
    const float* quantities,
    int N,
    int num_fields,
    const float* boxsizes,
    const int* gridnums,
    const bool* periodic,
    bool use_openmp,
    int omp_threads,
    float* fields,
    float* weights
);

void tsc_3d_cpp(
    const float* pos,
    const float* quantities,
    int N,
    int num_fields,
    const float* boxsizes,
    const int* gridnums,
    const bool* periodic,
    bool use_openmp,
    int omp_threads,
    float* fields,
    float* weights
);

void tsc_2d_adaptive_cpp(
    const float* pos,
    const float* quantities,
    int N,
    int num_fields,
    const float* boxsizes,
    const int* gridnums,
    const bool* periodic,
    const float* pcellsizesHalf,
    bool use_openmp,
    int omp_threads,
    float* fields,
    float* weights
);

void tsc_3d_adaptive_cpp(
    const float* pos,
    const float* quantities,
    int N,
    int num_fields,
    const float* boxsizes,
    const int* gridnums,
    const bool* periodic,
    const float* pcellsizesHalf,
    bool use_openmp,
    int omp_threads,
    float* fields,
    float* weights
);

void isotropic_kernel_deposition_2d_cpp(
    const float* pos,
    const float* quantities,
    const float* hsm,
    int N,
    int num_fields,
    const float* boxsizes,
    const int* gridnums,
    const bool* periodic,
    const std::string& kernel_name,
    const std::string& integration_method,
    bool use_openmp,
    int omp_threads,
    float* fields,
    float* weights
);

void isotropic_kernel_deposition_3d_cpp(
    const float* pos,
    const float* quantities,
    const float* hsm,
    int N,
    int num_fields,
    const float* boxsizes,
    const int* gridnums,
    const bool* periodic,
    const std::string& kernel_name,
    const std::string& integration_method,
    bool use_openmp,
    int omp_threads,
    float* fields,
    float* weights
);

void anisotropic_kernel_deposition_2d_cpp(
    const float* pos,
    const float* quantities,
    const float* hmat_eigvecs,
    const float* hmat_eigvals,
    int N,
    int num_fields,
    const float* boxsizes,
    const int* gridnums,
    const bool* periodic,
    const std::string& kernel_name,
    const std::string& integration_method,
    bool use_openmp,
    int omp_threads,
    float* fields,
    float* weights
);

void anisotropic_kernel_deposition_3d_cpp(
    const float* pos,
    const float* quantities,
    const float* hmat_eigvecs,
    const float* hmat_eigvals,
    int N,
    int num_fields,
    const float* boxsizes,
    const int* gridnums,
    const bool* periodic,
    const std::string& kernel_name,
    const std::string& integration_method,
    bool use_openmp,
    int omp_threads,
    float* fields,
    float* weights
);