#pragma once
#include <vector>
#include <string>

void ngp_2d_cpp(
    const float* positions,
    const float* quantities,
    int num_particles,
    int num_fields,
    const float* boxsizes,
    const int* gridnums,
    bool periodic,
    bool use_openmp,
    int omp_threads,
    float* fields,
    float* weights
);

void ngp_3d_cpp(
    const float* positions,
    const float* quantities,
    int num_particles,
    int num_fields,
    const float* boxsizes,
    const int* gridnums,
    bool periodic,
    bool use_openmp,
    int omp_threads,
    float* fields,
    float* weights
);

void cic_2d_cpp(
    const float* positions,
    const float* quantities,
    int num_particles,
    int num_fields,
    const float* boxsizes,
    const int* gridnums,
    bool periodic,
    bool use_openmp,
    int omp_threads,
    float* fields,
    float* weights
);

void cic_3d_cpp(
    const float* positions,
    const float* quantities,
    int num_particles,
    int num_fields,
    const float* boxsizes,
    const int* gridnums,
    bool periodic,
    bool use_openmp,
    int omp_threads,
    float* fields,
    float* weights
);

void cic_2d_adaptive_cpp(
    const float* positions,
    const float* quantities,
    const float* smoothing_lengths,
    int num_particles,
    int num_fields,
    const float* boxsizes,
    const int* gridnums,
    bool periodic,
    bool use_openmp,
    int omp_threads,
    float* fields,
    float* weights
);

void cic_3d_adaptive_cpp(
    const float* positions,
    const float* quantities,
    const float* smoothing_lengths,
    int num_particles,
    int num_fields,
    const float* boxsizes,
    const int* gridnums,
    bool periodic,
    bool use_openmp,
    int omp_threads,
    float* fields,
    float* weights
);

void tsc_2d_cpp(
    const float* positions,
    const float* quantities,
    int num_particles,
    int num_fields,
    const float* boxsizes,
    const int* gridnums,
    bool periodic,
    bool use_openmp,
    int omp_threads,
    float* fields,
    float* weights
);

void tsc_3d_cpp(
    const float* positions,
    const float* quantities,
    int num_particles,
    int num_fields,
    const float* boxsizes,
    const int* gridnums,
    bool periodic,
    bool use_openmp,
    int omp_threads,
    float* fields,
    float* weights
);

void tsc_2d_adaptive_cpp(
    const float* positions,
    const float* quantities,
    const float* smoothing_lengths,
    int num_particles,
    int num_fields,
    const float* boxsizes,
    const int* gridnums,
    bool periodic,
    bool use_openmp,
    int omp_threads,
    float* fields,
    float* weights
);

void tsc_3d_adaptive_cpp(
    const float* positions,
    const float* quantities,
    const float* smoothing_lengths,
    int num_particles,
    int num_fields,
    const float* boxsizes,
    const int* gridnums,
    bool periodic,
    bool use_openmp,
    int omp_threads,
    float* fields,
    float* weights
);

void separable_kernel_deposition_2d_cpp(
    const float* positions,
    const float* quantities,
    const float* smoothing_lengths,
    int num_particles,
    int num_fields,
    const float* boxsizes,
    const int* gridnums,
    bool periodic,
    const std::string& kernel_name,
    const std::string& integration_method,
    bool use_openmp,
    int omp_threads,
    float* fields,
    float* weights
);

void isotropic_kernel_deposition_2d_cpp(
    const float* positions,
    const float* quantities,
    const float* smoothing_lengths,
    int num_particles,
    int num_fields,
    const float* boxsizes,
    const int* gridnums,
    bool periodic,
    const std::string& kernel_name,
    const std::string& integration_method,
    int min_kernel_evaluations,
    bool use_openmp,
    int omp_threads,
    float* fields,
    float* weights
);

void isotropic_kernel_deposition_3d_cpp(
    const float* positions,
    const float* quantities,
    const float* smoothing_lengths,
    int num_particles,
    int num_fields,
    const float* boxsizes,
    const int* gridnums,
    bool periodic,
    const std::string& kernel_name,
    const std::string& integration_method,
    int min_kernel_evaluations,
    bool use_openmp,
    int omp_threads,
    float* fields,
    float* weights
);

void anisotropic_kernel_deposition_2d_cpp(
    const float* positions,
    const float* quantities,
    const float* smoothing_tensor_eigvecs,
    const float* smoothing_tensor_eigvals,
    int num_particles,
    int num_fields,
    const float* boxsizes,
    const int* gridnums,
    bool periodic,
    const std::string& kernel_name,
    const std::string& integration_method,
    int min_kernel_evaluations,
    bool use_openmp,
    int omp_threads,
    float* fields,
    float* weights
);

void anisotropic_kernel_deposition_3d_cpp(
    const float* positions,
    const float* quantities,
    const float* smoothing_tensor_eigvecs,
    const float* smoothing_tensor_eigvals,
    int num_particles,
    int num_fields,
    const float* boxsizes,
    const int* gridnums,
    bool periodic,
    const std::string& kernel_name,
    const std::string& integration_method,
    int min_kernel_evaluations,
    bool use_openmp,
    int omp_threads,
    float* fields,
    float* weights
);