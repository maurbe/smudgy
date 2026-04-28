#include <iostream>
#include <cassert>

#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>
#include <array>
#include <string>
#include <stdexcept>
#include <optional>
#if defined(_OPENMP)
#include <omp.h>
#endif
#include "_kernels.h"
#include "_integration.h"
#include "_functions.h"

// =============================================================================
// Utilities
// =============================================================================

inline int apply_pbc(int idx, int gridnum, bool periodic) {
    if (!periodic) {
        return idx;
    }
    int r = idx % gridnum;
    return (r < 0) ? r + gridnum : r;
}

inline bool is_outside_domain(int index, int gridnum) {
    return (index < 0 || index >= gridnum);
}

inline std::optional<int> cell_index_from_pos(float pos, float boxsize, int gridnum, bool periodic) {
    const float eps = 1e-6f; // small tolerance for floating-point errors

    if (periodic) {
        pos = std::fmod(pos, boxsize);
        if (pos < 0.0f) pos += boxsize;
        // Safety clamp to [0, boxsize)
        if (pos >= boxsize) pos -= boxsize;  // handles pos ≈ boxsize due to rounding
    } 
    else {
        if (pos < 0.0f || pos >= boxsize) return std::nullopt;
    }

    // Compute index
    int idx = static_cast<int>(pos * static_cast<float>(gridnum) / boxsize);

    // Clamp for safety
    if (idx < 0) idx = 0;
    if (idx >= gridnum) idx = gridnum - 1;

    // Non-periodic check
    if (!periodic && is_outside_domain(idx, gridnum)) return std::nullopt;

    return idx;
}

inline float wrap_distance_if_periodic(float delta, float boxsize, bool periodic) {
    if (!periodic) {
        return delta;
    }
    const float half_box = 0.5f * boxsize;
    if (delta > half_box) delta -= boxsize;
    if (delta < -half_box) delta += boxsize;
    return delta;
}

// =============================================================================
// OpenMP helpers
// =============================================================================

inline bool allow_openmp(bool requested) {
#if defined(_OPENMP)
    return requested;
#else
    (void)requested;
    return false;
#endif
}

inline int resolve_openmp_threads(bool parallel, int omp_threads) {
    return (parallel && omp_threads > 0) ? omp_threads : 0;
}

template <typename Func>
inline void for_each_particle(int N, bool parallel, int threads, const Func& func) {
#if defined(_OPENMP)
    if (parallel) {
        if (threads > 0) {
#pragma omp parallel for schedule(static) num_threads(threads)
            for (int n = 0; n < N; ++n) {
                func(n);
            }
        } else {
#pragma omp parallel for schedule(static)
            for (int n = 0; n < N; ++n) {
                func(n);
            }
        }
        return;
    }
#else
    (void)threads;
#endif
    for (int n = 0; n < N; ++n) {
        func(n);
    }
}

inline void accumulate(float* array, int idx, float value, bool parallel) {
#if defined(_OPENMP)
    if (parallel) {
#pragma omp atomic update
        array[idx] += value;
        return;
    }
#endif
    array[idx] += value;
}

inline void accumulate_fields(float* fields,
                              int base_idx,
                              const float* particle_values,
                              int num_fields,
                              float scale,
                              bool parallel) {
    for (int f = 0; f < num_fields; ++f) {
        accumulate(fields, base_idx + f, scale * particle_values[f], parallel);
    }
}

inline void accumulate_weight(float* weights, int idx, float value, bool parallel) {
    accumulate(weights, idx, value, parallel);
}

// =============================================================================
// pure C++ deposition functions
// =============================================================================

void ngp_2d_cpp(
    const float* positions,     // (num_particles, 2)
    const float* quantities,    // (num_particles, num_fields)
    int num_particles,                      
    int num_fields,
    const float* boxsizes,      // (2,)
    const int* gridnums,        // (2,)
    bool periodic,
    bool use_openmp,            
    int omp_threads,
    float* fields,              // (gridnum_x, gridnum_y, num_fields)
    float* weights              // (gridnum_x, gridnum_y)
) {
    // resolve openMP settings
    const bool parallel = allow_openmp(use_openmp);
    const int threads = resolve_openmp_threads(parallel, omp_threads);

    // extract grid parameters and precompute inverse cell sizes
    const int gridnum_x = gridnums[0];
    const int gridnum_y = gridnums[1];
    const float inv_cell_size_x = static_cast<float>(gridnum_x) / boxsizes[0];
    const float inv_cell_size_y = static_cast<float>(gridnum_y) / boxsizes[1];

    // prepare output arrays
    const int field_stride_x = gridnum_y * num_fields;
    const int field_stride_y = num_fields;
    std::memset(fields,  0, sizeof(float) * gridnum_x * gridnum_y * num_fields);
    std::memset(weights, 0, sizeof(float) * gridnum_x * gridnum_y);

    // iterate over particles and accumulate to grid
    for_each_particle(num_particles, parallel, threads, [&](int n) {

        // compute cell index of mother cell
        int ix = static_cast<int>(positions[2 * n + 0] * inv_cell_size_x);
        int iy = static_cast<int>(positions[2 * n + 1] * inv_cell_size_y);

        // early-out if particle lies outside the grid domain
        if (is_outside_domain(ix, gridnum_x)) return;
        if (is_outside_domain(iy, gridnum_y)) return;

        // deposit to grid
        const int base_idx   = ix * field_stride_x + iy * field_stride_y;
        const int weight_idx = ix * gridnum_y + iy;
        const float* particle = quantities + n * num_fields;
        accumulate_fields(fields, base_idx, particle, num_fields, 1.0f, parallel);
        accumulate_weight(weights, weight_idx, 1.0f, parallel);
    });
}


void ngp_3d_cpp(
    const float* positions,     // (num_particles, 3)
    const float* quantities,    // (num_particles, num_fields)
    int num_particles,
    int num_fields,
    const float* boxsizes,      // (3,)
    const int* gridnums,        // (3,)
    bool periodic,
    bool use_openmp,
    int omp_threads,
    float* fields,              // (gridnum_x, gridnum_y, gridnum_z, num_fields)
    float* weights              // (gridnum_x, gridnum_y, gridnum_z)
) {
    // resolve openMP settings
    const bool parallel = allow_openmp(use_openmp);
    const int threads = resolve_openmp_threads(parallel, omp_threads);

    // extract grid parameters and precompute inverse cell sizes
    const int gridnum_x = gridnums[0];
    const int gridnum_y = gridnums[1];
    const int gridnum_z = gridnums[2];
    const float inv_cell_size_x = static_cast<float>(gridnum_x) / boxsizes[0];
    const float inv_cell_size_y = static_cast<float>(gridnum_y) / boxsizes[1];
    const float inv_cell_size_z = static_cast<float>(gridnum_z) / boxsizes[2];

    // prepare output arrays
    const int field_stride_x = gridnum_y * gridnum_z * num_fields;
    const int field_stride_y = gridnum_z * num_fields;
    const int field_stride_z = num_fields;
    std::memset(fields,  0, sizeof(float) * gridnum_x * gridnum_y * gridnum_z * num_fields);
    std::memset(weights, 0, sizeof(float) * gridnum_x * gridnum_y * gridnum_z);

    // iterate over particles and accumulate to grid
    for_each_particle(num_particles, parallel, threads, [&](int n) {

        // compute cell index of mother cell
        int ix = static_cast<int>(positions[3 * n + 0] * inv_cell_size_x);
        int iy = static_cast<int>(positions[3 * n + 1] * inv_cell_size_y);
        int iz = static_cast<int>(positions[3 * n + 2] * inv_cell_size_z);

        // early-out if particle lies outside the grid domain
        if (is_outside_domain(ix, gridnum_x)) return;
        if (is_outside_domain(iy, gridnum_y)) return;
        if (is_outside_domain(iz, gridnum_z)) return;

        // deposit to grid
        const int base_idx   = ix * field_stride_x + iy * field_stride_y + iz * field_stride_z;
        const int weight_idx = ix * gridnum_y * gridnum_z + iy * gridnum_z + iz;
        const float* particle = quantities + n * num_fields;
        accumulate_fields(fields, base_idx, particle, num_fields, 1.0f, parallel);
        accumulate_weight(weights, weight_idx, 1.0f, parallel);
    });
}


// =============================================================================
// Separable kernel deposition (2D)
// =============================================================================

void separable_kernel_deposition_2d_cpp(
    const float* positions,             // (num_particles, 2)
    const float* quantities,            // (num_particles, num_fields)
    const float* smoothing_lengths,     // (num_particles, 2)
    const int num_particles,
    const int num_fields,
    const float* boxsizes,              // (2,)
    const int* gridnums,                // (2,)
    const bool periodic,
    const std::string& kernel_name,
    const std::string& integration_method,
    const bool use_openmp,
    const int omp_threads,
    float* fields,                      // (gridnum_x, gridnum_y, num_fields)
    float* weights                      // (gridnum_x, gridnum_y)
) {
    // resolve openMP settings
    const bool parallel = allow_openmp(use_openmp);
    const int threads = resolve_openmp_threads(parallel, omp_threads);

    // set up the kernel and cache integral samples
    auto kernel = create_separable_kernel(kernel_name, 2);
    SeparableKernel* kernel_ptr = kernel.get();
    const float kernel_support = kernel->support();

    // extract boxsize parameters
    const float boxsize_x = boxsizes[0];
    const float boxsize_y = boxsizes[1];

    // extract grid parameters
    const int gridnum_x = gridnums[0];
    const int gridnum_y = gridnums[1];

    // compute cell sizes and related parameters
    const float cellSize_x = boxsize_x / static_cast<float>(gridnum_x);
    const float cellSize_y = boxsize_y / static_cast<float>(gridnum_y);
    
    // precompute inverse cell size to save divisions
    const float cellSize_x_inv = 1.0f / cellSize_x;
    const float cellSize_y_inv = 1.0f / cellSize_y;

    // precompute max kernel support in cell units for clipping ( if periodic)
    float max_support_x = 0.5f * boxsize_x / kernel_support;
    float max_support_y = 0.5f * boxsize_y / kernel_support;
    float max_support = std::min({max_support_x, max_support_y});

    // compute strides for fields/weights and set up output arrays
    const int stride_x = gridnum_y * num_fields;
    const int stride_y = num_fields;
    const int weight_stride_x = gridnum_y;
    const int weight_stride_y = 1;

    // setup memory for output arrays
    std::memset(fields,  0, sizeof(float) * gridnum_x * gridnum_y * num_fields);
    std::memset(weights, 0, sizeof(float) * gridnum_x * gridnum_y);

    // perform for loop over particles
    for_each_particle(num_particles, parallel, threads, [&](int n) {

        // gather relevant particle data
        float x_phys = positions[2 * n + 0];
        float y_phys = positions[2 * n + 1];

        // resolve smoothing lengths in physical units
        float hsm_x_phys = smoothing_lengths[2 * n + 0];
        float hsm_y_phys = smoothing_lengths[2 * n + 1];
        if (periodic) {
            float hsm_max = std::max(hsm_x_phys, hsm_y_phys);
            if (hsm_max > max_support) {
                float scale = max_support / hsm_max;
                hsm_x_phys *= scale;
                hsm_y_phys *= scale;
            }
        }

        // convert to cell units
        float x_cell = x_phys * cellSize_x_inv;
        float y_cell = y_phys * cellSize_y_inv;

        // convert to cell units
        float hsm_x_cell = hsm_x_phys * cellSize_x_inv;
        float hsm_y_cell = hsm_y_phys * cellSize_y_inv;

        // compute kernel support in cell units
        float support_x_cell = kernel_support * hsm_x_cell;
        float support_y_cell = kernel_support * hsm_y_cell;

        // compute inclusive index bounds within the kernel support
        int i_min = static_cast<int>(std::floor(x_cell - support_x_cell));
        int j_min = static_cast<int>(std::floor(y_cell - support_y_cell));
        int i_max = static_cast<int>(std::ceil (x_cell + support_x_cell)) - 1;
        int j_max = static_cast<int>(std::ceil (y_cell + support_y_cell)) - 1;
        const float* particle = quantities + n * num_fields;

        /* for separable kernels, the integrals are typically known analytically 
        so we can compute them exactly for each cell without needing to cache samples on a grid */
        for (int i = i_min; i <= i_max; ++i) {
            for (int j = j_min; j <= j_max; ++j) {

                // check periodicity and apply PBC if needed, otherwise early-out if outside domain
                int ii = apply_pbc(i, gridnum_x, periodic);
                int jj = apply_pbc(j, gridnum_y, periodic);
                if (is_outside_domain(ii, gridnum_x) || is_outside_domain(jj, gridnum_y)) continue;

                // compute cell bounds in "q" units ( (box_edge - particle_pos) / hsm )
                float qx_left  = (i - x_cell) / hsm_x_cell;
                float qy_left  = (j - y_cell) / hsm_y_cell;
                float qx_right = ((i + 1) - x_cell) / hsm_x_cell;
                float qy_right = ((j + 1) - y_cell) / hsm_y_cell;

                // compute total integral over the current cell using limits in q-space
                std::vector<float> q_bounds = {qx_left, qx_right, qy_left, qy_right};
                float integral = kernel_ptr->sigma() * kernel_ptr->evaluate_integral(q_bounds);
            
                // deposit into current cell
                int base_idx = ii * stride_x + jj * stride_y;
                int weight_idx = ii * weight_stride_x + jj * weight_stride_y;
                accumulate_fields(fields, base_idx, particle, num_fields, integral, parallel);
                accumulate_weight(weights, weight_idx, integral, parallel);
            }
        }
    });
}

// =============================================================================
// Separable kernel deposition (3D)
// =============================================================================

void separable_kernel_deposition_3d_cpp(
    const float* positions,                 // (num_particles, 3)
    const float* quantities,                // (num_particles, num_fields)
    const float* smoothing_lengths,         // (num_particles, 3)
    const int num_particles,
    const int num_fields,
    const float* boxsizes,                  // (3,)
    const int* gridnums,                    // (3,)
    const bool periodic,
    const std::string& kernel_name,
    const std::string& integration_method,
    const bool use_openmp,
    const int omp_threads,
    float* fields,                          // (gridnum_x, gridnum_y, gridnum_z, num_fields)
    float* weights                          // (gridnum_x, gridnum_y, gridnum_z)
) {
    // resolve openMP settings
    const bool parallel = allow_openmp(use_openmp);
    const int threads = resolve_openmp_threads(parallel, omp_threads);

    // set up the kernel and cache integral samples
    auto kernel = create_separable_kernel(kernel_name, 3);
    SeparableKernel* kernel_ptr = kernel.get();
    const float kernel_support = kernel->support();

    // extract boxsize parameters
    const float boxsize_x = boxsizes[0];
    const float boxsize_y = boxsizes[1];
    const float boxsize_z = boxsizes[2];

    // extract grid parameters
    const int gridnum_x = gridnums[0];
    const int gridnum_y = gridnums[1];
    const int gridnum_z = gridnums[2];

    // compute cell sizes and related parameters
    const float cellSize_x = boxsize_x / static_cast<float>(gridnum_x);
    const float cellSize_y = boxsize_y / static_cast<float>(gridnum_y);
    const float cellSize_z = boxsize_z / static_cast<float>(gridnum_z);
    
    // precompute inverse cell size to save divisions
    const float cellSize_x_inv = 1.0f / cellSize_x;
    const float cellSize_y_inv = 1.0f / cellSize_y;
    const float cellSize_z_inv = 1.0f / cellSize_z;

    // precompute max kernel support in cell units for clipping ( if periodic)
    float max_support_x = 0.5f * boxsize_x / kernel_support;
    float max_support_y = 0.5f * boxsize_y / kernel_support;
    float max_support_z = 0.5f * boxsize_z / kernel_support;
    float max_support = std::min({max_support_x, max_support_y, max_support_z});

    // compute strides for fields/weights and set up output arrays
    const int stride_x = gridnum_y * gridnum_z * num_fields;
    const int stride_y = gridnum_z * num_fields;
    const int stride_z = num_fields;
    const int weight_stride_x = gridnum_y * gridnum_z;
    const int weight_stride_y = gridnum_z;
    const int weight_stride_z = 1;

    // setup memory for output arrays
    std::memset(fields,  0, sizeof(float) * gridnum_x * gridnum_y * gridnum_z * num_fields);
    std::memset(weights, 0, sizeof(float) * gridnum_x * gridnum_y * gridnum_z);

    // perform for loop over particles
    for_each_particle(num_particles, parallel, threads, [&](int n) {

        // gather relevant particle data
        float x_phys = positions[3 * n + 0];
        float y_phys = positions[3 * n + 1];
        float z_phys = positions[3 * n + 2];

        // resolve smoothing lengths in physical units
        float hsm_x_phys = smoothing_lengths[3 * n + 0];
        float hsm_y_phys = smoothing_lengths[3 * n + 1];
        float hsm_z_phys = smoothing_lengths[3 * n + 2];

        if (periodic) {
            float hsm_max = std::max({hsm_x_phys, hsm_y_phys, hsm_z_phys});
            if (hsm_max > max_support) {
                float scale = max_support / hsm_max;
                hsm_x_phys *= scale;
                hsm_y_phys *= scale;
                hsm_z_phys *= scale;
            }
        }

        // convert to cell units
        float x_cell = x_phys * cellSize_x_inv;
        float y_cell = y_phys * cellSize_y_inv;
        float z_cell = z_phys * cellSize_z_inv;

        // convert to cell units
        float hsm_x_cell = hsm_x_phys * cellSize_x_inv;
        float hsm_y_cell = hsm_y_phys * cellSize_y_inv;
        float hsm_z_cell = hsm_z_phys * cellSize_z_inv;

        // compute kernel support in cell units
        float support_x_cell = kernel_support * hsm_x_cell;
        float support_y_cell = kernel_support * hsm_y_cell;
        float support_z_cell = kernel_support * hsm_z_cell;

        // compute inclusive index bounds within the kernel support
        int i_min = static_cast<int>(std::floor(x_cell - support_x_cell));
        int j_min = static_cast<int>(std::floor(y_cell - support_y_cell));
        int k_min = static_cast<int>(std::floor(z_cell - support_z_cell));
        int i_max = static_cast<int>(std::ceil (x_cell + support_x_cell));
        int j_max = static_cast<int>(std::ceil (y_cell + support_y_cell));
        int k_max = static_cast<int>(std::ceil (z_cell + support_z_cell));
        const float* particle = quantities + n * num_fields;

        // for separable kernels, typically the integrals are known analytically,
        // so we can compute them exactly for each cell without needing to cache samples on a grid
        for (int i = i_min; i <= i_max; ++i) {
            for (int j = j_min; j <= j_max; ++j) {
                for (int k = k_min; k <= k_max; ++k) {

                    // check periodicity and apply PBC if needed, otherwise early-out if outside domain
                    int ii = apply_pbc(i, gridnum_x, periodic);
                    int jj = apply_pbc(j, gridnum_y, periodic);
                    int kk = apply_pbc(k, gridnum_z, periodic);
                    if (is_outside_domain(ii, gridnum_x) || is_outside_domain(jj, gridnum_y) || is_outside_domain(kk, gridnum_z)) continue;

                    // compute cell bounds in "q" space ( (box_edge - particle_pos) / hsm )
                    float qx_left  = (i - x_cell) / hsm_x_cell;
                    float qy_left  = (j - y_cell) / hsm_y_cell;
                    float qz_left  = (k - z_cell) / hsm_z_cell;
                    float qx_right = ((i + 1) - x_cell) / hsm_x_cell;
                    float qy_right = ((j + 1) - y_cell) / hsm_y_cell;
                    float qz_right = ((k + 1) - z_cell) / hsm_z_cell;
                    
                    // compute total integral over the current cell using q-space bounds
                    std::vector<float> q_bounds = {qx_left, qx_right, qy_left, qy_right, qz_left, qz_right};
                    float integral = kernel_ptr->sigma() * kernel_ptr->evaluate_integral(q_bounds);

                    // deposit to grid
                    int base_idx = ii * stride_x + jj * stride_y + kk * stride_z;
                    int weight_idx = ii * weight_stride_x + jj * weight_stride_y + kk * weight_stride_z;
                    accumulate_fields(fields, base_idx, particle, num_fields, integral, parallel);
                    accumulate_weight(weights, weight_idx, integral, parallel);
                }
            }
        }
    });
}

// =============================================================================
// Isotropic kernel deposition (2D)
// =============================================================================

void isotropic_kernel_deposition_2d_cpp(
    const float* positions,             // (num_particles, 2)
    const float* quantities,            // (num_particles, num_fields)
    const float* smoothing_lengths,     // (num_particles)
    int num_particles,
    int num_fields,
    const float* boxsizes,              // (2,)
    const int* gridnums,                // (2,)
    bool periodic,
    const std::string& kernel_name,
    const std::string& integration_method,
    const int num_kernel_evaluations_per_axis,
    const float eta_crit,
    bool use_openmp,
    int omp_threads,
    float* fields,                      // (gridnum_x, gridnum_y, num_fields)
    float* weights                      // (gridnum_x, gridnum_y)
) {
    // resolve openMP settings
    const bool parallel = allow_openmp(use_openmp);
    const int threads = resolve_openmp_threads(parallel, omp_threads);
    
    // set up the kernel and cache integral samples
    auto kernel = create_spherical_kernel(kernel_name, 2);
    SphericalKernel* kernel_ptr = kernel.get();
    const float kernel_support = kernel->support();
    const auto kernel_samples = build_kernel_sample_grid(*kernel, num_kernel_evaluations_per_axis);
    const int num_samples = kernel_samples.count;

    // extract boxsize parameters
    const float boxsize_x = boxsizes[0];
    const float boxsize_y = boxsizes[1];

    // extract grid parameters
    const int gridnum_x = gridnums[0];
    const int gridnum_y = gridnums[1];

    // compute cell sizes and related parameters
    const float cellSize_x = boxsize_x / static_cast<float>(gridnum_x);
    const float cellSize_y = boxsize_y / static_cast<float>(gridnum_y);

    // precompute inverse cell size to save divisions
    const float cellSize_x_inv = 1.0f / cellSize_x;
    const float cellSize_y_inv = 1.0f / cellSize_y;

    // precompute max kernel support in cell units for clipping ( if periodic)
    float max_support_x = 0.5f * boxsize_x / kernel_support;
    float max_support_y = 0.5f * boxsize_y / kernel_support;
    float max_support = std::min({max_support_x, max_support_y});

    // compute strides for fields
    const int stride_x = gridnum_y * num_fields;
    const int stride_y = num_fields;

    // compute strides for weights
    const int weight_stride_x = gridnum_y;
    const int weight_stride_y = 1;

    // setup memory for output arrays
    std::memset(fields,  0, sizeof(float) * gridnum_x * gridnum_y * num_fields);
    std::memset(weights, 0, sizeof(float) * gridnum_x * gridnum_y);

    // perform for loop over particles
    for_each_particle(num_particles, parallel, threads, [&](int n) {

        // gather relevant particle data
        float x_phys = positions[2 * n + 0];
        float y_phys = positions[2 * n + 1];

        // resolve smoothing length (clip max = boxsize/2) and kernel prefactor
        float hsm_phys = smoothing_lengths[n];
        if (periodic) {
            if (hsm_phys > max_support) hsm_phys = max_support;
        }
        float kernel_prefactor = kernel_ptr->sigma() / (hsm_phys * hsm_phys);

        // convert to cell units
        float x_cell = x_phys * cellSize_x_inv;
        float y_cell = y_phys * cellSize_y_inv;

        // compute kernel support in physical and cell units
        float support_phys = kernel_support * hsm_phys;
        float support_x_cell = support_phys * cellSize_x_inv;
        float support_y_cell = support_phys * cellSize_y_inv;

        // compute inclusive index bounds within the kernel support
        int i_min = static_cast<int>(std::floor(x_cell - support_x_cell));
        int j_min = static_cast<int>(std::floor(y_cell - support_y_cell));
        int i_max = static_cast<int>(std::ceil(x_cell + support_x_cell));
        int j_max = static_cast<int>(std::ceil(y_cell + support_y_cell));
        const float* particle = quantities + n * num_fields;

        // compute total number of cells in the kernel support
        int num_cells_x = i_max - i_min;
        int num_cells_y = j_max - j_min;
        int total_cells = num_cells_x * num_cells_y;

        // compute eta
        float eta = std::min({hsm_phys / cellSize_x, hsm_phys / cellSize_y});

        // if the number of cells is small, use the cached kernel samples to evaluate the kernel at each sample point
        if (eta < eta_crit) {
            
            // -> iteration happens over the kernel sample points
            for (int s = 0; s < num_samples; ++s) {
                
                // kernel sample positions mapped to physical space
                float x_sample = x_phys + kernel_samples.coords[2 * s + 0] * hsm_phys;
                float y_sample = y_phys + kernel_samples.coords[2 * s + 1] * hsm_phys;
                
                // given the geometry, determine the cell into which sample falls
                auto ix = cell_index_from_pos(x_sample, boxsize_x, gridnum_x, periodic);
                auto iy = cell_index_from_pos(y_sample, boxsize_y, gridnum_y, periodic);
                if (!ix || !iy) continue;

                // gather the kernel sample integral (fraction)
                float integral = kernel_samples.integrals[s];
                if (integral == 0.0f) continue;

                // deposit to grid
                int base_idx = (*ix) * stride_x + (*iy) * stride_y;
                int weight_idx = (*ix) * weight_stride_x + (*iy) * weight_stride_y;
                accumulate_fields(fields, base_idx, particle, num_fields, integral, parallel);
                accumulate_weight(weights, weight_idx, integral, parallel);
            }
        }
        // if the number of cells is large, iterate over affected cells and compute kernel integral over cell domain
        else {
            float total_weight = 0.0f;

            std::vector<int> base_indices;
            std::vector<int> weight_indices;
            std::vector<float> integrals;

            base_indices.reserve(total_cells);
            weight_indices.reserve(total_cells);
            integrals.reserve(total_cells);

            // FIRST PASS: compute integrals only
            for (int a = i_min; a <= i_max; ++a) {
                int an = apply_pbc(a, gridnum_x, periodic);
                if (is_outside_domain(an, gridnum_x)) continue;

                for (int b = j_min; b <= j_max; ++b) {
                    int bn = apply_pbc(b, gridnum_y, periodic);
                    if (is_outside_domain(bn, gridnum_y)) continue;

                    auto eval = [&](float ox, float oy) {
                        float dx = (x_cell - (a + ox)) * cellSize_x;
                        float dy = (y_cell - (b + oy)) * cellSize_y;

                        dx = wrap_distance_if_periodic(dx, boxsize_x, periodic);
                        dy = wrap_distance_if_periodic(dy, boxsize_y, periodic);

                        float r = std::sqrt(dx * dx + dy * dy);
                        float q = r / hsm_phys;
                        return kernel_prefactor * kernel_ptr->evaluate(q);
                    };

                    float integral = integrate_cell_2d(integration_method, eval);
                    integral *= cellSize_x * cellSize_y;
                    total_weight += integral;

                    // if cell outside support -> integral = 0, so we skip deposition, which is good
                    // the branch above takes care of the case where kernel is contained within a single cell (-> integral = 1.0)

                    int base_idx   = an * stride_x + bn * stride_y;
                    int weight_idx = an * weight_stride_x + bn * weight_stride_y;

                    base_indices.push_back(base_idx);
                    weight_indices.push_back(weight_idx);
                    integrals.push_back(integral);
                }
            }

            // normalization factor
            float correction = 1.0f / total_weight;

            // SECOND PASS: deposit once (corrected)
            const size_t count = integrals.size();
            for (size_t i = 0; i < count; ++i) {
                float w = integrals[i] * correction;

                accumulate_fields(fields, base_indices[i], particle, num_fields, w, parallel);
                accumulate_weight(weights, weight_indices[i], w, parallel);
            }
        }
    });
}


// =============================================================================
// Isotropic kernel deposition (3D)
// =============================================================================

void isotropic_kernel_deposition_3d_cpp(
    const float* positions,         // (num_particles, 3)
    const float* quantities,        // (num_particles, num_fields)
    const float* smoothing_lengths, // (num_particles)
    int num_particles,
    int num_fields,
    const float* boxsizes,          // (3,)
    const int* gridnums,            // (3,)
    bool periodic,
    const std::string& kernel_name,
    const std::string& integration_method,
    const int num_kernel_evaluations_per_axis,
    const float eta_crit,
    bool use_openmp,
    int omp_threads,
    float* fields,                  // (gridnum_x, gridnum_y, gridnum_z, num_fields)
    float* weights                  // (gridnum_x, gridnum_y, gridnum_z)
) {
    // resolve openMP settings
    const bool parallel = allow_openmp(use_openmp);
    const int threads = resolve_openmp_threads(parallel, omp_threads);

    // set up the kernel and cache integral samples
    auto kernel = create_spherical_kernel(kernel_name, 3);
    SphericalKernel* kernel_ptr = kernel.get();
    const auto kernel_samples = build_kernel_sample_grid(*kernel, num_kernel_evaluations_per_axis);
    const int num_samples = kernel_samples.count;
    const float kernel_support = kernel->support();

    // extract boxsize parameters
    const float boxsize_x = boxsizes[0];
    const float boxsize_y = boxsizes[1];
    const float boxsize_z = boxsizes[2];

    // extract grid parameters
    const int gridnum_x = gridnums[0];
    const int gridnum_y = gridnums[1];
    const int gridnum_z = gridnums[2];

    // compute cell sizes and related parameters
    const float cellSize_x = boxsize_x / static_cast<float>(gridnum_x);
    const float cellSize_y = boxsize_y / static_cast<float>(gridnum_y);
    const float cellSize_z = boxsize_z / static_cast<float>(gridnum_z);

    // precompute inverse cell size to save divisions
    const float cellSize_x_inv = 1.0f / cellSize_x;
    const float cellSize_y_inv = 1.0f / cellSize_y;
    const float cellSize_z_inv = 1.0f / cellSize_z;

    // precompute max kernel support in cell units for clipping ( if periodic)
    float max_support_x = 0.5f * boxsize_x / kernel_support;
    float max_support_y = 0.5f * boxsize_y / kernel_support;
    float max_support_z = 0.5f * boxsize_z / kernel_support;
    float max_support = std::min({max_support_x, max_support_y, max_support_z});

    // compute strides for fields
    const int stride_x = gridnum_y * gridnum_z * num_fields;
    const int stride_y = gridnum_z * num_fields;
    const int stride_z = num_fields;

    // compute strides for weights
    const int weight_stride_x = gridnum_y * gridnum_z;
    const int weight_stride_y = gridnum_z;
    const int weight_stride_z = 1;

    // setup memory for output arrays
    std::memset(fields,  0, sizeof(float) * gridnum_x * gridnum_y * gridnum_z * num_fields);
    std::memset(weights, 0, sizeof(float) * gridnum_x * gridnum_y * gridnum_z);

    // perform for loop over particles
    for_each_particle(num_particles, parallel, threads, [&](int n) {

        // convert particle position to cell units
        float x_phys = positions[3 * n + 0];
        float y_phys = positions[3 * n + 1];
        float z_phys = positions[3 * n + 2];

        // resolve smoothing length (clip max = boxsize/2) and kernel prefactor
        float hsm_phys = smoothing_lengths[n];
        if (periodic) {
            if (hsm_phys > max_support) hsm_phys = max_support;
        }
        float kernel_prefactor = kernel_ptr->sigma() / (hsm_phys * hsm_phys * hsm_phys);

        float x_cell = x_phys * cellSize_x_inv;
        float y_cell = y_phys * cellSize_y_inv;
        float z_cell = z_phys * cellSize_z_inv;

        // compute kernel support in physical and cell units
        float support_phys = kernel_support * hsm_phys;
        float support_x_cell = support_phys / cellSize_x;
        float support_y_cell = support_phys / cellSize_y;
        float support_z_cell = support_phys / cellSize_z;
        //float kernel_normalization = kernel_ptr->normalization(detH);

        // compute inclusive index bounds within the kernel support
        int i_min = static_cast<int>(std::floor(x_cell - support_x_cell));
        int j_min = static_cast<int>(std::floor(y_cell - support_y_cell));
        int k_min = static_cast<int>(std::floor(z_cell - support_z_cell));
        int i_max = static_cast<int>(std::ceil(x_cell + support_x_cell));
        int j_max = static_cast<int>(std::ceil(y_cell + support_y_cell));
        int k_max = static_cast<int>(std::ceil(z_cell + support_z_cell));
        const float* particle = quantities + n * num_fields;

        // compute total number of cells in the kernel support
        int num_cells_x = i_max - i_min;
        int num_cells_y = j_max - j_min;
        int num_cells_z = k_max - k_min;
        int total_cells = num_cells_x * num_cells_y * num_cells_z;

        // compute eta
        float eta = std::min({hsm_phys / cellSize_x, hsm_phys / cellSize_y, hsm_phys / cellSize_z});

        // follow anti-aliasing strategy
        if (eta < eta_crit) {
            
            for (int s = 0; s < num_samples; ++s) {

                // kernel sample positions and mapping to physical space
                float x_sample = x_phys + kernel_samples.coords[3 * s + 0] * hsm_phys;
                float y_sample = y_phys + kernel_samples.coords[3 * s + 1] * hsm_phys;
                float z_sample = z_phys + kernel_samples.coords[3 * s + 2] * hsm_phys;

                auto ix = cell_index_from_pos(x_sample, boxsize_x, gridnum_x, periodic);
                auto iy = cell_index_from_pos(y_sample, boxsize_y, gridnum_y, periodic);
                auto iz = cell_index_from_pos(z_sample, boxsize_z, gridnum_z, periodic);
                if (!ix || !iy || !iz) continue;

                // gather the kernel sample integral (fraction)
                float integral = kernel_samples.integrals[s];

                // deposit to grid
                int base_idx = (*ix) * stride_x + (*iy) * stride_y + (*iz) * stride_z;
                int weight_idx = (*ix) * weight_stride_x + (*iy) * weight_stride_y + (*iz);
                accumulate_fields(fields, base_idx, particle, num_fields, integral, parallel);
                accumulate_weight(weights, weight_idx, integral, parallel);
            }
        }
        // if the number of cells is large, iterate over affected cells and compute kernel integral over cell domain
        else {
            float total_weight = 0.0f;

            std::vector<int> base_indices;
            std::vector<int> weight_indices;
            std::vector<float> integrals;

            base_indices.reserve(total_cells);
            weight_indices.reserve(total_cells);
            integrals.reserve(total_cells);

            // FIRST PASS: compute integrals only
            for (int a = i_min; a <= i_max; ++a) {
                int an = apply_pbc(a, gridnum_x, periodic);
                if (is_outside_domain(an, gridnum_x)) continue;

                for (int b = j_min; b <= j_max; ++b) {
                    int bn = apply_pbc(b, gridnum_y, periodic);
                    if (is_outside_domain(bn, gridnum_y)) continue;

                    for (int c = k_min; c <= k_max; ++c) {
                        int cn = apply_pbc(c, gridnum_z, periodic);
                        if (is_outside_domain(cn, gridnum_z)) continue;

                        auto eval = [&](float ox, float oy, float oz) {
                            float dx = (x_cell - (a + ox)) * cellSize_x;
                            float dy = (y_cell - (b + oy)) * cellSize_y;
                            float dz = (z_cell - (c + oz)) * cellSize_z;

                            dx = wrap_distance_if_periodic(dx, boxsize_x, periodic);
                            dy = wrap_distance_if_periodic(dy, boxsize_y, periodic);
                            dz = wrap_distance_if_periodic(dz, boxsize_z, periodic);

                            float r = std::sqrt(dx * dx + dy * dy + dz * dz);
                            float q = r / hsm_phys;
                            return kernel_prefactor * kernel_ptr->evaluate(q);
                        };

                        float integral = integrate_cell_3d(integration_method, eval);
                        integral *= cellSize_x * cellSize_y * cellSize_z;
                        total_weight += integral;

                        int base_idx   = an * stride_x + bn * stride_y + cn * stride_z;
                        int weight_idx = an * weight_stride_x + bn * weight_stride_y + cn;

                        base_indices.push_back(base_idx);
                        weight_indices.push_back(weight_idx);
                        integrals.push_back(integral);
                    }
                }
            }

            // compute correction factor to ensure normalization
            float correction = 1.0f / total_weight;

            // SECOND PASS: deposit once (corrected)
            const size_t count = integrals.size();
            float corrected_total_weight = 0.0f;
            for (size_t i = 0; i < count; ++i) {
                float w = integrals[i] * correction;

                accumulate_fields(fields, base_indices[i], particle, num_fields, w, parallel);
                accumulate_weight(weights, weight_indices[i], w, parallel);
                corrected_total_weight += w;
            }
        }
    });
}


// =============================================================================
// Anisotropic kernel deposition (2D)
// =============================================================================

void anisotropic_kernel_deposition_2d_cpp(
    const float* positions,             // (num_particles, 2)
    const float* quantities,            // (num_particles, num_fields)
    const float* hmat_eigvecs,          // (num_particles, 4) - stored as [v00, v10, v01, v11] for each particle
    const float* hmat_eigvals,          // (num_particles, 2) - stored as [lambda0, lambda1] for each particle
    int num_particles,
    int num_fields,
    const float* boxsizes,              // (2,)
    const int* gridnums,                // (2,)
    bool periodic,
    const std::string& kernel_name,
    const std::string& integration_method,
    const int num_kernel_evaluations_per_axis,
    const float eta_crit,
    bool use_openmp,
    int omp_threads,
    float* fields,                      // (gridnum_x, gridnum_y, num_fields)   
    float* weights                      // (gridnum_x, gridnum_y)
) {
    // resolve openMP settings
    const bool parallel = allow_openmp(use_openmp);
    const int threads = resolve_openmp_threads(parallel, omp_threads);

    // set up the kernel and cache integral samples
    auto kernel = create_spherical_kernel(kernel_name, 2);
    SphericalKernel* kernel_ptr = kernel.get();
    const float kernel_support = kernel->support();
    const auto kernel_samples = build_kernel_sample_grid(*kernel, num_kernel_evaluations_per_axis);
    const int num_samples = kernel_samples.count;

    // extract boxsize parameters
    const float boxsize_x = boxsizes[0];
    const float boxsize_y = boxsizes[1];

    // extract grid parameters
    const int gridnum_x = gridnums[0];
    const int gridnum_y = gridnums[1];

    // compute cell sizes and related parameters
    const float cellSize_x = boxsize_x / static_cast<float>(gridnum_x);
    const float cellSize_y = boxsize_y / static_cast<float>(gridnum_y);
    
    // precompute inverse cell size to save divisions
    const float cellSize_x_inv = 1.0f / cellSize_x;
    const float cellSize_y_inv = 1.0f / cellSize_y;

    // precompute max kernel support in cell units for clipping ( if periodic)
    float max_support_x = 0.5f * boxsize_x / kernel_support;
    float max_support_y = 0.5f * boxsize_y / kernel_support;
    float max_support = std::min({max_support_x, max_support_y});

    // compute strides for fields
    const int stride_x = gridnum_y * num_fields;
    const int stride_y = num_fields;

    // compute strides for weights
    const int weight_stride_x = gridnum_y;
    const int weight_stride_y = 1;

    // setup memory for output arrays
    std::memset(fields,  0, sizeof(float) * gridnum_x * gridnum_y * num_fields);
    std::memset(weights, 0, sizeof(float) * gridnum_x * gridnum_y);
    
    // perform for loop over particles
    for_each_particle(num_particles, parallel, threads, [&](int n) {
        
        // gather relevant particle data
        float x_phys = positions[2 * n + 0];
        float y_phys = positions[2 * n + 1];

        // resolve the smoothing tensors / eigvals (not eigvecs, since they are normalized)
        const float* evecs = &hmat_eigvecs[n * 4];
        float evals_phys[2] = { hmat_eigvals[n * 2 + 0], hmat_eigvals[n * 2 + 1] };

        if (periodic) {
            // check whether the max eval exceeds the max support and scale if applicable
            float max_eval = std::max({evals_phys[0], evals_phys[1]});
            if (max_eval > max_support) {
                float scale = max_support / max_eval;
                evals_phys[0] *= scale;
                evals_phys[1] *= scale;
            }
        }
        float evals_cell[2] = { evals_phys[0] / cellSize_x, evals_phys[1] / cellSize_y };
        float detH = evals_phys[0] * evals_phys[1];
        float kernel_prefactor = kernel_ptr->sigma() / detH;
        
        // convert particle position to cell units
        float x_cell = x_phys * cellSize_x_inv;
        float y_cell = y_phys * cellSize_y_inv;

        // figure out the extent of the kernel 
        float support_x_cell = kernel_support * std::sqrt(
            std::pow(evecs[0] * evals_cell[0], 2) +
            std::pow(evecs[2] * evals_cell[1], 2)
        );
        float support_y_cell = kernel_support * std::sqrt(
            std::pow(evecs[1] * evals_cell[0], 2) +
            std::pow(evecs[3] * evals_cell[1], 2)
        );

        // compute inclusive index bounds within the kernel support
        int i_min = static_cast<int>(std::floor(x_cell - support_x_cell));
        int i_max = static_cast<int>(std::ceil(x_cell + support_x_cell)) - 1;
        int j_min = static_cast<int>(std::floor(y_cell - support_y_cell));
        int j_max = static_cast<int>(std::ceil(y_cell + support_y_cell)) - 1;
        const float* particle = quantities + n * num_fields;

        // compute total number of cells in the kernel support
        int num_cells_x = i_max - i_min + 1;
        int num_cells_y = j_max - j_min + 1;
        int total_cells = num_cells_x * num_cells_y;

        // compute eta / eta_crit to determine which integration to use
        float eta = std::min({evals_phys[0] / cellSize_x, evals_phys[1] / cellSize_y});

        // if eta < eta_crit, numerical quadrature is not enough, need to switch to kernel integral samples
        if (eta < eta_crit) {

            for (int s = 0; s < num_samples; ++s) {

                // kernel sample positions and mapping to physical space
                float x = x_phys + evecs[0] * (evals_phys[0] * kernel_samples.coords[2 * s + 0])
                                      + evecs[2] * (evals_phys[1] * kernel_samples.coords[2 * s + 1]);
                float y = y_phys + evecs[1] * (evals_phys[0] * kernel_samples.coords[2 * s + 0])
                                      + evecs[3] * (evals_phys[1] * kernel_samples.coords[2 * s + 1]);

                auto ix = cell_index_from_pos(x, boxsize_x, gridnum_x, periodic);
                auto iy = cell_index_from_pos(y, boxsize_y, gridnum_y, periodic);
                if (!ix || !iy) continue;

                // gather the kernel sample integral (fraction)
                float integral = kernel_samples.integrals[s];

                // deposit to grid
                int base_idx = (*ix) * stride_x + (*iy) * stride_y;
                int weight_idx = (*ix) * weight_stride_x + (*iy) * weight_stride_y;
                accumulate_fields(fields, base_idx, particle, num_fields, integral, parallel);
                accumulate_weight(weights, weight_idx, integral, parallel);
            }
        } 
        // if the number of cells is large, iterate over affected cells and compute kernel integral over cell domain
        else {
            float total_weight = 0.0f;

            std::vector<int> base_indices;
            std::vector<int> weight_indices;
            std::vector<float> integrals;

            base_indices.reserve(total_cells);
            weight_indices.reserve(total_cells);
            integrals.reserve(total_cells);

            // FIRST PASS: compute integrals only
            for (int a = i_min; a <= i_max; ++a) {
                int an = apply_pbc(a, gridnum_x, periodic);
                if (is_outside_domain(an, gridnum_x)) continue;

                for (int b = j_min; b <= j_max; ++b) {
                    int bn = apply_pbc(b, gridnum_y, periodic);
                    if (is_outside_domain(bn, gridnum_y)) continue;

                    auto eval = [&](float ox, float oy) {
                        float dx = (x_cell - (a + ox)) * cellSize_x;
                        float dy = (y_cell - (b + oy)) * cellSize_y;

                        dx = wrap_distance_if_periodic(dx, boxsize_x, periodic);
                        dy = wrap_distance_if_periodic(dy, boxsize_y, periodic);

                        float xi1 = (evecs[0] * dx + evecs[1] * dy) / evals_phys[0];
                        float xi2 = (evecs[2] * dx + evecs[3] * dy) / evals_phys[1];
                        float q = std::sqrt(xi1 * xi1 + xi2 * xi2);

                        return kernel_prefactor * kernel_ptr->evaluate(q);
                    };

                    float integral = integrate_cell_2d(integration_method, eval);
                    integral *= cellSize_x * cellSize_y;
                    total_weight += integral;

                    int base_idx   = an * stride_x + bn * stride_y;
                    int weight_idx = an * weight_stride_x + bn;

                    base_indices.push_back(base_idx);
                    weight_indices.push_back(weight_idx);
                    integrals.push_back(integral);
                }
            }

            // compute correction factor to ensure normalization
            float correction = 1.0f / total_weight;

            // SECOND PASS: deposit once (corrected)
            const size_t count = integrals.size();
            for (size_t i = 0; i < count; ++i) {
                float w = integrals[i] * correction;

                accumulate_fields(fields, base_indices[i], particle, num_fields, w, parallel);
                accumulate_weight(weights, weight_indices[i], w, parallel);
            }
        }
    });
}


// =============================================================================
// Anisotropic kernel deposition (3D)
// =============================================================================

void anisotropic_kernel_deposition_3d_cpp(
    const float* positions,           // (num_particles, 3)
    const float* quantities,    // (num_particles, num_fields)
    const float* hmat_eigvecs,  // (num_particles, 9) - column-major eigenvectors
    const float* hmat_eigvals,  // (num_particles, 3) - eigenvalues per particle
    int num_particles,
    int num_fields,
    const float* boxsizes,      // (3,)
    const int* gridnums,        // (3,)
    bool periodic,              // (3,)
    const std::string& kernel_name,
    const std::string& integration_method,
    const int num_kernel_evaluations_per_axis,
    const float eta_crit,
    bool use_openmp,
    int omp_threads,
    float* fields,              // (gridnum_x, gridnum_y, gridnum_z, num_fields)
    float* weights              // (gridnum_x, gridnum_y, gridnum_z)
) {
    // resolve openMP settings
    const bool parallel = allow_openmp(use_openmp);
    const int threads = resolve_openmp_threads(parallel, omp_threads);
    
    // set up the kernel and cache integral samples
    auto kernel = create_spherical_kernel(kernel_name, 3);
    SphericalKernel* kernel_ptr = kernel.get();
    const float kernel_support = kernel->support();
    const auto kernel_samples = build_kernel_sample_grid(*kernel, num_kernel_evaluations_per_axis);
    const int num_samples = kernel_samples.count;

    // extract boxsize parameters
    const float boxsize_x = boxsizes[0];
    const float boxsize_y = boxsizes[1];
    const float boxsize_z = boxsizes[2];

    // extract grid parameters
    const int gridnum_x = gridnums[0];
    const int gridnum_y = gridnums[1];
    const int gridnum_z = gridnums[2];

    // compute cell sizes and related parameters
    const float cellSize_x = boxsize_x / static_cast<float>(gridnum_x);
    const float cellSize_y = boxsize_y / static_cast<float>(gridnum_y);
    const float cellSize_z = boxsize_z / static_cast<float>(gridnum_z);

    // precompute inverse cell size to save divisions
    const float cellSize_x_inv = 1.0f / cellSize_x;
    const float cellSize_y_inv = 1.0f / cellSize_y;
    const float cellSize_z_inv = 1.0f / cellSize_z;

    // precompute max kernel support in cell units for clipping ( if periodic)
    float max_support_x = 0.5f * boxsize_x / kernel_support;
    float max_support_y = 0.5f * boxsize_y / kernel_support;
    float max_support_z = 0.5f * boxsize_z / kernel_support;
    float max_support = std::min({max_support_x, max_support_y, max_support_z});

    // compute strides for fields
    const int stride_x = gridnum_y * gridnum_z * num_fields;
    const int stride_y = gridnum_z * num_fields;
    const int stride_z = num_fields;
    
    // compute strides for weights
    const int weight_stride_x = gridnum_y * gridnum_z;
    const int weight_stride_y = gridnum_z;
    const int weight_stride_z = 1;

    // setup memory for output arrays
    std::memset(fields,  0, sizeof(float) * gridnum_x * gridnum_y * gridnum_z * num_fields);
    std::memset(weights, 0, sizeof(float) * gridnum_x * gridnum_y * gridnum_z);

    // perform for loop over particles
    for_each_particle(num_particles, parallel, threads, [&](int n) {
        
        // gather relevant values for the current particle
        float x_phys = positions[3 * n + 0];
        float y_phys = positions[3 * n + 1];
        float z_phys = positions[3 * n + 2];
        
        const float* evecs = &hmat_eigvecs[n * 9];
        float evals_phys[3] = { hmat_eigvals[n * 3 + 0], hmat_eigvals[n * 3 + 1], hmat_eigvals[n * 3 + 2] };

        if (periodic) {
            // check whether the max eval exceeds the max support and scale if applicable
            float max_eval = std::max({evals_phys[0], evals_phys[1], evals_phys[2]});
            if (max_eval > max_support) {
                float scale = max_support / max_eval;
                evals_phys[0] *= scale;
                evals_phys[1] *= scale;
                evals_phys[2] *= scale;
            }
        }
        float evals_cell[3] = { evals_phys[0] / cellSize_x, evals_phys[1] / cellSize_y, evals_phys[2] / cellSize_z };
        float detH = evals_phys[0] * evals_phys[1] * evals_phys[2];
        float kernel_prefactor = kernel_ptr->sigma() / detH;

        // convert particle position to cell units
        float x_cell = x_phys * cellSize_x_inv;
        float y_cell = y_phys * cellSize_y_inv;
        float z_cell = z_phys * cellSize_z_inv;

        // estimate kernel support along each axis in cell units
        float support_x_cell = kernel_support * std::sqrt(
            std::pow(evecs[0] * evals_cell[0], 2) +
            std::pow(evecs[3] * evals_cell[1], 2) +
            std::pow(evecs[6] * evals_cell[2], 2)         
        );
        float support_y_cell = kernel_support * std::sqrt(
            std::pow(evecs[1] * evals_cell[0], 2) +
            std::pow(evecs[4] * evals_cell[1], 2) +
            std::pow(evecs[7] * evals_cell[2], 2)
        );
        float support_z_cell = kernel_support * std::sqrt(
            std::pow(evecs[2] * evals_cell[0], 2) +
            std::pow(evecs[5] * evals_cell[1], 2) +
            std::pow(evecs[8] * evals_cell[2], 2)
        );

        // compute inclusive index bounds within the kernel support
        int i_min = static_cast<int>(std::floor(x_cell - support_x_cell));
        int i_max = static_cast<int>(std::ceil(x_cell + support_x_cell));
        int j_min = static_cast<int>(std::floor(y_cell - support_y_cell));
        int j_max = static_cast<int>(std::ceil(y_cell + support_y_cell));
        int k_min = static_cast<int>(std::floor(z_cell - support_z_cell));
        int k_max = static_cast<int>(std::ceil(z_cell + support_z_cell));
        const float* particle = quantities + n * num_fields;

        // compute total number of cells in the kernel support
        int num_cells_x = i_max - i_min;
        int num_cells_y = j_max - j_min;
        int num_cells_z = k_max - k_min;
        int total_cells = num_cells_x * num_cells_y * num_cells_z;

        // compute eta
        float eta = std::min({evals_phys[0] / cellSize_x, evals_phys[1] / cellSize_y, evals_phys[2] / cellSize_z});

        // small-support path: use cached kernel samples
        if (eta < eta_crit) {
            
            for (int s = 0; s < num_samples; ++s) {

                // kernel sample positions and mapping to physical space
                float x_phys = x_phys + evecs[0] * (evals_phys[0] * kernel_samples.coords[3 * s + 0])
                                      + evecs[3] * (evals_phys[1] * kernel_samples.coords[3 * s + 1])
                                      + evecs[6] * (evals_phys[2] * kernel_samples.coords[3 * s + 2]);
                float y_phys = y_phys + evecs[1] * (evals_phys[0] * kernel_samples.coords[3 * s + 0])
                                      + evecs[4] * (evals_phys[1] * kernel_samples.coords[3 * s + 1])
                                      + evecs[7] * (evals_phys[2] * kernel_samples.coords[3 * s + 2]);
                float z_phys = z_phys + evecs[2] * (evals_phys[0] * kernel_samples.coords[3 * s + 0])
                                      + evecs[5] * (evals_phys[1] * kernel_samples.coords[3 * s + 1])
                                      + evecs[8] * (evals_phys[2] * kernel_samples.coords[3 * s + 2]);

                auto ix = cell_index_from_pos(x_phys, boxsize_x, gridnum_x, periodic);
                auto iy = cell_index_from_pos(y_phys, boxsize_y, gridnum_y, periodic);
                auto iz = cell_index_from_pos(z_phys, boxsize_z, gridnum_z, periodic);
                if (!ix || !iy || !iz) continue;

                // gather the kernel sample integral (fraction)
                float integral = kernel_samples.integrals[s];

                // deposit to grid
                int base_idx = (*ix) * stride_x + (*iy) * stride_y + (*iz) * stride_z;
                int weight_idx = (*ix) * weight_stride_x + (*iy) * weight_stride_y + (*iz) * weight_stride_z;
                accumulate_fields(fields, base_idx, particle, num_fields, integral, parallel);
                accumulate_weight(weights, weight_idx, integral, parallel);
            }
        }
        // large-support path: integrate directly over grid cells
        else {
            float total_weight = 0.0f;

            std::vector<int> base_indices;
            std::vector<int> weight_indices;
            std::vector<float> integrals;

            base_indices.reserve(total_cells);
            weight_indices.reserve(total_cells);
            integrals.reserve(total_cells);

            // FIRST PASS: compute integrals only
            for (int a = i_min; a <= i_max; ++a) {
                int an = apply_pbc(a, gridnum_x, periodic);
                if (is_outside_domain(an, gridnum_x)) continue;

                for (int b = j_min; b <= j_max; ++b) {
                    int bn = apply_pbc(b, gridnum_y, periodic);
                    if (is_outside_domain(bn, gridnum_y)) continue;

                    for (int c = k_min; c <= k_max; ++c) {
                        int cn = apply_pbc(c, gridnum_z, periodic);
                        if (is_outside_domain(cn, gridnum_z)) continue;

                        auto eval = [&](float ox, float oy, float oz) {
                            float dx = (x_cell - (a + ox)) * cellSize_x;
                            float dy = (y_cell - (b + oy)) * cellSize_y;
                            float dz = (z_cell - (c + oz)) * cellSize_z;

                            dx = wrap_distance_if_periodic(dx, boxsize_x, periodic);
                            dy = wrap_distance_if_periodic(dy, boxsize_y, periodic);
                            dz = wrap_distance_if_periodic(dz, boxsize_z, periodic);

                            float xi1 = (evecs[0] * dx + evecs[1] * dy + evecs[2] * dz) / evals_phys[0];
                            float xi2 = (evecs[3] * dx + evecs[4] * dy + evecs[5] * dz) / evals_phys[1];
                            float xi3 = (evecs[6] * dx + evecs[7] * dy + evecs[8] * dz) / evals_phys[2];

                            float q = std::sqrt(xi1 * xi1 + xi2 * xi2 + xi3 * xi3);
                            return kernel_prefactor * kernel_ptr->evaluate(q);
                        };

                        float integral = integrate_cell_3d(integration_method, eval);
                        integral *= cellSize_x * cellSize_y * cellSize_z;
                        total_weight += integral;

                        int base_idx   = an * stride_x + bn * stride_y + cn * stride_z;
                        int weight_idx = an * weight_stride_x + bn * weight_stride_y + cn * weight_stride_z;

                        base_indices.push_back(base_idx);
                        weight_indices.push_back(weight_idx);
                        integrals.push_back(integral);
                    }
                }
            }
            
            // compute correction factor to ensure normalization
            float correction = 1.0f / total_weight;

            // SECOND PASS: deposit once (corrected)
            const size_t count = integrals.size();
            for (size_t i = 0; i < count; ++i) {
                float w = integrals[i] * correction;

                accumulate_fields(fields, base_indices[i], particle, num_fields, w, parallel);
                accumulate_weight(weights, weight_indices[i], w, parallel);
            }
        }
    });
}
