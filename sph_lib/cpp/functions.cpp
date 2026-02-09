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
#include "kernels.h"
#include "functions.h"

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
    // Periodic: wrap position into [0, boxsize) before indexing.
    if (periodic) {
        pos = std::fmod(pos, boxsize);
        if (pos < 0.0f) pos += boxsize;
    // Non-periodic: early-out if position lies outside the domain.
    } else if (pos < 0.0f || pos >= boxsize) {
        return std::nullopt;
    }

    // Convert position to cell index; for non-periodic, guard against edge round-off.
    int idx = static_cast<int>(pos * (static_cast<float>(gridnum) / boxsize));
    if (!periodic && (idx < 0 || idx >= gridnum)) {
        return std::nullopt;
    }
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

// -----------------------------------------------------------------------------
// OpenMP helpers
// -----------------------------------------------------------------------------

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
    const float* pos,           // (N, 2)
    const float* quantities,    // (N, num_fields)
    int N,                      
    int num_fields,
    const float* boxsizes,      // (2,)
    const int* gridnums,        // (2,)
    const bool* periodic,       // (2,)
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
    for_each_particle(N, parallel, threads, [&](int n) {

        // compute cell index of mother cell
        int ix = static_cast<int>(pos[2 * n + 0] * inv_cell_size_x);
        int iy = static_cast<int>(pos[2 * n + 1] * inv_cell_size_y);

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
    const float* pos,           // (N, 3)
    const float* quantities,    // (N, num_fields)
    int N,
    int num_fields,
    const float* boxsizes,      // (3,)
    const int* gridnums,        // (3,)
    const bool* periodic,       // (3,)
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
    for_each_particle(N, parallel, threads, [&](int n) {

        // compute cell index of mother cell
        int ix = static_cast<int>(pos[3 * n + 0] * inv_cell_size_x);
        int iy = static_cast<int>(pos[3 * n + 1] * inv_cell_size_y);
        int iz = static_cast<int>(pos[3 * n + 2] * inv_cell_size_z);

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


void cic_2d_cpp(
    const float* pos,           // (N, 2)
    const float* quantities,    // (N, num_fields)
    int N,
    int num_fields,
    const float* boxsizes,      // (2,)
    const int* gridnums,        // (2,)
    const bool* periodic,       // (2,)
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
    
    // extract periodicity flags
    const bool periodic_x = periodic[0];
    const bool periodic_y = periodic[1];

    // prepare output arrays
    const int field_stride_x = gridnum_y * num_fields;
    const int field_stride_y = num_fields;
    std::memset(fields,  0, sizeof(float) * gridnum_x * gridnum_y * num_fields);
    std::memset(weights, 0, sizeof(float) * gridnum_x * gridnum_y);

    // iterate over particles and accumulate to grid
    for_each_particle(N, parallel, threads, [&](int n) {

        // identify indices of the 4 surrounding grid cells
        float xpos = pos[2 * n + 0] * inv_cell_size_x;
        float ypos = pos[2 * n + 1] * inv_cell_size_y;
        int i0 = static_cast<int>(std::floor(xpos));
        int j0 = static_cast<int>(std::floor(ypos));
        int i1 = i0 + 1;
        int j1 = j0 + 1;

        // compute overlaps for the 4 surrounding grid cells
        float dx = xpos - i0;
        float dy = ypos - j0;
        float dx_ = 1.0f - dx;
        float dy_ = 1.0f - dy;

        // compute weights for the 4 surrounding grid cells based on overlaps
        float w00 = dx_ * dy_;
        float w10 = dx  * dy_;
        float w01 = dx_ * dy;
        float w11 = dx  * dy;
        const float* particle = quantities + n * num_fields;

        // define helper function for deposition
        auto deposit = [&](int i, int j, float w) {
            int ii = i;
            int jj = j;

            // wrap indices if periodic
            ii = apply_pbc(ii, gridnum_x, periodic_x);
            jj = apply_pbc(jj, gridnum_y, periodic_y);

            // early-out if cell lies outside the grid domain (only relevant for non-periodic case)
            if (is_outside_domain(ii, gridnum_x)) return;
            if (is_outside_domain(jj, gridnum_y)) return;

            // deposit to grid
            int base_idx   = ii * field_stride_x + jj * field_stride_y;
            int weight_idx = ii * gridnum_y + jj;
            accumulate_fields(fields, base_idx, particle, num_fields, w, parallel);
            accumulate_weight(weights, weight_idx, w, parallel);
        };

        // perform deposition to the 4 surrounding grid cells
        deposit(i0, j0, w00);
        deposit(i1, j0, w10);
        deposit(i0, j1, w01);
        deposit(i1, j1, w11);
    });
}


void cic_3d_cpp(
    const float* pos,           // (N, 3)
    const float* quantities,    // (N, num_fields)
    int N,
    int num_fields,
    const float* boxsizes,      // (3,)
    const int* gridnums,        // (3,)
    const bool* periodic,
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

    // extract periodicity flags
    const bool periodic_x = periodic[0];
    const bool periodic_y = periodic[1];
    const bool periodic_z = periodic[2];

    // prepare output arrays
    const int field_stride_x = gridnum_y * gridnum_z * num_fields;
    const int field_stride_y = gridnum_z * num_fields;
    const int field_stride_z = num_fields;
    std::memset(fields,  0, sizeof(float) * gridnum_x * gridnum_y * gridnum_z * num_fields);
    std::memset(weights, 0, sizeof(float) * gridnum_x * gridnum_y * gridnum_z);

    // iterate over particles and accumulate to grid
    for_each_particle(N, parallel, threads, [&](int n) {

        // identify indices of the 8 surrounding grid cells
        float xpos = pos[3 * n + 0] * inv_cell_size_x;
        float ypos = pos[3 * n + 1] * inv_cell_size_y;
        float zpos = pos[3 * n + 2] * inv_cell_size_z;
        int i0 = static_cast<int>(std::floor(xpos));
        int j0 = static_cast<int>(std::floor(ypos));
        int k0 = static_cast<int>(std::floor(zpos));
        int i1 = i0 + 1;
        int j1 = j0 + 1;
        int k1 = k0 + 1;

        // compute overlaps for the 8 surrounding grid cells
        float dx = xpos - i0;
        float dy = ypos - j0;
        float dz = zpos - k0;
        float dx_ = 1.0f - dx;
        float dy_ = 1.0f - dy;
        float dz_ = 1.0f - dz;

        // compute weights for the 8 surrounding grid cells based on overlaps
        float w000 = dx_ * dy_ * dz_;
        float w100 = dx  * dy_ * dz_;
        float w010 = dx_ * dy  * dz_;
        float w110 = dx  * dy  * dz_;
        float w001 = dx_ * dy_ * dz;
        float w101 = dx  * dy_ * dz;
        float w011 = dx_ * dy  * dz;
        float w111 = dx  * dy  * dz;
        const float* particle = quantities + n * num_fields;

        // define helper function for deposition
        auto deposit = [&](int i, int j, int k, float w) {
            int ii = i;
            int jj = j;
            int kk = k;

            // wrap indices if periodic
            ii = apply_pbc(ii, gridnum_x, periodic_x);
            jj = apply_pbc(jj, gridnum_y, periodic_y);
            kk = apply_pbc(kk, gridnum_z, periodic_z);

            // early-out if particle lies outside the grid domain (only relevant for non-periodic case)
            if (is_outside_domain(ii, gridnum_x)) return;
            if (is_outside_domain(jj, gridnum_y)) return;
            if (is_outside_domain(kk, gridnum_z)) return;

            // deposit to grid
            int base_idx   = ii * field_stride_x + jj * field_stride_y + kk * field_stride_z;
            int weight_idx = ii * gridnum_y * gridnum_z + jj * gridnum_z + kk;
            accumulate_fields(fields, base_idx, particle, num_fields, w, parallel);
            accumulate_weight(weights, weight_idx, w, parallel);
        };

        // perform deposition to the 8 surrounding grid cells
        deposit(i0, j0, k0, w000);
        deposit(i1, j0, k0, w100);
        deposit(i0, j1, k0, w010);
        deposit(i1, j1, k0, w110);
        deposit(i0, j0, k1, w001);
        deposit(i1, j0, k1, w101);
        deposit(i0, j1, k1, w011);
        deposit(i1, j1, k1, w111);
    });
}


void cic_2d_adaptive_cpp(
    const float* pos,           // (N, 2)
    const float* quantities,    // (N, num_fields)
    int N,
    int num_fields,
    const float* boxsizes,      // (2,)
    const int* gridnums,        // (2,)
    const bool* periodic,       // (2,)
    const float* pcellsizesHalf,// (N)
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
    
    // extract periodicity flags
    const bool periodic_x = periodic[0];
    const bool periodic_y = periodic[1];

    // prepare output arrays
    const int stride_x = gridnum_y * num_fields;
    const int stride_y = num_fields;
    const int weight_stride_x = gridnum_y;
    std::memset(fields,  0, sizeof(float) * gridnum_x * gridnum_y * num_fields);
    std::memset(weights, 0, sizeof(float) * gridnum_x * gridnum_y);

    // iterate over particles and accumulate to grid
    for_each_particle(N, parallel, threads, [&](int n) {

        // compute half cell sizes in grid units and particle volume
        float pcs_x = pcellsizesHalf[n] * inv_cell_size_x;
        float pcs_y = pcellsizesHalf[n] * inv_cell_size_y;
        float V = (2.0f * pcs_x) * (2.0f * pcs_y);

        // compute mother cell index and bounding box
        float xpos = pos[2 * n + 0] * inv_cell_size_x;
        float ypos = pos[2 * n + 1] * inv_cell_size_y;
        float c1 = xpos - pcs_x, c2 = xpos + pcs_x;
        float c3 = ypos - pcs_y, c4 = ypos + pcs_y;

        // compute inclusive index bounds that the particle overlaps with
        int i_min = static_cast<int>(std::round(xpos - pcs_x - 0.5f));
        int i_max = static_cast<int>(xpos + pcs_x);
        int j_min = static_cast<int>(std::round(ypos - pcs_y - 0.5f));
        int j_max = static_cast<int>(ypos + pcs_y);
        const float* particle = quantities + n * num_fields;

        // iterate over all grid cells that the particle overlaps with
        for (int i = i_min; i <= i_max; ++i) {
            int ii = i;

            // check periodicity and apply PBC if needed, otherwise early-out if outside domain
            ii = apply_pbc(ii, gridnum_x, periodic_x);
            if (is_outside_domain(ii, gridnum_x)) continue;

            // compute cell edge coordinates for current grid cell
            float e1 = static_cast<float>(i);
            float e2 = e1 + 1.0f;

            for (int j = j_min; j <= j_max; ++j) {
                int jj = j;

                // check periodicity and apply PBC if needed, otherwise early-out if outside domain
                jj = apply_pbc(jj, gridnum_y, periodic_y);
                if (is_outside_domain(jj, gridnum_y)) continue;

                // compute cell edge coordinates for current grid cell
                float e3 = static_cast<float>(j);
                float e4 = e3 + 1.0f;

                // compute intersection fraction of total area between particle and current grid cell
                float intersec_x = std::fmin(e2, c2) - std::fmax(e1, c1);
                float intersec_y = std::fmin(e4, c4) - std::fmax(e3, c3);
                float fraction = (intersec_x * intersec_y) / V;

                // skip if no overlap
                if (fraction <= 0.0f) continue;

                // deposit to grid
                int base_idx = ii * stride_x + jj * stride_y;
                int weight_idx = ii * weight_stride_x + jj;
                accumulate_fields(fields, base_idx, particle, num_fields, fraction, parallel);
                accumulate_weight(weights, weight_idx, fraction, parallel);
            }
        }
    });
}

void cic_3d_adaptive_cpp(
    const float* pos,         // (N,3)
    const float* quantities,  // (N,num_fields)
    int N,
    int num_fields,
    const float* boxsizes,
    const int* gridnums,
    const bool* periodic,
    const float* pcellsizesHalf, // (N)
    bool use_openmp,
    int omp_threads,
    float* fields,            // (gridnum_x, gridnum_y, gridnum_z, num_fields)
    float* weights            // (gridnum_x, gridnum_y, gridnum_z)
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
    
    // extract periodicity flags
    const bool periodic_x = periodic[0];
    const bool periodic_y = periodic[1];
    const bool periodic_z = periodic[2];

    // prepare output arrays
    const int stride_x = gridnum_y * gridnum_z * num_fields;
    const int stride_y = gridnum_z * num_fields;
    const int stride_z = num_fields;
    const int weight_stride_x = gridnum_y * gridnum_z;
    const int weight_stride_y = gridnum_z;
    const int weight_stride_z = 1;
    std::memset(fields,  0, sizeof(float) * gridnum_x * gridnum_y * gridnum_z * num_fields);
    std::memset(weights, 0, sizeof(float) * gridnum_x * gridnum_y * gridnum_z);

    // iterate over particles and accumulate to grid
    for_each_particle(N, parallel, threads, [&](int n) {

        // compute half cell sizes in grid units and particle volume
        float pcs_x = pcellsizesHalf[n] * inv_cell_size_x;
        float pcs_y = pcellsizesHalf[n] * inv_cell_size_y;
        float pcs_z = pcellsizesHalf[n] * inv_cell_size_z;
        float V = (2.0f * pcs_x) * (2.0f * pcs_y) * (2.0f * pcs_z);

        // compute mother cell index and bounding box
        float xpos = pos[3 * n + 0] * inv_cell_size_x;
        float ypos = pos[3 * n + 1] * inv_cell_size_y;
        float zpos = pos[3 * n + 2] * inv_cell_size_z;
        float c1 = xpos - pcs_x, c2 = xpos + pcs_x;
        float c3 = ypos - pcs_y, c4 = ypos + pcs_y;
        float c5 = zpos - pcs_z, c6 = zpos + pcs_z;

        // compute inclusive index bounds that the particle overlaps with
        int i_min = static_cast<int>(std::round(xpos - pcs_x - 0.5f));
        int i_max = static_cast<int>(xpos + pcs_x);
        int j_min = static_cast<int>(std::round(ypos - pcs_y - 0.5f));
        int j_max = static_cast<int>(ypos + pcs_y);
        int k_min = static_cast<int>(std::round(zpos - pcs_z - 0.5f));
        int k_max = static_cast<int>(zpos + pcs_z);
        const float* particle = quantities + n * num_fields;

        // iterate over all grid cells that the particle overlaps with
        for (int i = i_min; i <= i_max; ++i) {
            int ii = i;

            // check periodicity and apply PBC if needed, otherwise early-out if outside domain
            ii = apply_pbc(ii, gridnum_x, periodic_x);
            if (is_outside_domain(ii, gridnum_x)) continue;

            // compute cell edge coordinates for current grid cell
            float e1 = static_cast<float>(i);
            float e2 = e1 + 1.0f;

            for (int j = j_min; j <= j_max; ++j) {
                int jj = j;

                // check periodicity and apply PBC if needed, otherwise early-out if outside domain
                jj = apply_pbc(jj, gridnum_y, periodic_y);
                if (is_outside_domain(jj, gridnum_y)) continue;

                // compute cell edge coordinates for current grid cell
                float e3 = static_cast<float>(j);
                float e4 = e3 + 1.0f;

                for (int k = k_min; k <= k_max; ++k) {
                    int kk = k;

                    // check periodicity and apply PBC if needed, otherwise early-out if outside domain
                    kk = apply_pbc(kk, gridnum_z, periodic_z);
                    if (is_outside_domain(kk, gridnum_z)) continue;

                    // compute cell edge coordinates for current grid cell
                    float e5 = static_cast<float>(k);
                    float e6 = e5 + 1.0f;

                    // compute intersection fraction of total volume between particle and current grid cell
                    float intersec_x = std::fmin(e2, c2) - std::fmax(e1, c1);
                    float intersec_y = std::fmin(e4, c4) - std::fmax(e3, c3);
                    float intersec_z = std::fmin(e6, c6) - std::fmax(e5, c5);
                    float fraction = (intersec_x * intersec_y * intersec_z) / V;

                     // skip if no overlap
                    if (fraction <= 0.0f) continue;

                    // deposit to grid
                    int base_idx = ii * stride_x + jj * stride_y + kk * stride_z;
                    int weight_idx = ii * weight_stride_x + jj * weight_stride_y + kk * weight_stride_z;
                    accumulate_fields(fields, base_idx, particle, num_fields, fraction, parallel);
                    accumulate_weight(weights, weight_idx, fraction, parallel);
                }
            }
        }
    });
}


// TSC weight for a single offset distance
inline std::array<float, 3> tsc_weights(float d) {
    // weights for neighbor offsets -1,0,+1
    std::array<float, 3> w;
    w[0] = 0.5f * (1.5f - d) * (1.5f - d);
    w[1] = 0.75f - (d - 1.0f) * (d - 1.0f);
    w[2] = 0.5f * (d - 0.5f) * (d - 0.5f);
    return w;
}

// Triangular Shaped Cloud deposition in 2D
void tsc_2d_cpp(
    const float* pos,           // (N, 2)
    const float* quantities,    // (N, num_fields)
    int N,
    int num_fields,
    const float* boxsizes,      // (2,)
    const int* gridnums,        // (2,)
    const bool* periodic,       // (2,)
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
    
    // extract periodicity flags
    const bool periodic_x = periodic[0];
    const bool periodic_y = periodic[1];

    // prepare output arrays
    const int stride_x = gridnum_y * num_fields;
    const int stride_y = num_fields;
    std::memset(fields,  0, sizeof(float) * gridnum_x * gridnum_y * num_fields);
    std::memset(weights, 0, sizeof(float) * gridnum_x * gridnum_y);

    // Neighbor offsets
    const int offsets[3] = {-1, 0, 1};

    // iterate over particles and accumulate to grid
    for_each_particle(N, parallel, threads, [&](int n) {

        // compute normalized position in grid units
        float xpos = pos[2 * n + 0] * inv_cell_size_x;
        float ypos = pos[2 * n + 1] * inv_cell_size_y;
        int i_base = static_cast<int>(std::floor(xpos));
        int j_base = static_cast<int>(std::floor(ypos));

        // compute fractional distance within cell
        float dx = xpos - i_base;
        float dy = ypos - j_base;

        auto wx = tsc_weights(dx);
        auto wy = tsc_weights(dy);
        const float* particle = quantities + n * num_fields;

        // iterate over the 3x3 neighboring grid cells
        for (int i = 0; i < 3; ++i) {
            int ix = i_base + offsets[i];

            // check periodicity and apply PBC if needed, early-out if outside domain
            ix = apply_pbc(ix, gridnum_x, periodic_x);
            if (is_outside_domain(ix, gridnum_x)) continue;
            
            for (int j = 0; j < 3; ++j) {
                int iy = j_base + offsets[j];

                // check periodicity and apply PBC if needed, early-out if outside domain
                iy = apply_pbc(iy, gridnum_y, periodic_y);
                if (is_outside_domain(iy, gridnum_y)) continue;

                // compute combined weight for this neighbor cell
                float w = wx[i] * wy[j];

                // skip if weight is zero
                if (w == 0.0f) continue;

                // deposit to grid
                int base_idx   = ix * stride_x + iy * stride_y;
                int weight_idx = ix * gridnum_y + iy;
                accumulate_fields(fields, base_idx, particle, num_fields, w, parallel);
                accumulate_weight(weights, weight_idx, w, parallel);
            }
        }
    });
}

void tsc_3d_cpp(
    const float* pos,           // (N, 3)
    const float* quantities,    // (N, num_fields)
    int N,
    int num_fields,
    const float* boxsizes,      // (3,) 
    const int* gridnums,        // (3,)
    const bool* periodic,       // (3,)
    bool use_openmp,
    int omp_threads,
    float* fields,              // (gridnum_x, gridnum_y, gridnum_z, num_fields)
    float* weights              // (gridnum_x, gridnum_y, gridnum_z)
) {
    // resolve openMP settings
    const bool parallel = allow_openmp(use_openmp);
    const int threads = resolve_openmp_threads(parallel, omp_threads);

    const int gridnum_x = gridnums[0];
    const int gridnum_y = gridnums[1];
    const int gridnum_z = gridnums[2];
    const bool periodic_x = periodic[0];
    const bool periodic_y = periodic[1];
    const bool periodic_z = periodic[2];
    const float inv_cell_size_x = static_cast<float>(gridnum_x) / boxsizes[0];
    const float inv_cell_size_y = static_cast<float>(gridnum_y) / boxsizes[1];
    const float inv_cell_size_z = static_cast<float>(gridnum_z) / boxsizes[2];

    // Strides for C-contiguous layout (x, y, z, f)
    const int stride_x = gridnum_y * gridnum_z * num_fields;
    const int stride_y = gridnum_z * num_fields;
    const int stride_z = num_fields;

    // Zero output arrays
    std::memset(fields,  0, sizeof(float) * gridnum_x * gridnum_y * gridnum_z * num_fields);
    std::memset(weights, 0, sizeof(float) * gridnum_x * gridnum_y * gridnum_z);

    // Neighbor offsets
    const int offsets[3] = {-1, 0, 1};

    for_each_particle(N, parallel, threads, [&](int n) {

        // compute normalized position in grid units
        float xpos = pos[3 * n + 0] * inv_cell_size_x;
        float ypos = pos[3 * n + 1] * inv_cell_size_y;
        float zpos = pos[3 * n + 2] * inv_cell_size_z;

        // compute base cell index (mother cell)
        int i_base = static_cast<int>(std::floor(xpos));
        int j_base = static_cast<int>(std::floor(ypos));
        int k_base = static_cast<int>(std::floor(zpos));

        // compute fractional distance within cell
        float dx = xpos - i_base;
        float dy = ypos - j_base;
        float dz = zpos - k_base;

        // compute TSC weights for each dimension
        auto wx = tsc_weights(dx);
        auto wy = tsc_weights(dy);
        auto wz = tsc_weights(dz);
        const float* particle = quantities + n * num_fields;

        // iterate over the 3x3x3 neighboring grid cells
        for (int i = 0; i < 3; ++i) {
            int ix = i_base + offsets[i];

            // check periodicity and apply PBC if needed, early-out if outside domain
            ix = apply_pbc(ix, gridnum_x, periodic_x);
            if (is_outside_domain(ix, gridnum_x)) continue;
            
            for (int j = 0; j < 3; ++j) {
                int iy = j_base + offsets[j];

                // check periodicity and apply PBC if needed, early-out if outside domain
                iy = apply_pbc(iy, gridnum_y, periodic_y);
                if (is_outside_domain(iy, gridnum_y)) continue;
                
                for (int k = 0; k < 3; ++k) {
                    int iz = k_base + offsets[k];

                    // check periodicity and apply PBC if needed, early-out if outside domain
                    iz = apply_pbc(iz, gridnum_z, periodic_z);
                    if (is_outside_domain(iz, gridnum_z)) continue;

                    // compute combined weight for this neighbor cell
                    float w = wx[i] * wy[j] * wz[k];
                    if (w == 0.0f) continue;

                    // deposit to grid
                    int base_idx   = ix * stride_x + iy * stride_y + iz * stride_z;
                    int weight_idx = ix * gridnum_y * gridnum_z + iy * gridnum_z + iz;
                    accumulate_fields(fields, base_idx, particle, num_fields, w, parallel);
                    accumulate_weight(weights, weight_idx, w, parallel);
                }
            }
        }
    });
}


// 1D CDF of TSC kernel for adaptive cell size h
inline float tsc_cdf_1d(float z, float h) {
    float s = z < 0 ? -1.0f : 1.0f;
    float x = std::abs(z) / h;
    float integral = 0.0f;

    if (x < 0.5f) {
        integral = (0.75f * x - x*x*x/3.0f) * h;
    } else if (x < 1.5f) {
        float a = 1.5f - x;
        integral = 0.5f * h - 0.5f * a * a * h;
    } else {
        integral = 0.5f * h;
    }

    return s > 0 ? 0.5f + integral / h : 0.5f - integral / h;
}

// Integrated weight over a grid cell
inline float tsc_integrated_weight_1d(float x_center, float cell_left, float cell_right, float h) {
    return tsc_cdf_1d(cell_right - x_center, h) - tsc_cdf_1d(cell_left - x_center, h);
}

void tsc_2d_adaptive_cpp(
    const float* pos,           // (N, 2)
    const float* quantities,    // (N, num_fields)
    int N,
    int num_fields,
    const float* boxsizes,      // (2,)
    const int* gridnums,        // (2,)
    const bool* periodic,       // (2,)
    const float* pcellsizesHalf,// (N)
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
    
    // extract periodicity flags
    const bool periodic_x = periodic[0];
    const bool periodic_y = periodic[1];

    // prepare output arrays
    const int stride_x = gridnum_y * num_fields;
    const int stride_y = num_fields;
    const int weight_stride_x = gridnum_y;
    const int weight_stride_y = 1;
    std::memset(fields,  0, sizeof(float) * gridnum_x * gridnum_y * num_fields);
    std::memset(weights, 0, sizeof(float) * gridnum_x * gridnum_y);

    // iterate over particles and accumulate to grid
    for_each_particle(N, parallel, threads, [&](int n) {

        // compute normalized position in grid units
        float x = pos[2 * n + 0] * inv_cell_size_x;
        float y = pos[2 * n + 1] * inv_cell_size_y;
        float h_x = pcellsizesHalf[n] * inv_cell_size_x;
        float h_y = pcellsizesHalf[n] * inv_cell_size_y;
        float support_x = 1.5f * h_x;
        float support_y = 1.5f * h_y;

        // compute bounding box of grid cells that the particle overlaps with
        int i_min = static_cast<int>(std::floor(x - support_x));
        int i_max = static_cast<int>(std::ceil(x + support_x));
        int j_min = static_cast<int>(std::floor(y - support_y));
        int j_max = static_cast<int>(std::ceil(y + support_y));
        const float* particle = quantities + n * num_fields;

        // iterate over grid cells in x direction
        for (int i = i_min; i <= i_max; ++i) {
            int ii = i;

            // check periodicity and apply PBC if needed, otherwise early-out if outside domain
            ii = apply_pbc(ii, gridnum_x, periodic_x);
            if (is_outside_domain(ii, gridnum_x)) continue;

            // compute integrated weight for this cell in x direction
            float wx = tsc_integrated_weight_1d(x, float(i), float(i+1), h_x);
            if (wx == 0.0f) continue;

            // iterate over grid cells in y direction
            for (int j = j_min; j <= j_max; ++j) {
                int jj = j;

                // check periodicity and apply PBC if needed, otherwise early-out if outside domain
                jj = apply_pbc(jj, gridnum_y, periodic_y);
                if (is_outside_domain(jj, gridnum_y)) continue;

                // compute integrated weight for this cell in y direction
                float wy = tsc_integrated_weight_1d(y, float(j), float(j+1), h_y);
                if (wy == 0.0f) continue;

                // compute combined weight for this grid cell
                float w = wx * wy;

                // deposit to grid  
                int base_idx = ii * stride_x + jj * stride_y;
                int weight_idx = ii * weight_stride_x + jj * weight_stride_y;
                accumulate_fields(fields, base_idx, particle, num_fields, w, parallel);
                accumulate_weight(weights, weight_idx, w, parallel);
            }
        }
    });
}

void tsc_3d_adaptive_cpp(
    const float* pos,           // (N,3)
    const float* quantities,    // (N,num_fields)
    int N,
    int num_fields,
    const float* boxsizes,      // (3,)
    const int* gridnums,        // (3,)
    const bool* periodic,       // (3,)
    const float* pcellsizesHalf,// (N)
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
    
    // extract periodicity flags
    const bool periodic_x = periodic[0];
    const bool periodic_y = periodic[1];
    const bool periodic_z = periodic[2];

    // prepare output arrays
    const int stride_x = gridnum_y * gridnum_z * num_fields;
    const int stride_y = gridnum_z * num_fields;
    const int stride_z = num_fields;
    const int weight_stride_x = gridnum_y * gridnum_z;
    const int weight_stride_y = gridnum_z;
    const int weight_stride_z = 1;
    std::memset(fields,  0, sizeof(float) * gridnum_x * gridnum_y * gridnum_z * num_fields);
    std::memset(weights, 0, sizeof(float) * gridnum_x * gridnum_y * gridnum_z);

    // iterate over particles and accumulate to grid
    for_each_particle(N, parallel, threads, [&](int n) {

        // compute normalized position in grid units
        float x = pos[3 * n + 0] * inv_cell_size_x;
        float y = pos[3 * n + 1] * inv_cell_size_y;
        float z = pos[3 * n + 2] * inv_cell_size_z;
        float h_x = pcellsizesHalf[n] * inv_cell_size_x;
        float h_y = pcellsizesHalf[n] * inv_cell_size_y;
        float h_z = pcellsizesHalf[n] * inv_cell_size_z;
        float support_x = 1.5f * h_x;
        float support_y = 1.5f * h_y;
        float support_z = 1.5f * h_z;

        // compute bounding box of grid cells that the particle overlaps with
        int i_min = static_cast<int>(std::floor(x - support_x));
        int i_max = static_cast<int>(std::ceil(x + support_x));
        int j_min = static_cast<int>(std::floor(y - support_y));
        int j_max = static_cast<int>(std::ceil(y + support_y));
        int k_min = static_cast<int>(std::floor(z - support_z));
        int k_max = static_cast<int>(std::ceil(z + support_z));
        const float* particle = quantities + n * num_fields;

        // iterate over grid cells in x direction
        for (int i = i_min; i <= i_max; ++i) {
            int ii = i;

            // check periodicity and apply PBC if needed, early-out if outside domain
            ii = apply_pbc(ii, gridnum_x, periodic_x);
            if (is_outside_domain(ii, gridnum_x)) continue;
            
            // compute integrated weight for this cell in x direction
            float wx = tsc_integrated_weight_1d(x, float(i), float(i+1), h_x);
            if (wx == 0.0f) continue;

            // iterate over grid cells in y direction
            for (int j = j_min; j <= j_max; ++j) {
                int jj = j;

                // check periodicity and apply PBC if needed, early-out if outside domain
                jj = apply_pbc(jj, gridnum_y, periodic_y);
                if (is_outside_domain(jj, gridnum_y)) continue;

                // compute integrated weight for this cell in y direction
                float wy = tsc_integrated_weight_1d(y, float(j), float(j+1), h_y);
                if (wy == 0.0f) continue;

                // iterate over grid cells in z direction
                for (int k = k_min; k <= k_max; ++k) {
                    int kk = k;

                    // check periodicity and apply PBC if needed, early-out if outside domain
                    kk = apply_pbc(kk, gridnum_z, periodic_z);
                    if (is_outside_domain(kk, gridnum_z)) continue;

                    // compute integrated weight for this cell in z direction
                    float wz = tsc_integrated_weight_1d(z, float(k), float(k+1), h_z);
                    if (wz == 0.0f) continue;

                    // compute combined weight for this grid cell
                    float w = wx * wy * wz;

                    // deposit to grid
                    int base_idx = ii * stride_x + jj * stride_y + kk * stride_z;
                    int weight_idx = ii * weight_stride_x + jj * weight_stride_y + kk * weight_stride_z;
                    accumulate_fields(fields, base_idx, particle, num_fields, w, parallel);
                    accumulate_weight(weights, weight_idx, w, parallel);
                }
            }
        }
    });
}


// =============================================================================
// Helper functions for iso- and anisotropic kernel depositions
// =============================================================================

template <typename Eval2D>
static float integrate_cell_2d(const std::string& method, const Eval2D& eval) {
    if (method == "midpoint") {
        return eval(0.5f, 0.5f);
    }

    if (method == "trapezoidal") {
        float sum = 0.0f;
        sum += eval(0.0f, 0.0f);
        sum += eval(1.0f, 0.0f);
        sum += eval(0.0f, 1.0f);
        sum += eval(1.0f, 1.0f);
        return (sum / 4.0f);
    }

    if (method == "simpson") {
        float sum = 0.0f;

        // corners
        sum += eval(0.0f, 0.0f);
        sum += eval(1.0f, 0.0f);
        sum += eval(0.0f, 1.0f);
        sum += eval(1.0f, 1.0f);

        // edge midpoints
        sum += 4.0f * eval(0.5f, 0.0f);
        sum += 4.0f * eval(0.5f, 1.0f);
        sum += 4.0f * eval(0.0f, 0.5f);
        sum += 4.0f * eval(1.0f, 0.5f);

        // center
        sum += 16.0f * eval(0.5f, 0.5f);

        return (sum / 36.0f);
    }

    throw std::invalid_argument("Unknown integration method: " + method);
}

template <typename Eval3D>
static float integrate_cell_3d(const std::string& method, const Eval3D& eval) {
    if (method == "midpoint") {
        return eval(0.5f, 0.5f, 0.5f);
    }

    if (method == "trapezoidal") {
        float sum = 0.0f;
        for (int i = 0; i <= 1; ++i)
            for (int j = 0; j <= 1; ++j)
                for (int k = 0; k <= 1; ++k)
                    sum += eval(i, j, k);
        return (sum / 8.0f);
    }

    if (method == "simpson") {
        float sum = 0.0f;

        // corners
        for (int i = 0; i <= 1; ++i)
            for (int j = 0; j <= 1; ++j)
                for (int k = 0; k <= 1; ++k)
                    sum += eval(i, j, k);

        // edge midpoints
        for (int i = 0; i <= 1; ++i)
            for (int j = 0; j <= 1; ++j) {
                sum += 4.0f * eval(0.5f, i, j);
                sum += 4.0f * eval(i, 0.5f, j);
                sum += 4.0f * eval(i, j, 0.5f);
            }

        // face centers
        sum += 16.0f * eval(0.5f, 0.5f, 0.0f);
        sum += 16.0f * eval(0.5f, 0.5f, 1.0f);
        sum += 16.0f * eval(0.5f, 0.0f, 0.5f);
        sum += 16.0f * eval(0.5f, 1.0f, 0.5f);
        sum += 16.0f * eval(0.0f, 0.5f, 0.5f);
        sum += 16.0f * eval(1.0f, 0.5f, 0.5f);

        // center
        sum += 64.0f * eval(0.5f, 0.5f, 0.5f);

        return (sum / 216.0f);
    }

    throw std::invalid_argument("Unknown integration method: " + method);
}


// =============================================================================
// SPH isotropic kernel deposition (2D)
// =============================================================================

void isotropic_kernel_deposition_2d_cpp(
    const float* pos,           // (N, 2)
    const float* quantities,    // (N, num_fields)
    const float* hsm,           // (N)
    int N,
    int num_fields,
    const float* boxsizes,      // (2,)
    const int* gridnums,        // (2,)
    const bool* periodic,       // (2,)
    const std::string& kernel_name,
    const std::string& integration_method,
    int min_kernel_evaluations,
    bool use_openmp,
    int omp_threads,
    float* fields,             // (gridnum_x, gridnum_y, num_fields)
    float* weights             // (gridnum_x, gridnum_y)
) {
    // resolve openMP settings
    const bool parallel = allow_openmp(use_openmp);
    const int threads = resolve_openmp_threads(parallel, omp_threads);

    // set up the kernel and cache integral samples
    auto kernel = create_kernel(kernel_name, 2);
    SPHKernel* kernel_ptr = kernel.get();
    const float kernel_support = kernel->support();
    const auto kernel_samples = build_kernel_sample_grid(*kernel, min_kernel_evaluations);

    // extract boxsize parameters
    const float boxsize_x = boxsizes[0];
    const float boxsize_y = boxsizes[1];

    // extract grid parameters
    const int gridnum_x = gridnums[0];
    const int gridnum_y = gridnums[1];

    // determine periodicity for each axis
    const bool periodic_x = periodic[0];
    const bool periodic_y = periodic[1];

    // compute cell sizes and related parameters
    const float cellSize_x = boxsize_x / static_cast<float>(gridnum_x);
    const float cellSize_y = boxsize_y / static_cast<float>(gridnum_y);

    // compute strides for fields/weights and set up output arrays
    const int stride_x = gridnum_y * num_fields;
    const int stride_y = num_fields;
    const int weight_stride_x = gridnum_y;
    std::memset(fields,  0, sizeof(float) * gridnum_x * gridnum_y * num_fields);
    std::memset(weights, 0, sizeof(float) * gridnum_x * gridnum_y);

    // perform for loop over particles
    for_each_particle(N, parallel, threads, [&](int n) {

        // gather relevant values for the current particle
        float hsm_current = hsm[n];
        float detH = hsm_current * hsm_current;
        
        // convert particle position to cell units
        float x_pos = pos[2 * n + 0];
        float y_pos = pos[2 * n + 1];
        float x_cell = x_pos / cellSize_x;
        float y_cell = y_pos / cellSize_y;

        // identify mother cell of particle
        int i = static_cast<int>(x_cell);
        int j = static_cast<int>(y_cell);

        // compute kernel support in physical and cell units
        float support_phys = kernel_support * hsm_current;
        float support_x_cell = support_phys / cellSize_x;
        float support_y_cell = support_phys / cellSize_y;
        float kernel_normalization = kernel_ptr->normalization(detH);

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

        // if the number of cells is small, use the cached kernel samples to evaluate the kernel at each sample point
        // -> iteration happens over the kernel sample points
        if (total_cells < min_kernel_evaluations) {
            const int count = kernel_samples.count;
            for (int s = 0; s < count; ++s) {
                
                // kernel sample positions mapped to physical space
                float x_phys = x_pos + kernel_samples.coords[2 * s + 0] * hsm_current;
                float y_phys = y_pos + kernel_samples.coords[2 * s + 1] * hsm_current;
                
                // given the geometry, determine the cell into which sample falls
                auto ix = cell_index_from_pos(x_phys, boxsize_x, gridnum_x, periodic_x);
                auto iy = cell_index_from_pos(y_phys, boxsize_y, gridnum_y, periodic_y);
                if (!ix) continue;
                if (!iy) continue;

                // gather the kernel sample integral (fraction)
                float integral = kernel_normalization * kernel_samples.integrals[s];
                if (integral == 0.0f) continue;

                // deposit to grid
                int base_idx = (*ix) * stride_x + (*iy) * stride_y;
                int weight_idx = (*ix) * weight_stride_x + (*iy);
                accumulate_fields(fields, base_idx, particle, num_fields, integral, parallel);
                accumulate_weight(weights, weight_idx, integral, parallel);
            }
        }
        // if the number of cells is large, iterate over affected cells and compute kernel integral over cell domain
        else {
            for (int a = i_min; a <= i_max; ++a) {
                int an = a;
                an = apply_pbc(an, gridnum_x, periodic_x);
                if (is_outside_domain(an, gridnum_x)) continue;

                for (int b = j_min; b <= j_max; ++b) {
                    int bn = b;
                    bn = apply_pbc(bn, gridnum_y, periodic_y);
                    if (is_outside_domain(bn, gridnum_y)) continue;

                    // set up helper function for integral evaluation using method
                    auto eval = [&](float ox, float oy) {
                        float dx = (x_cell - (a + ox)) * cellSize_x;
                        float dy = (y_cell - (b + oy)) * cellSize_y;

                        // optionally apply PBC wrapping
                        dx = wrap_distance_if_periodic(dx, boxsize_x, periodic_x);
                        dy = wrap_distance_if_periodic(dy, boxsize_y, periodic_y);

                        float r = std::sqrt(dx * dx + dy * dy);
                        float q = r / hsm_current;
                        return kernel_ptr->evaluate(q);
                    };

                    // compute kernel integral over cell using method
                    float integral = integrate_cell_2d(integration_method, eval);
                    integral *= kernel_normalization;
                    if (integral == 0.0f) continue;

                    // deposit to grid
                    int base_idx = an * stride_x + bn * stride_y;
                    int weight_idx = an * weight_stride_x + bn;
                    accumulate_fields(fields, base_idx, particle, num_fields, integral, parallel);
                    accumulate_weight(weights, weight_idx, integral, parallel);
                }
            }
        }
    });
}


// =============================================================================
// SPH isotropic kernel deposition (3D)
// =============================================================================

void isotropic_kernel_deposition_3d_cpp(
    const float* pos,           // (N, 3)
    const float* quantities,    // (N, num_fields)
    const float* hsm,           // (N)
    int N,
    int num_fields,
    const float* boxsizes,      // (3,)
    const int* gridnums,        // (3,)
    const bool* periodic,       // (3,)
    const std::string& kernel_name,
    const std::string& integration_method,
    int min_kernel_evaluations,
    bool use_openmp,
    int omp_threads,
    float* fields,             // (gridnum_x, gridnum_y, gridnum_z, num_fields)
    float* weights             // (gridnum_x, gridnum_y, gridnum_z)
) {
    // resolve openMP settings
    const bool parallel = allow_openmp(use_openmp);
    const int threads = resolve_openmp_threads(parallel, omp_threads);

    // set up the kernel and cache integral samples
    auto kernel = create_kernel(kernel_name, 3);
    SPHKernel* kernel_ptr = kernel.get();
    const auto kernel_samples = build_kernel_sample_grid(*kernel, min_kernel_evaluations);
    const float kernel_support = kernel->support();

    // extract boxsize parameters
    const float boxsize_x = boxsizes[0];
    const float boxsize_y = boxsizes[1];
    const float boxsize_z = boxsizes[2];

    // extract grid parameters
    const int gridnum_x = gridnums[0];
    const int gridnum_y = gridnums[1];
    const int gridnum_z = gridnums[2];

    // determine periodicity for each axis
    const bool periodic_x = periodic[0];
    const bool periodic_y = periodic[1];
    const bool periodic_z = periodic[2];

    // compute cell sizes and related parameters
    const float cellSize_x = boxsize_x / static_cast<float>(gridnum_x);
    const float cellSize_y = boxsize_y / static_cast<float>(gridnum_y);
    const float cellSize_z = boxsize_z / static_cast<float>(gridnum_z);

    // compute strides for fields/weights and set up output arrays
    const int stride_x = gridnum_y * gridnum_z * num_fields;
    const int stride_y = gridnum_z * num_fields;
    const int stride_z = num_fields;
    const int weight_stride_x = gridnum_y * gridnum_z;
    const int weight_stride_y = gridnum_z;
    std::memset(fields,  0, sizeof(float) * gridnum_x * gridnum_y * gridnum_z * num_fields);
    std::memset(weights, 0, sizeof(float) * gridnum_x * gridnum_y * gridnum_z);

    // perform for loop over particles
    for_each_particle(N, parallel, threads, [&](int n) {

        // gather relevant values for the current particle
        float hsm_current = hsm[n];
        float detH = hsm_current * hsm_current * hsm_current;

        // convert particle position to cell units
        float x_pos = pos[3 * n + 0];
        float y_pos = pos[3 * n + 1];
        float z_pos = pos[3 * n + 2];
        float x_cell = x_pos / cellSize_x;
        float y_cell = y_pos / cellSize_y;
        float z_cell = z_pos / cellSize_z;

        // identify mother cell of particle
        int i = static_cast<int>(x_cell);
        int j = static_cast<int>(y_cell);
        int k = static_cast<int>(z_cell);

        // compute kernel support in physical and cell units
        float support_phys = kernel_support * hsm_current;
        float support_x_cell = support_phys / cellSize_x;
        float support_y_cell = support_phys / cellSize_y;
        float support_z_cell = support_phys / cellSize_z;
        float kernel_normalization = kernel_ptr->normalization(detH);

        // compute inclusive index bounds within the kernel support
        int i_min = static_cast<int>(std::floor(x_cell - support_x_cell));
        int i_max = static_cast<int>(std::ceil(x_cell + support_x_cell)) - 1;
        int j_min = static_cast<int>(std::floor(y_cell - support_y_cell));
        int j_max = static_cast<int>(std::ceil(y_cell + support_y_cell)) - 1;
        int k_min = static_cast<int>(std::floor(z_cell - support_z_cell));
        int k_max = static_cast<int>(std::ceil(z_cell + support_z_cell)) - 1;
        const float* particle = quantities + n * num_fields;

        // compute total number of cells in the kernel support
        int num_cells_x = i_max - i_min + 1;
        int num_cells_y = j_max - j_min + 1;
        int num_cells_z = k_max - k_min + 1;
        int total_cells = num_cells_x * num_cells_y * num_cells_z;

        // if the number of cells is small, use the cached kernel samples to evaluate the kernel at each sample point
        // -> iteration changes over the kernel sample points
        if (total_cells < min_kernel_evaluations) {
            const int count = kernel_samples.count;
            for (int s = 0; s < count; ++s) {
                // kernel sample positions and mapping to physical space
                float x_phys = x_pos + kernel_samples.coords[3 * s + 0] * hsm_current;
                float y_phys = y_pos + kernel_samples.coords[3 * s + 1] * hsm_current;
                float z_phys = z_pos + kernel_samples.coords[3 * s + 2] * hsm_current;

                auto ix = cell_index_from_pos(x_phys, boxsize_x, gridnum_x, periodic_x);
                if (!ix) continue;
                auto iy = cell_index_from_pos(y_phys, boxsize_y, gridnum_y, periodic_y);
                if (!iy) continue;
                auto iz = cell_index_from_pos(z_phys, boxsize_z, gridnum_z, periodic_z);
                if (!iz) continue;

                // gather the kernel sample integral (fraction)
                float integral = kernel_normalization * kernel_samples.integrals[s];
                if (integral == 0.0f) continue;

                // deposit to grid
                int base_idx = (*ix) * stride_x + (*iy) * stride_y + (*iz) * stride_z;
                int weight_idx = (*ix) * weight_stride_x + (*iy) * weight_stride_y + (*iz);
                accumulate_fields(fields, base_idx, particle, num_fields, integral, parallel);
                accumulate_weight(weights, weight_idx, integral, parallel);
            }
        }
        // if the number of cells is large, iterate over affected cells and compute kernel integral over cell domain
        else {
            for (int a = i_min; a <= i_max; ++a) {
                int an = a;
                an = apply_pbc(an, gridnum_x, periodic_x);
                if (is_outside_domain(an, gridnum_x)) continue;

                for (int b = j_min; b <= j_max; ++b) {
                    int bn = b;
                    bn = apply_pbc(bn, gridnum_y, periodic_y);
                    if (is_outside_domain(bn, gridnum_y)) continue;

                    for (int c = k_min; c <= k_max; ++c) {
                        int cn = c;
                        cn = apply_pbc(cn, gridnum_z, periodic_z);
                        if (is_outside_domain(cn, gridnum_z)) continue;

                        // set up helper function for integral evaluation using method
                        auto eval = [&](float ox, float oy, float oz) {
                            float dx = (x_cell - (a + ox)) * cellSize_x;
                            float dy = (y_cell - (b + oy)) * cellSize_y;
                            float dz = (z_cell - (c + oz)) * cellSize_z;

                            // optionally apply PBC wrapping
                            dx = wrap_distance_if_periodic(dx, boxsize_x, periodic_x);
                            dy = wrap_distance_if_periodic(dy, boxsize_y, periodic_y);
                            dz = wrap_distance_if_periodic(dz, boxsize_z, periodic_z);

                            float r = std::sqrt(dx * dx + dy * dy + dz * dz);
                            float q = r / hsm_current;
                            return kernel_ptr->evaluate(q);
                        };

                        // compute kernel integral over cell using method
                        float integral = integrate_cell_3d(integration_method, eval);
                        integral *= kernel_normalization;
                        if (integral == 0.0f) continue;

                        // deposit to grid
                        int base_idx = an * stride_x + bn * stride_y + cn * stride_z;
                        int weight_idx = an * weight_stride_x + bn * weight_stride_y + cn;
                        accumulate_fields(fields, base_idx, particle, num_fields, integral, parallel);
                        accumulate_weight(weights, weight_idx, integral, parallel);
                    }
                }
            }
        }
    });
}


// =============================================================================
// SPH anisotropic kernel deposition (2D)
// =============================================================================

void anisotropic_kernel_deposition_2d_cpp(
    const float* pos,           // (N, 2)
    const float* quantities,    // (N, num_fields)
    const float* hmat_eigvecs,  // (N, 4) - stored as [v00, v10, v01, v11] for each particle
    const float* hmat_eigvals,  // (N, 2) - stored as [lambda0, lambda1] for each particle
    int N,
    int num_fields,
    const float* boxsizes,      // (2,)
    const int* gridnums,        // (2,)
    const bool* periodic,       // (2,)
    const std::string& kernel_name,
    const std::string& integration_method,
    int min_kernel_evaluations,
    bool use_openmp,
    int omp_threads,
    float* fields,              // (gridnum_x, gridnum_y, num_fields)   
    float* weights              // (gridnum_x, gridnum_y)
) {
    // resolve openMP settings
    const bool parallel = allow_openmp(use_openmp);
    const int threads = resolve_openmp_threads(parallel, omp_threads);

    // set up the kernel and cache integral samples
    auto kernel = create_kernel(kernel_name, 2);
    SPHKernel* kernel_ptr = kernel.get();
    const float kernel_support = kernel->support();
    const auto kernel_samples = build_kernel_sample_grid(*kernel, min_kernel_evaluations);

    // extract boxsize parameters
    const float boxsize_x = boxsizes[0];
    const float boxsize_y = boxsizes[1];

    // extract grid parameters
    const int gridnum_x = gridnums[0];
    const int gridnum_y = gridnums[1];

    // determine periodicity for each axis
    const bool periodic_x = periodic[0];
    const bool periodic_y = periodic[1];

    // compute cell sizes and related parameters
    const float cellSize_x = boxsize_x / static_cast<float>(gridnum_x);
    const float cellSize_y = boxsize_y / static_cast<float>(gridnum_y);
    
    // compute strides for fields/weights and set up output arrays
    const int stride_x = gridnum_y * num_fields;
    const int stride_y = num_fields;
    const int weight_stride_x = gridnum_y;
    std::memset(fields,  0, sizeof(float) * gridnum_x * gridnum_y * num_fields);
    std::memset(weights, 0, sizeof(float) * gridnum_x * gridnum_y);
    
    // perform for loop over particles
    for_each_particle(N, parallel, threads, [&](int n) {
        
        // gather relevant values for the current particle
        const float* vecs = &hmat_eigvecs[n * 4];
        const float* vals = &hmat_eigvals[n * 2];
        float vals_gu[2] = { vals[0] / cellSize_x, vals[1] / cellSize_y };
        float detH = vals[0] * vals[1];
        float kernel_normalization = kernel_ptr->normalization(detH);
        
        // convert particle position to cell units
        float x_pos = pos[2 * n + 0];
        float y_pos = pos[2 * n + 1];
        float x_cell = x_pos / cellSize_x;
        float y_cell = y_pos / cellSize_y;

        // identify mother cell of particle
        int i = static_cast<int>(x_cell);
        int j = static_cast<int>(y_cell);

        // figure out the extent of the kernel 
        float support_x_cell = kernel_support * std::sqrt(
            (vecs[0] * vals_gu[0]) * (vecs[0] * vals_gu[0]) +
            (vecs[2] * vals_gu[1]) * (vecs[2] * vals_gu[1])
        );
        float support_y_cell = kernel_support * std::sqrt(
            (vecs[1] * vals_gu[0]) * (vecs[1] * vals_gu[0]) +
            (vecs[3] * vals_gu[1]) * (vecs[3] * vals_gu[1])
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

        // if the number of cells is small, use the cached kernel samples to evaluate the kernel at each sample point
        // -> iteration changes over the kernel sample points
        if (total_cells < min_kernel_evaluations) {
            const int count = kernel_samples.count;

            for (int s = 0; s < count; ++s) {

                // kernel sample positions and mapping to physical space
                float x_phys = x_pos + vecs[0] * (vals[0] * kernel_samples.coords[2 * s + 0])
                                      + vecs[2] * (vals[1] * kernel_samples.coords[2 * s + 1]);
                float y_phys = y_pos + vecs[1] * (vals[0] * kernel_samples.coords[2 * s + 0])
                                      + vecs[3] * (vals[1] * kernel_samples.coords[2 * s + 1]);

                auto ix = cell_index_from_pos(x_phys, boxsize_x, gridnum_x, periodic_x);
                if (!ix) continue;
                auto iy = cell_index_from_pos(y_phys, boxsize_y, gridnum_y, periodic_y);
                if (!iy) continue;

                // gather the kernel sample integral (fraction)
                float integral = kernel_normalization * kernel_samples.integrals[s];
                if (integral == 0.0f) continue;

                // deposit to grid
                int base_idx = (*ix) * stride_x + (*iy) * stride_y;
                int weight_idx = (*ix) * weight_stride_x + (*iy);
                accumulate_fields(fields, base_idx, particle, num_fields, integral, parallel);
                accumulate_weight(weights, weight_idx, integral, parallel);
            }
        } 
        // if the number of cells is large, iterate over affected cells and compute kernel integral over cell domain
        else {
            for (int a = i_min; a <= i_max; ++a) {
                int an = a;
                an = apply_pbc(an, gridnum_x, periodic_x);
                if (is_outside_domain(an, gridnum_x)) continue;

                for (int b = j_min; b <= j_max; ++b) {
                    int bn = b;
                    bn = apply_pbc(bn, gridnum_y, periodic_y);
                    if (is_outside_domain(bn, gridnum_y)) continue;

                    // set up helper function for integral evaluation using method
                    auto eval = [&](float ox, float oy) {
                        float dx = (x_cell - (a + ox)) * cellSize_x;
                        float dy = (y_cell - (b + oy)) * cellSize_y;

                        // optionally apply PBC wrapping
                        dx = wrap_distance_if_periodic(dx, boxsize_x, periodic_x);
                        dy = wrap_distance_if_periodic(dy, boxsize_y, periodic_y);

                        // compute q in transformed space and evaluate kernel
                        float xi1 = (vecs[0] * dx + vecs[1] * dy) / vals[0];
                        float xi2 = (vecs[2] * dx + vecs[3] * dy) / vals[1];
                        float q = std::sqrt(xi1 * xi1 + xi2 * xi2);
                        return kernel_ptr->evaluate(q);
                    };

                    // compute kernel integral over cell using method
                    float integral = integrate_cell_2d(integration_method, eval);
                    integral *= kernel_normalization;
                    if (integral == 0.0f) continue;

                    // deposit to grid
                    int base_idx = an * stride_x + bn * stride_y;
                    int weight_idx = an * weight_stride_x + bn;
                    accumulate_fields(fields, base_idx, particle, num_fields, integral, parallel);
                    accumulate_weight(weights, weight_idx, integral, parallel);
                }
            }
        }
    });
}


// =============================================================================
// SPH anisotropic kernel deposition (3D)
// =============================================================================

void anisotropic_kernel_deposition_3d_cpp(
    const float* pos,           // (N, 3)
    const float* quantities,    // (N, num_fields)
    const float* hmat_eigvecs,  // (N, 9) - column-major eigenvectors
    const float* hmat_eigvals,  // (N, 3) - eigenvalues per particle
    int N,
    int num_fields,
    const float* boxsizes,      // (3,)
    const int* gridnums,        // (3,)
    const bool* periodic,       // (3,)
    const std::string& kernel_name,
    const std::string& integration_method,
    int min_kernel_evaluations,
    bool use_openmp,
    int omp_threads,
    float* fields,              // (gridnum_x, gridnum_y, gridnum_z, num_fields)
    float* weights              // (gridnum_x, gridnum_y, gridnum_z)
) {
    // resolve openMP settings
    const bool parallel = allow_openmp(use_openmp);
    const int threads = resolve_openmp_threads(parallel, omp_threads);
    
    // set up the kernel and cache integral samples
    auto kernel = create_kernel(kernel_name, 3);
    SPHKernel* kernel_ptr = kernel.get();
    const float kernel_support = kernel->support();
    const auto kernel_samples = build_kernel_sample_grid(*kernel, min_kernel_evaluations);

    // extract boxsize parameters
    const float boxsize_x = boxsizes[0];
    const float boxsize_y = boxsizes[1];
    const float boxsize_z = boxsizes[2];

    // extract grid parameters
    const int gridnum_x = gridnums[0];
    const int gridnum_y = gridnums[1];
    const int gridnum_z = gridnums[2];

    // determine periodicity for each axis
    const bool periodic_x = periodic[0];
    const bool periodic_y = periodic[1];
    const bool periodic_z = periodic[2];

    // compute cell sizes and related parameters
    const float cellSize_x = boxsize_x / static_cast<float>(gridnum_x);
    const float cellSize_y = boxsize_y / static_cast<float>(gridnum_y);
    const float cellSize_z = boxsize_z / static_cast<float>(gridnum_z);

    // compute strides for fields/weights and set up output arrays
    const int stride_x = gridnum_y * gridnum_z * num_fields;
    const int stride_y = gridnum_z * num_fields;
    const int stride_z = num_fields;
    const int weight_stride_x = gridnum_y * gridnum_z;
    const int weight_stride_y = gridnum_z;
    const int weight_stride_z = 1;
    std::memset(fields,  0, sizeof(float) * gridnum_x * gridnum_y * gridnum_z * num_fields);
    std::memset(weights, 0, sizeof(float) * gridnum_x * gridnum_y * gridnum_z);

    // perform for loop over particles
    for_each_particle(N, parallel, threads, [&](int n) {
        // gather relevant values for the current particle
        const float* vecs = &hmat_eigvecs[n * 9];
        const float* vals = &hmat_eigvals[n * 3];

        // compute detH and normalization
        float vals_gu[3] = { vals[0] / cellSize_x, vals[1] / cellSize_y, vals[2] / cellSize_z };
        float detH = vals[0] * vals[1] * vals[2];
        float kernel_normalization = kernel_ptr->normalization(detH);

        // convert particle position to cell units
        float x_pos = pos[3 * n + 0];
        float y_pos = pos[3 * n + 1];
        float z_pos = pos[3 * n + 2];
        float x_cell = x_pos / cellSize_x;
        float y_cell = y_pos / cellSize_y;
        float z_cell = z_pos / cellSize_z;

        // identify mother cell of particle
        int i = static_cast<int>(x_cell);
        int j = static_cast<int>(y_cell);
        int k = static_cast<int>(z_cell);

        // estimate kernel support along each axis in cell units
        float support_x_cell = kernel_support * std::sqrt(
            (vecs[0] * vals_gu[0]) * (vecs[0] * vals_gu[0]) +
            (vecs[3] * vals_gu[1]) * (vecs[3] * vals_gu[1]) +
            (vecs[6] * vals_gu[2]) * (vecs[6] * vals_gu[2])
        );
        float support_y_cell = kernel_support * std::sqrt(
            (vecs[1] * vals_gu[0]) * (vecs[1] * vals_gu[0]) +
            (vecs[4] * vals_gu[1]) * (vecs[4] * vals_gu[1]) +
            (vecs[7] * vals_gu[2]) * (vecs[7] * vals_gu[2])
        );
        float support_z_cell = kernel_support * std::sqrt(
            (vecs[2] * vals_gu[0]) * (vecs[2] * vals_gu[0]) +
            (vecs[5] * vals_gu[1]) * (vecs[5] * vals_gu[1]) +
            (vecs[8] * vals_gu[2]) * (vecs[8] * vals_gu[2])
        );

        // compute inclusive index bounds within the kernel support
        int i_min = static_cast<int>(std::floor(x_cell - support_x_cell));
        int i_max = static_cast<int>(std::ceil(x_cell + support_x_cell)) - 1;
        int j_min = static_cast<int>(std::floor(y_cell - support_y_cell));
        int j_max = static_cast<int>(std::ceil(y_cell + support_y_cell)) - 1;
        int k_min = static_cast<int>(std::floor(z_cell - support_z_cell));
        int k_max = static_cast<int>(std::ceil(z_cell + support_z_cell)) - 1;
        const float* particle = quantities + n * num_fields;

        // compute total number of cells in the kernel support
        int num_cells_x = i_max - i_min + 1;
        int num_cells_y = j_max - j_min + 1;
        int num_cells_z = k_max - k_min + 1;
        int total_cells = num_cells_x * num_cells_y * num_cells_z;

        // small-support path: use cached kernel samples
        if (total_cells < min_kernel_evaluations) {
            const int count = kernel_samples.count;
            for (int s = 0; s < count; ++s) {
                // kernel sample positions and mapping to physical space
                float x_phys = x_pos + vecs[0] * (vals[0] * kernel_samples.coords[3 * s + 0])
                                      + vecs[3] * (vals[1] * kernel_samples.coords[3 * s + 1])
                                      + vecs[6] * (vals[2] * kernel_samples.coords[3 * s + 2]);
                float y_phys = y_pos + vecs[1] * (vals[0] * kernel_samples.coords[3 * s + 0])
                                      + vecs[4] * (vals[1] * kernel_samples.coords[3 * s + 1])
                                      + vecs[7] * (vals[2] * kernel_samples.coords[3 * s + 2]);
                float z_phys = z_pos + vecs[2] * (vals[0] * kernel_samples.coords[3 * s + 0])
                                      + vecs[5] * (vals[1] * kernel_samples.coords[3 * s + 1])
                                      + vecs[8] * (vals[2] * kernel_samples.coords[3 * s + 2]);

                auto ix = cell_index_from_pos(x_phys, boxsize_x, gridnum_x, periodic_x);
                if (!ix) continue;
                auto iy = cell_index_from_pos(y_phys, boxsize_y, gridnum_y, periodic_y);
                if (!iy) continue;
                auto iz = cell_index_from_pos(z_phys, boxsize_z, gridnum_z, periodic_z);
                if (!iz) continue;

                // gather the kernel sample integral (fraction)
                float integral = kernel_normalization * kernel_samples.integrals[s];
                if (integral == 0.0f) continue;

                // deposit to grid
                int base_idx = (*ix) * stride_x + (*iy) * stride_y + (*iz) * stride_z;
                int weight_idx = (*ix) * weight_stride_x + (*iy) * weight_stride_y + (*iz);
                accumulate_fields(fields, base_idx, particle, num_fields, integral, parallel);
                accumulate_weight(weights, weight_idx, integral, parallel);
            }
        }
        // large-support path: integrate directly over grid cells
        else {
            for (int a = i_min; a <= i_max; ++a) {
                int an = a;
                an = apply_pbc(an, gridnum_x, periodic_x);
                if (is_outside_domain(an, gridnum_x)) continue;

                for (int b = j_min; b <= j_max; ++b) {
                    int bn = b;
                    bn = apply_pbc(bn, gridnum_y, periodic_y);
                    if (is_outside_domain(bn, gridnum_y)) continue;

                    for (int c = k_min; c <= k_max; ++c) {
                        int cn = c;
                        cn = apply_pbc(cn, gridnum_z, periodic_z);
                        if (is_outside_domain(cn, gridnum_z)) continue;

                        // set up helper function for integral evaluation using method
                        auto eval = [&](float ox, float oy, float oz) {
                            float dx = (x_cell - (a + ox)) * cellSize_x;
                            float dy = (y_cell - (b + oy)) * cellSize_y;
                            float dz = (z_cell - (c + oz)) * cellSize_z;

                            // optionally apply PBC wrapping
                            dx = wrap_distance_if_periodic(dx, boxsize_x, periodic_x);
                            dy = wrap_distance_if_periodic(dy, boxsize_y, periodic_y);
                            dz = wrap_distance_if_periodic(dz, boxsize_z, periodic_z);

                            // compute q in transformed space and evaluate kernel
                            float xi1 = (vecs[0] * dx + vecs[1] * dy + vecs[2] * dz) / vals[0];
                            float xi2 = (vecs[3] * dx + vecs[4] * dy + vecs[5] * dz) / vals[1];
                            float xi3 = (vecs[6] * dx + vecs[7] * dy + vecs[8] * dz) / vals[2];
                            float q = std::sqrt(xi1 * xi1 + xi2 * xi2 + xi3 * xi3);
                            return kernel_ptr->evaluate(q);
                        };

                        // compute kernel integral over cell using method
                        float integral = integrate_cell_3d(integration_method, eval);
                        integral *= kernel_normalization;
                        if (integral == 0.0f) continue;

                        // deposit to grid
                        int base_idx = an * stride_x + bn * stride_y + cn * stride_z;
                        int weight_idx = an * weight_stride_x + bn * weight_stride_y + cn;
                        accumulate_fields(fields, base_idx, particle, num_fields, integral, parallel);
                        accumulate_weight(weights, weight_idx, integral, parallel);
                    }
                }
            }
        }
    });
}
