#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>
#include <array>
#include <string>
#include <stdexcept>
#if defined(_OPENMP)
#include <omp.h>
#endif
#include "kernels.h"
#include "functions.h"

// =============================================================================
// Utilities
// =============================================================================

inline int apply_pbc(int idx, int gridnum) {
    int r = idx % gridnum;
    return (r < 0) ? r + gridnum : r;
}

// Centralizes per-axis periodic checks and gracefully handles nullable arrays
inline bool axis_periodic(const bool* periodic, int axis) {
	return periodic ? periodic[axis] : false;
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
    const float* pos,            // (N, 2)
    const float* quantities,     // (N, num_fields)
    int N,
    int num_fields,
    const float* boxsizes,
    const int* gridnums,
    const bool* periodic,
    bool use_openmp,
    int omp_threads,
    float* fields,               // (gridnum_x, gridnum_y, num_fields)
    float* weights               // (gridnum_x, gridnum_y)
) {
    const int gridnum_x = gridnums[0];
    const int gridnum_y = gridnums[1];
    const float inv_dx_x = static_cast<float>(gridnum_x) / boxsizes[0];
    const float inv_dx_y = static_cast<float>(gridnum_y) / boxsizes[1];

    // strides for C-contiguous layout (x, y, f)
    const int field_stride_x = gridnum_y * num_fields;
    const int field_stride_y = num_fields;

    // zero output arrays
    std::memset(fields,  0, sizeof(float) * gridnum_x * gridnum_y * num_fields);
    std::memset(weights, 0, sizeof(float) * gridnum_x * gridnum_y);

    const bool parallel = allow_openmp(use_openmp);
    const int threads = (parallel && omp_threads > 0) ? omp_threads : 0;
    for_each_particle(N, parallel, threads, [&](int n) {
        int ix = static_cast<int>(pos[2 * n + 0] * inv_dx_x);
        int iy = static_cast<int>(pos[2 * n + 1] * inv_dx_y);

        if (ix < 0 || ix >= gridnum_x) {
            return;
        }
        if (iy < 0 || iy >= gridnum_y) {
            return;
        }

        const int base_idx   = ix * field_stride_x + iy * field_stride_y;
        const int weight_idx = ix * gridnum_y + iy;
        const float* particle = quantities + n * num_fields;

        accumulate_fields(fields, base_idx, particle, num_fields, 1.0f, parallel);
        accumulate_weight(weights, weight_idx, 1.0f, parallel);
    });
}


void ngp_3d_cpp(
    const float* pos,            // (N, 3)
    const float* quantities,     // (N, num_fields)
    int N,
    int num_fields,
    const float* boxsizes,
    const int* gridnums,
    const bool* periodic,
    bool use_openmp,
    int omp_threads,
    float* fields,               // (gridnum_x, gridnum_y, gridnum_z, num_fields)
    float* weights               // (gridnum_x, gridnum_y, gridnum_z)
) {
    const int gridnum_x = gridnums[0];
    const int gridnum_y = gridnums[1];
    const int gridnum_z = gridnums[2];
    const float inv_dx_x = static_cast<float>(gridnum_x) / boxsizes[0];
    const float inv_dx_y = static_cast<float>(gridnum_y) / boxsizes[1];
    const float inv_dx_z = static_cast<float>(gridnum_z) / boxsizes[2];

    // strides for C-contiguous layout (x, y, z, f)
    const int field_stride_x = gridnum_y * gridnum_z * num_fields;
    const int field_stride_y = gridnum_z * num_fields;
    const int field_stride_z = num_fields;

    // zero output arrays
    std::memset(fields,  0, sizeof(float) * gridnum_x * gridnum_y * gridnum_z * num_fields);
    std::memset(weights, 0, sizeof(float) * gridnum_x * gridnum_y * gridnum_z);

    const bool parallel = allow_openmp(use_openmp);
    const int threads = (parallel && omp_threads > 0) ? omp_threads : 0;
    for_each_particle(N, parallel, threads, [&](int n) {
        int ix = static_cast<int>(pos[3 * n + 0] * inv_dx_x);
        int iy = static_cast<int>(pos[3 * n + 1] * inv_dx_y);
        int iz = static_cast<int>(pos[3 * n + 2] * inv_dx_z);

        if (ix < 0 || ix >= gridnum_x) {
            return;
        }

        if (iy < 0 || iy >= gridnum_y) {
            return;
        }

        if (iz < 0 || iz >= gridnum_z) {
            return;
        }

        const int base_idx   = ix * field_stride_x + iy * field_stride_y + iz * field_stride_z;
        const int weight_idx = ix * gridnum_y * gridnum_z + iy * gridnum_z + iz;
        const float* particle = quantities + n * num_fields;

        accumulate_fields(fields, base_idx, particle, num_fields, 1.0f, parallel);
        accumulate_weight(weights, weight_idx, 1.0f, parallel);
    });
}


void cic_2d_cpp(
    const float* pos,            // (N, 2)
    const float* quantities,     // (N, num_fields)
    int N,
    int num_fields,
    const float* boxsizes,
    const int* gridnums,
    const bool* periodic,
    bool use_openmp,
    int omp_threads,
    float* fields,               // (gridnum_x, gridnum_y, num_fields)
    float* weights               // (gridnum_x, gridnum_y)
) {
    const int gridnum_x = gridnums[0];
    const int gridnum_y = gridnums[1];
    const bool periodic_x = axis_periodic(periodic, 0);
    const bool periodic_y = axis_periodic(periodic, 1);
    const float inv_dx_x = static_cast<float>(gridnum_x) / boxsizes[0];
    const float inv_dx_y = static_cast<float>(gridnum_y) / boxsizes[1];

    // strides for C-contiguous layout (x, y, f)
    const int field_stride_x = gridnum_y * num_fields;
    const int field_stride_y = num_fields;

    // zero output arrays
    std::memset(fields,  0, sizeof(float) * gridnum_x * gridnum_y * num_fields);
    std::memset(weights, 0, sizeof(float) * gridnum_x * gridnum_y);

    const bool parallel = allow_openmp(use_openmp);
    const int threads = (parallel && omp_threads > 0) ? omp_threads : 0;
    for_each_particle(N, parallel, threads, [&](int n) {
        float xpos = pos[2 * n + 0] * inv_dx_x;
        float ypos = pos[2 * n + 1] * inv_dx_y;

        if ((!periodic_x && (xpos < 0.0f || xpos >= gridnum_x)) ||
            (!periodic_y && (ypos < 0.0f || ypos >= gridnum_y))) {
            return;
        }

        int i0 = static_cast<int>(std::floor(xpos));
        int j0 = static_cast<int>(std::floor(ypos));
        int i1 = i0 + 1;
        int j1 = j0 + 1;

        float dx = xpos - i0;
        float dy = ypos - j0;
        float dx_ = 1.0f - dx;
        float dy_ = 1.0f - dy;
        if ((!periodic_x && (xpos < 0.5f || xpos > gridnum_x - 0.5f)) ||
            (!periodic_y && (ypos < 0.5f || ypos > gridnum_y - 0.5f))) {
            return;
        }

        if (periodic_x) {
            i0 = apply_pbc(i0, gridnum_x);
            i1 = apply_pbc(i1, gridnum_x);
        }
        if (periodic_y) {
            j0 = apply_pbc(j0, gridnum_y);
            j1 = apply_pbc(j1, gridnum_y);
        }

        float w00 = dx_ * dy_;
        float w10 = dx  * dy_;
        float w01 = dx_ * dy;
        float w11 = dx  * dy;
        const float* particle = quantities + n * num_fields;

        auto deposit = [&](int ix, int jy, float w) {
            if (!periodic_x && (ix < 0 || ix >= gridnum_x)) return;
            if (!periodic_y && (jy < 0 || jy >= gridnum_y)) return;

            int base_idx   = ix * field_stride_x + jy * field_stride_y;
            int weight_idx = ix * gridnum_y + jy;
            accumulate_fields(fields, base_idx, particle, num_fields, w, parallel);
            accumulate_weight(weights, weight_idx, w, parallel);
        };

        deposit(i0, j0, w00);
        deposit(i1, j0, w10);
        deposit(i0, j1, w01);
        deposit(i1, j1, w11);
    });
}


void cic_3d_cpp(
    const float* pos,        // (N, 3)
    const float* quantities, // (N, num_fields)
    int N,
    int num_fields,
    const float* boxsizes,
    const int* gridnums,
    const bool* periodic,
    bool use_openmp,
    int omp_threads,
    float* fields,           // (gridnum_x, gridnum_y, gridnum_z, num_fields)
    float* weights           // (gridnum_x, gridnum_y, gridnum_z)
) {
    const int gridnum_x = gridnums[0];
    const int gridnum_y = gridnums[1];
    const int gridnum_z = gridnums[2];
    const bool periodic_x = axis_periodic(periodic, 0);
    const bool periodic_y = axis_periodic(periodic, 1);
    const bool periodic_z = axis_periodic(periodic, 2);
    const float inv_dx_x = static_cast<float>(gridnum_x) / boxsizes[0];
    const float inv_dx_y = static_cast<float>(gridnum_y) / boxsizes[1];
    const float inv_dx_z = static_cast<float>(gridnum_z) / boxsizes[2];

    // Strides for C-contiguous layout (x, y, z, f)
    const int field_stride_x = gridnum_y * gridnum_z * num_fields;
    const int field_stride_y = gridnum_z * num_fields;
    const int field_stride_z = num_fields;

    // Zero output arrays
    std::memset(fields,  0, sizeof(float) * gridnum_x * gridnum_y * gridnum_z * num_fields);
    std::memset(weights, 0, sizeof(float) * gridnum_x * gridnum_y * gridnum_z);

    const bool parallel = allow_openmp(use_openmp);
    const int threads = (parallel && omp_threads > 0) ? omp_threads : 0;
    for_each_particle(N, parallel, threads, [&](int n) {
        float xpos = pos[3 * n + 0] * inv_dx_x;
        float ypos = pos[3 * n + 1] * inv_dx_y;
        float zpos = pos[3 * n + 2] * inv_dx_z;

        if ((!periodic_x && (xpos < 0.0f || xpos >= gridnum_x)) ||
            (!periodic_y && (ypos < 0.0f || ypos >= gridnum_y)) ||
            (!periodic_z && (zpos < 0.0f || zpos >= gridnum_z))) {
            return;
        }

        int i0 = static_cast<int>(std::floor(xpos));
        int j0 = static_cast<int>(std::floor(ypos));
        int k0 = static_cast<int>(std::floor(zpos));
        int i1 = i0 + 1;
        int j1 = j0 + 1;
        int k1 = k0 + 1;

        float dx = xpos - i0;
        float dy = ypos - j0;
        float dz = zpos - k0;
        float dx_ = 1.0f - dx;
        float dy_ = 1.0f - dy;
        float dz_ = 1.0f - dz;

        if (periodic_x) {
            i0 = apply_pbc(i0, gridnum_x);
            i1 = apply_pbc(i1, gridnum_x);
        }
        if (periodic_y) {
            j0 = apply_pbc(j0, gridnum_y);
            j1 = apply_pbc(j1, gridnum_y);
        }
        if (periodic_z) {
            k0 = apply_pbc(k0, gridnum_z);
            k1 = apply_pbc(k1, gridnum_z);
        }

        float w000 = dx_ * dy_ * dz_;
        float w100 = dx  * dy_ * dz_;
        float w010 = dx_ * dy  * dz_;
        float w110 = dx  * dy  * dz_;
        float w001 = dx_ * dy_ * dz;
        float w101 = dx  * dy_ * dz;
        float w011 = dx_ * dy  * dz;
        float w111 = dx  * dy  * dz;
        const float* particle = quantities + n * num_fields;

        auto deposit = [&](int ix, int jy, int kz, float w) {
            if (!periodic_x && (ix < 0 || ix >= gridnum_x)) return;
            if (!periodic_y && (jy < 0 || jy >= gridnum_y)) return;
            if (!periodic_z && (kz < 0 || kz >= gridnum_z)) return;

            int base_idx   = ix * field_stride_x + jy * field_stride_y + kz * field_stride_z;
            int weight_idx = ix * gridnum_y * gridnum_z + jy * gridnum_z + kz;
            accumulate_fields(fields, base_idx, particle, num_fields, w, parallel);
            accumulate_weight(weights, weight_idx, w, parallel);
        };

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
    const float* pos,         // (N,2)
    const float* quantities,  // (N,num_fields)
    int N,
    int num_fields,
    const float* boxsizes,
    const int* gridnums,
    const bool* periodic,
    const float* pcellsizesHalf, // (N)
    bool use_openmp,
    int omp_threads,
    float* fields,            // (gridnum_x, gridnum_y, num_fields)
    float* weights            // (gridnum_x, gridnum_y)
) {
    const int gridnum_x = gridnums[0];
    const int gridnum_y = gridnums[1];
    const bool periodic_x = axis_periodic(periodic, 0);
    const bool periodic_y = axis_periodic(periodic, 1);
    const float cellSize_x = boxsizes[0] / static_cast<float>(gridnum_x);
    const float cellSize_y = boxsizes[1] / static_cast<float>(gridnum_y);
    const float inv_dx_x = 1.0f / cellSize_x;
    const float inv_dx_y = 1.0f / cellSize_y;
    const int stride_x = gridnum_y * num_fields;
    const int stride_y = num_fields;
    const int weight_stride_x = gridnum_y;

    std::memset(fields,  0, sizeof(float) * gridnum_x * gridnum_y * num_fields);
    std::memset(weights, 0, sizeof(float) * gridnum_x * gridnum_y);

    const bool parallel = allow_openmp(use_openmp);
    const int threads = (parallel && omp_threads > 0) ? omp_threads : 0;
    for_each_particle(N, parallel, threads, [&](int n) {
        float pcs_x = pcellsizesHalf[n] * inv_dx_x;
        float pcs_y = pcellsizesHalf[n] * inv_dx_y;
        float V = (2.0f * pcs_x) * (2.0f * pcs_y);

        float xpos = pos[2 * n + 0] * inv_dx_x;
        float ypos = pos[2 * n + 1] * inv_dx_y;

        int i = static_cast<int>(xpos);
        int j = static_cast<int>(ypos);

        int num_left   = i - static_cast<int>(std::round(xpos - pcs_x - 0.5f));
        int num_right  = static_cast<int>(xpos + pcs_x) - i;
        int num_bottom = j - static_cast<int>(std::round(ypos - pcs_y - 0.5f));
        int num_top    = static_cast<int>(ypos + pcs_y) - j;

        float c1 = xpos - pcs_x, c2 = xpos + pcs_x;
        float c3 = ypos - pcs_y, c4 = ypos + pcs_y;
        const float* particle = quantities + n * num_fields;

        for (int a = i - num_left; a <= i + num_right; ++a) {
            for (int b = j - num_bottom; b <= j + num_top; ++b) {
                float e1 = static_cast<float>(a);
                float e2 = e1 + 1.0f;
                float e3 = static_cast<float>(b);
                float e4 = e3 + 1.0f;

                float intersec_x = std::fmin(e2, c2) - std::fmax(e1, c1);
                float intersec_y = std::fmin(e4, c4) - std::fmax(e3, c3);
                float fraction = (intersec_x * intersec_y) / V;

                if (fraction <= 0.0f) continue;

                int an = a, bn = b;
                if (periodic_x) {
                    an = apply_pbc(a, gridnum_x);
                } else if (a < 0 || a >= gridnum_x) {
                    continue;
                }
                if (periodic_y) {
                    bn = apply_pbc(b, gridnum_y);
                } else if (b < 0 || b >= gridnum_y) {
                    continue;
                }

                int base_idx = an * stride_x + bn * stride_y;
                int weight_idx = an * weight_stride_x + bn;
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
    const int gridnum_x = gridnums[0];
    const int gridnum_y = gridnums[1];
    const int gridnum_z = gridnums[2];
    const bool periodic_x = axis_periodic(periodic, 0);
    const bool periodic_y = axis_periodic(periodic, 1);
    const bool periodic_z = axis_periodic(periodic, 2);
    const float cellSize_x = boxsizes[0] / static_cast<float>(gridnum_x);
    const float cellSize_y = boxsizes[1] / static_cast<float>(gridnum_y);
    const float cellSize_z = boxsizes[2] / static_cast<float>(gridnum_z);
    const float inv_dx_x = 1.0f / cellSize_x;
    const float inv_dx_y = 1.0f / cellSize_y;
    const float inv_dx_z = 1.0f / cellSize_z;
    const int stride_x = gridnum_y * gridnum_z * num_fields;
    const int stride_y = gridnum_z * num_fields;
    const int stride_z = num_fields;
    const int weight_stride_x = gridnum_y * gridnum_z;
    const int weight_stride_y = gridnum_z;
    const int weight_stride_z = 1;

    std::memset(fields,  0, sizeof(float) * gridnum_x * gridnum_y * gridnum_z * num_fields);
    std::memset(weights, 0, sizeof(float) * gridnum_x * gridnum_y * gridnum_z);

    const bool parallel = allow_openmp(use_openmp);
    const int threads = (parallel && omp_threads > 0) ? omp_threads : 0;
    for_each_particle(N, parallel, threads, [&](int n) {
        float pcs_x = pcellsizesHalf[n] * inv_dx_x;
        float pcs_y = pcellsizesHalf[n] * inv_dx_y;
        float pcs_z = pcellsizesHalf[n] * inv_dx_z;
        float V = (2.0f * pcs_x) * (2.0f * pcs_y) * (2.0f * pcs_z);

        float xpos = pos[3 * n + 0] * inv_dx_x;
        float ypos = pos[3 * n + 1] * inv_dx_y;
        float zpos = pos[3 * n + 2] * inv_dx_z;

        int i = static_cast<int>(xpos);
        int j = static_cast<int>(ypos);
        int k = static_cast<int>(zpos);

        int num_left   = i - static_cast<int>(std::round(xpos - pcs_x - 0.5f));
        int num_right  = static_cast<int>(xpos + pcs_x) - i;
        int num_bottom = j - static_cast<int>(std::round(ypos - pcs_y - 0.5f));
        int num_top    = static_cast<int>(ypos + pcs_y) - j;
        int num_back   = k - static_cast<int>(std::round(zpos - pcs_z - 0.5f));
        int num_fwd    = static_cast<int>(zpos + pcs_z) - k;

        float c1 = xpos - pcs_x, c2 = xpos + pcs_x;
        float c3 = ypos - pcs_y, c4 = ypos + pcs_y;
        float c5 = zpos - pcs_z, c6 = zpos + pcs_z;
        const float* particle = quantities + n * num_fields;

        for (int a = i - num_left; a <= i + num_right; ++a) {
            for (int b = j - num_bottom; b <= j + num_top; ++b) {
                for (int c = k - num_back; c <= k + num_fwd; ++c) {
                    float e1 = static_cast<float>(a);
                    float e2 = e1 + 1.0f;
                    float e3 = static_cast<float>(b);
                    float e4 = e3 + 1.0f;
                    float e5 = static_cast<float>(c);
                    float e6 = e5 + 1.0f;

                    float intersec_x = std::fmin(e2, c2) - std::fmax(e1, c1);
                    float intersec_y = std::fmin(e4, c4) - std::fmax(e3, c3);
                    float intersec_z = std::fmin(e6, c6) - std::fmax(e5, c5);
                    float fraction = (intersec_x * intersec_y * intersec_z) / V;

                    if (fraction <= 0.0f) continue;

                    int an = a, bn = b, cn = c;
                    if (periodic_x) {
                        an = apply_pbc(a, gridnum_x);
                    } else if (a < 0 || a >= gridnum_x) {
                        continue;
                    }
                    if (periodic_y) {
                        bn = apply_pbc(b, gridnum_y);
                    } else if (b < 0 || b >= gridnum_y) {
                        continue;
                    }
                    if (periodic_z) {
                        cn = apply_pbc(c, gridnum_z);
                    } else if (c < 0 || c >= gridnum_z) {
                        continue;
                    }

                    int base_idx = an * stride_x + bn * stride_y + cn * stride_z;
                    int weight_idx = an * weight_stride_x + bn * weight_stride_y + cn * weight_stride_z;
                    accumulate_fields(fields, base_idx, particle, num_fields, fraction, parallel);
                    accumulate_weight(weights, weight_idx, fraction, parallel);
                }
            }
        }
    });
}


// TSC weight for a single offset distance
inline void tsc_weights(float d, float w[3]) {
    // weights for neighbor offsets -1,0,+1
    w[0] = 0.5f * (1.5f - d) * (1.5f - d);
    w[1] = 0.75f - (d - 1.0f) * (d - 1.0f);
    w[2] = 0.5f * (d - 0.5f) * (d - 0.5f);
}

// Triangular Shaped Cloud deposition in 2D
void tsc_2d_cpp(
    const float* pos,        // (N,2)
    const float* quantities, // (N, num_fields)
    int N,
    int num_fields,
    const float* boxsizes,
    const int* gridnums,
    const bool* periodic,
    bool use_openmp,
    int omp_threads,
    float* fields,           // (gridnum_x, gridnum_y, num_fields)
    float* weights           // (gridnum_x, gridnum_y)
) {
    const int gridnum_x = gridnums[0];
    const int gridnum_y = gridnums[1];
    const bool periodic_x = axis_periodic(periodic, 0);
    const bool periodic_y = axis_periodic(periodic, 1);
    const float inv_dx_x = static_cast<float>(gridnum_x) / boxsizes[0];
    const float inv_dx_y = static_cast<float>(gridnum_y) / boxsizes[1];

    // Strides for C-contiguous layout (x, y, f)
    const int stride_x = gridnum_y * num_fields;
    const int stride_y = num_fields;

    // Zero output arrays
    std::memset(fields,  0, sizeof(float) * gridnum_x * gridnum_y * num_fields);
    std::memset(weights, 0, sizeof(float) * gridnum_x * gridnum_y);

    // Neighbor offsets
    const int offsets[3] = {-1, 0, 1};

    const bool parallel = allow_openmp(use_openmp);
    const int threads = (parallel && omp_threads > 0) ? omp_threads : 0;
    for_each_particle(N, parallel, threads, [&](int n) {
        float xpos = pos[2 * n + 0] * inv_dx_x;
        float ypos = pos[2 * n + 1] * inv_dx_y;

        int i_base = static_cast<int>(std::floor(xpos));
        int j_base = static_cast<int>(std::floor(ypos));

        float dx = xpos - i_base;
        float dy = ypos - j_base;

        float wx[3], wy[3];
        tsc_weights(dx, wx);
        tsc_weights(dy, wy);
        const float* particle = quantities + n * num_fields;

        for (int dx_i = 0; dx_i < 3; ++dx_i) {
            for (int dy_i = 0; dy_i < 3; ++dy_i) {
                int ix = i_base + offsets[dx_i];
                int iy = j_base + offsets[dy_i];

                if (periodic_x) {
                    ix = apply_pbc(ix, gridnum_x);
                } else if (ix < 0 || ix >= gridnum_x) {
                    continue;
                }

                if (periodic_y) {
                    iy = apply_pbc(iy, gridnum_y);
                } else if (iy < 0 || iy >= gridnum_y) {
                    continue;
                }

                float w = wx[dx_i] * wy[dy_i];
                if (w == 0.0f) continue;

                int base_idx   = ix * stride_x + iy * stride_y;
                int weight_idx = ix * gridnum_y + iy;
                accumulate_fields(fields, base_idx, particle, num_fields, w, parallel);
                accumulate_weight(weights, weight_idx, w, parallel);
            }
        }
    });
}

void tsc_3d_cpp(
    const float* pos,        // (N,3)
    const float* quantities, // (N,num_fields)
    int N,
    int num_fields,
    const float* boxsizes,
    const int* gridnums,
    const bool* periodic,
    bool use_openmp,
    int omp_threads,
    float* fields,           // (gridnum_x, gridnum_y, gridnum_z, num_fields)
    float* weights           // (gridnum_x, gridnum_y, gridnum_z)
) {
    const int gridnum_x = gridnums[0];
    const int gridnum_y = gridnums[1];
    const int gridnum_z = gridnums[2];
    const bool periodic_x = axis_periodic(periodic, 0);
    const bool periodic_y = axis_periodic(periodic, 1);
    const bool periodic_z = axis_periodic(periodic, 2);
    const float inv_dx_x = static_cast<float>(gridnum_x) / boxsizes[0];
    const float inv_dx_y = static_cast<float>(gridnum_y) / boxsizes[1];
    const float inv_dx_z = static_cast<float>(gridnum_z) / boxsizes[2];

    // Strides for C-contiguous layout (x, y, z, f)
    const int stride_x = gridnum_y * gridnum_z * num_fields;
    const int stride_y = gridnum_z * num_fields;
    const int stride_z = num_fields;

    // Zero output arrays
    std::memset(fields,  0, sizeof(float) * gridnum_x * gridnum_y * gridnum_z * num_fields);
    std::memset(weights, 0, sizeof(float) * gridnum_x * gridnum_y * gridnum_z);

    // Neighbor offsets
    const int offsets[3] = {-1, 0, 1};

    const bool parallel = allow_openmp(use_openmp);
    const int threads = (parallel && omp_threads > 0) ? omp_threads : 0;
    for_each_particle(N, parallel, threads, [&](int n) {
        float xpos = pos[3 * n + 0] * inv_dx_x;
        float ypos = pos[3 * n + 1] * inv_dx_y;
        float zpos = pos[3 * n + 2] * inv_dx_z;

        int i_base = static_cast<int>(std::floor(xpos));
        int j_base = static_cast<int>(std::floor(ypos));
        int k_base = static_cast<int>(std::floor(zpos));

        float dx = xpos - i_base;
        float dy = ypos - j_base;
        float dz = zpos - k_base;

        float wx[3], wy[3], wz[3];
        tsc_weights(dx, wx);
        tsc_weights(dy, wy);
        tsc_weights(dz, wz);
        const float* particle = quantities + n * num_fields;

        for (int dx_i = 0; dx_i < 3; ++dx_i) {
            for (int dy_i = 0; dy_i < 3; ++dy_i) {
                for (int dz_i = 0; dz_i < 3; ++dz_i) {
                    int ix = i_base + offsets[dx_i];
                    int iy = j_base + offsets[dy_i];
                    int iz = k_base + offsets[dz_i];

                    if (periodic_x) {
                        ix = apply_pbc(ix, gridnum_x);
                    } else if (ix < 0 || ix >= gridnum_x) {
                        continue;
                    }
                    if (periodic_y) {
                        iy = apply_pbc(iy, gridnum_y);
                    } else if (iy < 0 || iy >= gridnum_y) {
                        continue;
                    }
                    if (periodic_z) {
                        iz = apply_pbc(iz, gridnum_z);
                    } else if (iz < 0 || iz >= gridnum_z) {
                        continue;
                    }

                    float w = wx[dx_i] * wy[dy_i] * wz[dz_i];
                    if (w == 0.0f) continue;

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
    const float* pos,        // (N,2)
    const float* quantities, // (N,num_fields)
    int N,
    int num_fields,
    const float* boxsizes,
    const int* gridnums,
    const bool* periodic,
    const float* pcellsizesHalf, // (N)
    bool use_openmp,
    int omp_threads,
    float* fields,           // (gridnum_x, gridnum_y, num_fields)
    float* weights           // (gridnum_x, gridnum_y)
) {
    const int gridnum_x = gridnums[0];
    const int gridnum_y = gridnums[1];
    const bool periodic_x = axis_periodic(periodic, 0);
    const bool periodic_y = axis_periodic(periodic, 1);
    const float cellSize_x = boxsizes[0] / static_cast<float>(gridnum_x);
    const float cellSize_y = boxsizes[1] / static_cast<float>(gridnum_y);
    const float inv_dx_x = 1.0f / cellSize_x;
    const float inv_dx_y = 1.0f / cellSize_y;

    const int stride_x = gridnum_y * num_fields;
    const int stride_y = num_fields;
    const int weight_stride_x = gridnum_y;
    const int weight_stride_y = 1;

    std::memset(fields,  0, sizeof(float) * gridnum_x * gridnum_y * num_fields);
    std::memset(weights, 0, sizeof(float) * gridnum_x * gridnum_y);

    const bool parallel = allow_openmp(use_openmp);
    const int threads = (parallel && omp_threads > 0) ? omp_threads : 0;
    for_each_particle(N, parallel, threads, [&](int n) {
        float x = pos[2 * n + 0] * inv_dx_x;
        float y = pos[2 * n + 1] * inv_dx_y;
        float h_x = pcellsizesHalf[n] * inv_dx_x;
        float h_y = pcellsizesHalf[n] * inv_dx_y;

        float support_x = 1.5f * h_x;
        float support_y = 1.5f * h_y;
        int i_min = static_cast<int>(std::floor(x - support_x));
        int i_max = static_cast<int>(std::ceil(x + support_x));
        int j_min = static_cast<int>(std::floor(y - support_y));
        int j_max = static_cast<int>(std::ceil(y + support_y));
        const float* particle = quantities + n * num_fields;

        for (int i = i_min; i <= i_max; ++i) {
            int ii = i;
            if (periodic_x) ii = apply_pbc(i, gridnum_x);
            else if (ii < 0 || ii >= gridnum_x) continue;

            float wx = tsc_integrated_weight_1d(x, float(i), float(i+1), h_x);
            if (wx == 0.0f) continue;

            for (int j = j_min; j <= j_max; ++j) {
                int jj = j;
                if (periodic_y) jj = apply_pbc(j, gridnum_y);
                else if (jj < 0 || jj >= gridnum_y) continue;

                float wy = tsc_integrated_weight_1d(y, float(j), float(j+1), h_y);
                if (wy == 0.0f) continue;

                float w = wx * wy;
                int base_idx = ii * stride_x + jj * stride_y;
                int weight_idx = ii * weight_stride_x + jj * weight_stride_y;
                accumulate_fields(fields, base_idx, particle, num_fields, w, parallel);
                accumulate_weight(weights, weight_idx, w, parallel);
            }
        }
    });
}

void tsc_3d_adaptive_cpp(
    const float* pos,        // (N,3)
    const float* quantities, // (N,num_fields)
    int N,
    int num_fields,
    const float* boxsizes,
    const int* gridnums,
    const bool* periodic,
    const float* pcellsizesHalf, // (N)
    bool use_openmp,
    int omp_threads,
    float* fields,           // (gridnum_x, gridnum_y, gridnum_z, num_fields)
    float* weights           // (gridnum_x, gridnum_y, gridnum_z)
) {
    const int gridnum_x = gridnums[0];
    const int gridnum_y = gridnums[1];
    const int gridnum_z = gridnums[2];
    const bool periodic_x = axis_periodic(periodic, 0);
    const bool periodic_y = axis_periodic(periodic, 1);
    const bool periodic_z = axis_periodic(periodic, 2);
    const float cellSize_x = boxsizes[0] / static_cast<float>(gridnum_x);
    const float cellSize_y = boxsizes[1] / static_cast<float>(gridnum_y);
    const float cellSize_z = boxsizes[2] / static_cast<float>(gridnum_z);
    const float inv_dx_x = 1.0f / cellSize_x;
    const float inv_dx_y = 1.0f / cellSize_y;
    const float inv_dx_z = 1.0f / cellSize_z;

    const int stride_x = gridnum_y * gridnum_z * num_fields;
    const int stride_y = gridnum_z * num_fields;
    const int stride_z = num_fields;

    const int weight_stride_x = gridnum_y * gridnum_z;
    const int weight_stride_y = gridnum_z;
    const int weight_stride_z = 1;

    std::memset(fields,  0, sizeof(float) * gridnum_x * gridnum_y * gridnum_z * num_fields);
    std::memset(weights, 0, sizeof(float) * gridnum_x * gridnum_y * gridnum_z);

    const bool parallel = allow_openmp(use_openmp);
    const int threads = (parallel && omp_threads > 0) ? omp_threads : 0;
    for_each_particle(N, parallel, threads, [&](int n) {
        float x = pos[3 * n + 0] * inv_dx_x;
        float y = pos[3 * n + 1] * inv_dx_y;
        float z = pos[3 * n + 2] * inv_dx_z;
        float h_x = pcellsizesHalf[n] * inv_dx_x;
        float h_y = pcellsizesHalf[n] * inv_dx_y;
        float h_z = pcellsizesHalf[n] * inv_dx_z;

        float support_x = 1.5f * h_x;
        float support_y = 1.5f * h_y;
        float support_z = 1.5f * h_z;
        int i_min = static_cast<int>(std::floor(x - support_x));
        int i_max = static_cast<int>(std::ceil(x + support_x));
        int j_min = static_cast<int>(std::floor(y - support_y));
        int j_max = static_cast<int>(std::ceil(y + support_y));
        int k_min = static_cast<int>(std::floor(z - support_z));
        int k_max = static_cast<int>(std::ceil(z + support_z));
        const float* particle = quantities + n * num_fields;

        for (int i = i_min; i <= i_max; ++i) {
            int ii = i;
            if (periodic_x) ii = apply_pbc(i, gridnum_x);
            else if (ii < 0 || ii >= gridnum_x) continue;
            float wx = tsc_integrated_weight_1d(x, float(i), float(i+1), h_x);
            if (wx == 0.0f) continue;

            for (int j = j_min; j <= j_max; ++j) {
                int jj = j;
                if (periodic_y) jj = apply_pbc(j, gridnum_y);
                else if (jj < 0 || jj >= gridnum_y) continue;
                float wy = tsc_integrated_weight_1d(y, float(j), float(j+1), h_y);
                if (wy == 0.0f) continue;

                for (int k = k_min; k <= k_max; ++k) {
                    int kk = k;
                    if (periodic_z) kk = apply_pbc(k, gridnum_z);
                    else if (kk < 0 || kk >= gridnum_z) continue;
                    float wz = tsc_integrated_weight_1d(z, float(k), float(k+1), h_z);
                    if (wz == 0.0f) continue;

                    float w = wx * wy * wz;

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
// SPH isotropic kernel deposition (2D)
// =============================================================================

static float compute_fraction_isotropic_2d_cpp(
    const std::string& method,
    float xpos,
    float ypos,
    int a,
    int b,
    int gridnum_x,
    int gridnum_y,
    const bool* periodic,
    float h,
    SPHKernel* kernel
) {
    float sigma = kernel->normalization(h);
    const bool periodic_x = axis_periodic(periodic, 0);
    const bool periodic_y = axis_periodic(periodic, 1);

    if (method == "midpoint") {
        float dx = xpos - (a + 0.5f);
        float dy = ypos - (b + 0.5f);

        if (periodic_x) {
            if (dx > gridnum_x / 2.0f) dx -= gridnum_x;
            if (dx < -gridnum_x / 2.0f) dx += gridnum_x;
        }
        if (periodic_y) {
            if (dy > gridnum_y / 2.0f) dy -= gridnum_y;
            if (dy < -gridnum_y / 2.0f) dy += gridnum_y;
        }

        float r = std::sqrt(dx * dx + dy * dy);
        return kernel->weight(r, h) * sigma;
    }

    if (method == "trapezoidal") {
        const float offsets[4][2] = {
            {0.0f, 0.5f}, {1.0f, 0.5f},
            {0.5f, 0.0f}, {0.5f, 1.0f}
        };

        float sum = 0.0f;
        for (int i = 0; i < 4; ++i) {
            float dx = xpos - (a + offsets[i][0]);
            float dy = ypos - (b + offsets[i][1]);

            if (periodic_x) {
                if (dx > gridnum_x / 2.0f) dx -= gridnum_x;
                if (dx < -gridnum_x / 2.0f) dx += gridnum_x;
            }
            if (periodic_y) {
                if (dy > gridnum_y / 2.0f) dy -= gridnum_y;
                if (dy < -gridnum_y / 2.0f) dy += gridnum_y;
            }

            float r = std::sqrt(dx * dx + dy * dy);
            sum += kernel->weight(r, h);
        }

        return (sum / 4.0f) * sigma;
    }

    if (method == "simpson") {
        float sum = 0.0f;

        {
            float dx = xpos - (a + 0.5f);
            float dy = ypos - (b + 0.5f);

            if (periodic_x) {
                if (dx > gridnum_x / 2.0f) dx -= gridnum_x;
                if (dx < -gridnum_x / 2.0f) dx += gridnum_x;
            }
            if (periodic_y) {
                if (dy > gridnum_y / 2.0f) dy -= gridnum_y;
                if (dy < -gridnum_y / 2.0f) dy += gridnum_y;
            }

            float r = std::sqrt(dx * dx + dy * dy);
            sum += 4.0f * kernel->weight(r, h);
        }

        const float edge_offsets[4][2] = {
            {0.0f, 0.5f}, {1.0f, 0.5f},
            {0.5f, 0.0f}, {0.5f, 1.0f}
        };
        for (int i = 0; i < 4; ++i) {
            float dx = xpos - (a + edge_offsets[i][0]);
            float dy = ypos - (b + edge_offsets[i][1]);

                if (periodic_x) {
                    if (dx > gridnum_x / 2.0f) dx -= gridnum_x;
                    if (dx < -gridnum_x / 2.0f) dx += gridnum_x;
                }
                if (periodic_y) {
                    if (dy > gridnum_y / 2.0f) dy -= gridnum_y;
                    if (dy < -gridnum_y / 2.0f) dy += gridnum_y;
                }

            float r = std::sqrt(dx * dx + dy * dy);
            sum += 2.0f * kernel->weight(r, h);
        }

        for (int dx_c = 0; dx_c <= 1; ++dx_c) {
            for (int dy_c = 0; dy_c <= 1; ++dy_c) {
                float dx = xpos - (a + dx_c);
                float dy = ypos - (b + dy_c);

                    if (periodic_x) {
                        if (dx > gridnum_x / 2.0f) dx -= gridnum_x;
                        if (dx < -gridnum_x / 2.0f) dx += gridnum_x;
                    }
                    if (periodic_y) {
                        if (dy > gridnum_y / 2.0f) dy -= gridnum_y;
                        if (dy < -gridnum_y / 2.0f) dy += gridnum_y;
                    }

                float r = std::sqrt(dx * dx + dy * dy);
                sum += kernel->weight(r, h);
            }
        }

        return (sum / 16.0f) * sigma;
    }

    throw std::invalid_argument("Unknown integration method: " + method);
}


void isotropic_kernel_deposition_2d_cpp(
    const float* pos,          // (N, 2)
    const float* quantities,   // (N, num_fields)
    const float* hsm,          // (N)
    int N,
    int num_fields,
    const float* boxsizes,
    const int* gridnums,
    const bool* periodic,
    const std::string& kernel_name,
    const std::string& integration_method,
    bool use_openmp,
    int omp_threads,
    float* fields,             // (gridnum_x, gridnum_y, num_fields)
    float* weights             // (gridnum_x, gridnum_y)
) {
    auto kernel = create_kernel(kernel_name, 2, false);
    const int gridnum_x = gridnums[0];
    const int gridnum_y = gridnums[1];
    const bool periodic_x = axis_periodic(periodic, 0);
    const bool periodic_y = axis_periodic(periodic, 1);
    const float cellSize_x = boxsizes[0] / static_cast<float>(gridnum_x);
    const float cellSize_y = boxsizes[1] / static_cast<float>(gridnum_y);
    const float max_cell = std::max({1.0f, std::abs(cellSize_x), std::abs(cellSize_y)});
    if (std::abs(cellSize_x - cellSize_y) > 1e-6f * max_cell) {
        throw std::invalid_argument("isotropic_kernel_deposition_2d_cpp requires uniform cell sizes");
    }
    const float cellSize = cellSize_x;
    const float support_factor = kernel->support();

    const int stride_x = gridnum_y * num_fields;
    const int stride_y = num_fields;
    const int weight_stride_x = gridnum_y;

    std::memset(fields,  0, sizeof(float) * gridnum_x * gridnum_y * num_fields);
    std::memset(weights, 0, sizeof(float) * gridnum_x * gridnum_y);

    const bool parallel = allow_openmp(use_openmp);
    const int threads = (parallel && omp_threads > 0) ? omp_threads : 0;
    SPHKernel* kernel_ptr = kernel.get();
    for_each_particle(N, parallel, threads, [&](int n) {
        float hsn = hsm[n] / cellSize;
        float support = support_factor * hsn;

        float xpos = pos[2 * n + 0] / cellSize;
        float ypos = pos[2 * n + 1] / cellSize;

        int i = static_cast<int>(xpos);
        int j = static_cast<int>(ypos);

        int num_left   = i - static_cast<int>(xpos - support);
        int num_right  = static_cast<int>(xpos + support + 0.5f) - i;
        int num_bottom = j - static_cast<int>(ypos - support);
        int num_top    = static_cast<int>(ypos + support + 0.5f) - j;
        const float* particle = quantities + n * num_fields;

        for (int a = i - num_left; a <= i + num_right; ++a) {
            for (int b = j - num_bottom; b <= j + num_top; ++b) {
                float w = compute_fraction_isotropic_2d_cpp(
                    integration_method, xpos, ypos,
                    a, b, gridnum_x, gridnum_y, periodic, hsn, kernel_ptr
                );

                if (w == 0.0f) continue;

                int an = a;
                int bn = b;
                if (periodic_x) {
                    an = apply_pbc(an, gridnum_x);
                } else if (an < 0 || an >= gridnum_x) {
                    continue;
                }

                if (periodic_y) {
                    bn = apply_pbc(bn, gridnum_y);
                } else if (bn < 0 || bn >= gridnum_y) {
                    continue;
                }

                int base_idx = an * stride_x + bn * stride_y;
                int weight_idx = an * weight_stride_x + bn;
                accumulate_fields(fields, base_idx, particle, num_fields, w, parallel);
                accumulate_weight(weights, weight_idx, w, parallel);
            }
        }
    });
}


// =============================================================================
// SPH isotropic kernel deposition (3D)
// =============================================================================

static float compute_fraction_isotropic_3d_cpp(
    const std::string& method,
    float xpos,
    float ypos,
    float zpos,
    int a,
    int b,
    int c,
    int gridnum_x,
    int gridnum_y,
    int gridnum_z,
    const bool* periodic,
    float h,
    SPHKernel* kernel
) {
    float sigma = kernel->normalization(h);
    const bool periodic_x = axis_periodic(periodic, 0);
    const bool periodic_y = axis_periodic(periodic, 1);
    const bool periodic_z = axis_periodic(periodic, 2);

    auto wrap_axis = [&](float& delta, int gridnum, bool axis_flag) {
        if (!axis_flag) return;
        if (delta > gridnum / 2.0f) delta -= gridnum;
        if (delta < -gridnum / 2.0f) delta += gridnum;
    };

    if (method == "midpoint") {
        float dx = xpos - (a + 0.5f);
        float dy = ypos - (b + 0.5f);
        float dz = zpos - (c + 0.5f);

        wrap_axis(dx, gridnum_x, periodic_x);
        wrap_axis(dy, gridnum_y, periodic_y);
        wrap_axis(dz, gridnum_z, periodic_z);

        float r = std::sqrt(dx * dx + dy * dy + dz * dz);
        return kernel->weight(r, h) * sigma;
    }

    if (method == "trapezoidal") {
        const float offsets[6][3] = {
            {0.0f, 0.5f, 0.5f}, {1.0f, 0.5f, 0.5f},
            {0.5f, 0.0f, 0.5f}, {0.5f, 1.0f, 0.5f},
            {0.5f, 0.5f, 0.0f}, {0.5f, 0.5f, 1.0f}
        };
        float sum = 0.0f;
        for (int i = 0; i < 6; ++i) {
            float dx = xpos - (a + offsets[i][0]);
            float dy = ypos - (b + offsets[i][1]);
            float dz = zpos - (c + offsets[i][2]);
            wrap_axis(dx, gridnum_x, periodic_x);
            wrap_axis(dy, gridnum_y, periodic_y);
            wrap_axis(dz, gridnum_z, periodic_z);

            float r = std::sqrt(dx * dx + dy * dy + dz * dz);
            sum += kernel->weight(r, h);
        }

        return (sum / 6.0f) * sigma;
    }

    if (method == "simpson") {
        float sum = 0.0f;

        {
            float dx = xpos - (a + 0.5f);
            float dy = ypos - (b + 0.5f);
            float dz = zpos - (c + 0.5f);
            wrap_axis(dx, gridnum_x, periodic_x);
            wrap_axis(dy, gridnum_y, periodic_y);
            wrap_axis(dz, gridnum_z, periodic_z);
            float r = std::sqrt(dx * dx + dy * dy + dz * dz);
            sum += 8.0f * kernel->weight(r, h);
        }

        const float face_offsets[6][3] = {
            {0.0f, 0.5f, 0.5f}, {1.0f, 0.5f, 0.5f},
            {0.5f, 0.0f, 0.5f}, {0.5f, 1.0f, 0.5f},
            {0.5f, 0.5f, 0.0f}, {0.5f, 0.5f, 1.0f}
        };
        for (int i = 0; i < 6; ++i) {
            float dx = xpos - (a + face_offsets[i][0]);
            float dy = ypos - (b + face_offsets[i][1]);
            float dz = zpos - (c + face_offsets[i][2]);
            wrap_axis(dx, gridnum_x, periodic_x);
            wrap_axis(dy, gridnum_y, periodic_y);
            wrap_axis(dz, gridnum_z, periodic_z);
            float r = std::sqrt(dx * dx + dy * dy + dz * dz);
            sum += 4.0f * kernel->weight(r, h);
        }

        for (int dx_c = 0; dx_c <= 1; ++dx_c) {
            for (int dy_c = 0; dy_c <= 1; ++dy_c) {
                for (int dz_c = 0; dz_c <= 1; ++dz_c) {
                    float dx = xpos - (a + dx_c);
                    float dy = ypos - (b + dy_c);
                    float dz = zpos - (c + dz_c);
                    wrap_axis(dx, gridnum_x, periodic_x);
                    wrap_axis(dy, gridnum_y, periodic_y);
                    wrap_axis(dz, gridnum_z, periodic_z);
                    float r = std::sqrt(dx * dx + dy * dy + dz * dz);
                    sum += kernel->weight(r, h);
                }
            }
        }

        return (sum / 40.0f) * sigma;
    }

    throw std::invalid_argument("Unknown integration method: " + method);
}


void isotropic_kernel_deposition_3d_cpp(
    const float* pos,          // (N, 3)
    const float* quantities,   // (N, num_fields)
    const float* hsm,          // (N)
    int N,
    int num_fields,
    const float* boxsizes,
    const int* gridnums,
    const bool* periodic,
    const std::string& kernel_name,
    const std::string& integration_method,
    bool use_openmp,
    int omp_threads,
    float* fields,             // (gridnum_x, gridnum_y, gridnum_z, num_fields)
    float* weights             // (gridnum_x, gridnum_y, gridnum_z)
) {
    auto kernel = create_kernel(kernel_name, 3, false);
    const int gridnum_x = gridnums[0];
    const int gridnum_y = gridnums[1];
    const int gridnum_z = gridnums[2];
    const bool periodic_x = axis_periodic(periodic, 0);
    const bool periodic_y = axis_periodic(periodic, 1);
    const bool periodic_z = axis_periodic(periodic, 2);
    const float cellSize_x = boxsizes[0] / static_cast<float>(gridnum_x);
    const float cellSize_y = boxsizes[1] / static_cast<float>(gridnum_y);
    const float cellSize_z = boxsizes[2] / static_cast<float>(gridnum_z);
    const float max_cell = std::max({1.0f, std::abs(cellSize_x), std::abs(cellSize_y), std::abs(cellSize_z)});
    if (std::abs(cellSize_x - cellSize_y) > 1e-6f * max_cell ||
        std::abs(cellSize_x - cellSize_z) > 1e-6f * max_cell) {
        throw std::invalid_argument("isotropic_kernel_deposition_3d_cpp requires uniform cell sizes");
    }
    const float cellSize = cellSize_x;
    const float support_factor = kernel->support();

    const int stride_x = gridnum_y * gridnum_z * num_fields;
    const int stride_y = gridnum_z * num_fields;
    const int stride_z = num_fields;
    const int weight_stride_x = gridnum_y * gridnum_z;
    const int weight_stride_y = gridnum_z;

    std::memset(fields,  0, sizeof(float) * gridnum_x * gridnum_y * gridnum_z * num_fields);
    std::memset(weights, 0, sizeof(float) * gridnum_x * gridnum_y * gridnum_z);

    const bool parallel = allow_openmp(use_openmp);
    const int threads = (parallel && omp_threads > 0) ? omp_threads : 0;
    SPHKernel* kernel_ptr = kernel.get();
    for_each_particle(N, parallel, threads, [&](int n) {
        float hsn = hsm[n] / cellSize;
        float support = support_factor * hsn;

        float xpos = pos[3 * n + 0] / cellSize;
        float ypos = pos[3 * n + 1] / cellSize;
        float zpos = pos[3 * n + 2] / cellSize;

        int i = static_cast<int>(xpos);
        int j = static_cast<int>(ypos);
        int k = static_cast<int>(zpos);

        int num_left   = i - static_cast<int>(xpos - support);
        int num_right  = static_cast<int>(xpos + support + 0.5f) - i;
        int num_bottom = j - static_cast<int>(ypos - support);
        int num_top    = static_cast<int>(ypos + support + 0.5f) - j;
        int num_front  = k - static_cast<int>(zpos - support);
        int num_back   = static_cast<int>(zpos + support + 0.5f) - k;
        const float* particle = quantities + n * num_fields;

        for (int a = i - num_left; a <= i + num_right; ++a) {
            for (int b = j - num_bottom; b <= j + num_top; ++b) {
                for (int c = k - num_front; c <= k + num_back; ++c) {
                    float w = compute_fraction_isotropic_3d_cpp(
                        integration_method, xpos, ypos, zpos,
                        a, b, c, gridnum_x, gridnum_y, gridnum_z, periodic, hsn, kernel_ptr
                    );

                    if (w == 0.0f) continue;

                    int an = a;
                    int bn = b;
                    int cn = c;
                    if (periodic_x) {
                        an = apply_pbc(an, gridnum_x);
                    } else if (an < 0 || an >= gridnum_x) {
                        continue;
                    }

                    if (periodic_y) {
                        bn = apply_pbc(bn, gridnum_y);
                    } else if (bn < 0 || bn >= gridnum_y) {
                        continue;
                    }

                    if (periodic_z) {
                        cn = apply_pbc(cn, gridnum_z);
                    } else if (cn < 0 || cn >= gridnum_z) {
                        continue;
                    }

                    int base_idx = an * stride_x + bn * stride_y + cn * stride_z;
                    int weight_idx = an * weight_stride_x + bn * weight_stride_y + cn;
                    accumulate_fields(fields, base_idx, particle, num_fields, w, parallel);
                    accumulate_weight(weights, weight_idx, w, parallel);
                }
            }
        }
    });
}


// =============================================================================
// SPH anisotropic kernel deposition (2D)
// =============================================================================

static float compute_fraction_anisotropic_2d_cpp(
    const std::string& method,
    const float* vecs,
    const float* vals_gu,
    float xpos,
    float ypos,
    int a,
    int b,
    int gridnum_x,
    int gridnum_y,
    const bool* periodic,
    SPHKernel* kernel
) {
    float detH = vals_gu[0] * vals_gu[1];
    float sigma = kernel->normalization(detH);
    const bool periodic_x = axis_periodic(periodic, 0);
    const bool periodic_y = axis_periodic(periodic, 1);

    auto wrap_axis = [&](float& delta, int gridnum, bool axis_flag) {
        if (!axis_flag) return;
        if (delta > gridnum * 0.5f) delta -= gridnum;
        if (delta < -gridnum * 0.5f) delta += gridnum;
    };

    auto eval = [&](float ox, float oy) {
        float dx = xpos - (a + ox);
        float dy = ypos - (b + oy);
        wrap_axis(dx, gridnum_x, periodic_x);
        wrap_axis(dy, gridnum_y, periodic_y);

        float xi1 = (vecs[0] * dx + vecs[1] * dy) / vals_gu[0];
        float xi2 = (vecs[2] * dx + vecs[3] * dy) / vals_gu[1];
        float q = std::sqrt(xi1 * xi1 + xi2 * xi2);
        return kernel->weight(q, 1.0f);
    };

    if (method == "midpoint") {
        return eval(0.5f, 0.5f) * sigma;
    }

    if (method == "trapezoidal") {
        float sum = 0.0f;
        sum += eval(0.0f, 0.0f);
        sum += eval(1.0f, 0.0f);
        sum += eval(0.0f, 1.0f);
        sum += eval(1.0f, 1.0f);
        return (sum / 4.0f) * sigma;
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

        return (sum / 36.0f) * sigma;
    }

    throw std::invalid_argument("Unknown integration method: " + method);
}

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
) {
    auto kernel = create_kernel(kernel_name, 2, true);

    const int gridnum_x = gridnums[0];
    const int gridnum_y = gridnums[1];
    const bool periodic_x = axis_periodic(periodic, 0);
    const bool periodic_y = axis_periodic(periodic, 1);
    const float cellSize_x = boxsizes[0] / static_cast<float>(gridnum_x);
    const float cellSize_y = boxsizes[1] / static_cast<float>(gridnum_y);
    const float max_cell = std::max({1.0f, std::abs(cellSize_x), std::abs(cellSize_y)});
    if (std::abs(cellSize_x - cellSize_y) > 1e-6f * max_cell) {
        throw std::invalid_argument("anisotropic_kernel_deposition_2d_cpp requires uniform cell sizes");
    }
    const float cellSize = cellSize_x;
    const float support_factor = kernel->support();

    const int stride_x = gridnum_y * num_fields;
    const int stride_y = num_fields;
    const int weight_stride_x = gridnum_y;

    std::memset(fields,  0, sizeof(float) * gridnum_x * gridnum_y * num_fields);
    std::memset(weights, 0, sizeof(float) * gridnum_x * gridnum_y);

    const bool parallel = allow_openmp(use_openmp);
    const int threads = (parallel && omp_threads > 0) ? omp_threads : 0;
    SPHKernel* kernel_ptr = kernel.get();
    for_each_particle(N, parallel, threads, [&](int n) {
        const float* vecs = &hmat_eigvecs[n * 4];
        const float* vals = &hmat_eigvals[n * 2];

        float vals_gu[2] = { vals[0] / cellSize, vals[1] / cellSize };
        float krs = support_factor * std::max({ vals_gu[0], vals_gu[1] });

        float xpos = pos[2 * n + 0] / cellSize;
        float ypos = pos[2 * n + 1] / cellSize;

        int i = static_cast<int>(xpos);
        int j = static_cast<int>(ypos);

        int num_left   = i - static_cast<int>(xpos - krs);
        int num_right  = static_cast<int>(xpos + krs + 0.5f) - i;
        int num_bottom = j - static_cast<int>(ypos - krs);
        int num_top    = static_cast<int>(ypos + krs + 0.5f) - j;
        const float* particle = quantities + n * num_fields;

        for (int a = i - num_left; a <= i + num_right; ++a) {
            for (int b = j - num_bottom; b <= j + num_top; ++b) {
                float fraction = compute_fraction_anisotropic_2d_cpp(
                    integration_method, vecs, vals_gu,
                    xpos, ypos, a, b,
                    gridnum_x, gridnum_y,
                    periodic, kernel_ptr
                );

                if (fraction == 0.0f) continue;

                int an = a;
                int bn = b;
                if (periodic_x) {
                    an = apply_pbc(an, gridnum_x);
                } else if (an < 0 || an >= gridnum_x) {
                    continue;
                }

                if (periodic_y) {
                    bn = apply_pbc(bn, gridnum_y);
                } else if (bn < 0 || bn >= gridnum_y) {
                    continue;
                }

                int base_idx = an * stride_x + bn * stride_y;
                int weight_idx = an * weight_stride_x + bn;
                accumulate_fields(fields, base_idx, particle, num_fields, fraction, parallel);
                accumulate_weight(weights, weight_idx, fraction, parallel);
            }
        }
    });
}


// =============================================================================
// SPH anisotropic kernel deposition (3D)
// =============================================================================

static float compute_fraction_anisotropic_3d_cpp(
    const std::string& method,
    const float* vecs,
    const float* vals_gu,
    float xpos,
    float ypos,
    float zpos,
    int a,
    int b,
    int c,
    int gridnum_x,
    int gridnum_y,
    int gridnum_z,
    const bool* periodic,
    SPHKernel* kernel
) {
    float detH = vals_gu[0] * vals_gu[1] * vals_gu[2];
    float sigma = kernel->normalization(detH);
    const bool periodic_x = axis_periodic(periodic, 0);
    const bool periodic_y = axis_periodic(periodic, 1);
    const bool periodic_z = axis_periodic(periodic, 2);

    auto wrap_axis = [&](float& delta, int gridnum, bool axis_flag) {
        if (!axis_flag) return;
        if (delta > gridnum * 0.5f) delta -= gridnum;
        if (delta < -gridnum * 0.5f) delta += gridnum;
    };

    auto eval = [&](float ox, float oy, float oz) {
        float dx = xpos - (a + ox);
        float dy = ypos - (b + oy);
        float dz = zpos - (c + oz);
        wrap_axis(dx, gridnum_x, periodic_x);
        wrap_axis(dy, gridnum_y, periodic_y);
        wrap_axis(dz, gridnum_z, periodic_z);

        float xi1 = (vecs[0] * dx + vecs[1] * dy + vecs[2] * dz) / vals_gu[0];
        float xi2 = (vecs[3] * dx + vecs[4] * dy + vecs[5] * dz) / vals_gu[1];
        float xi3 = (vecs[6] * dx + vecs[7] * dy + vecs[8] * dz) / vals_gu[2];
        float q = std::sqrt(xi1 * xi1 + xi2 * xi2 + xi3 * xi3);
        return kernel->weight(q, 1.0f);
    };

    if (method == "midpoint") {
        return eval(0.5f, 0.5f, 0.5f) * sigma;
    }

    if (method == "trapezoidal") {
        float sum = 0.0f;
        for (int i = 0; i <= 1; ++i)
            for (int j = 0; j <= 1; ++j)
                for (int k = 0; k <= 1; ++k)
                    sum += eval(i, j, k);
        return (sum / 8.0f) * sigma;
    }

    if (method == "simpson") {
        float sum = 0.0f;

        // corners
        for (int i = 0; i <= 1; ++i)
            for (int j = 0; j <= 1; ++j)
                for (int k = 0; k <= 1; ++k)
                    sum += eval(i, j, k);

        // edge midpoints (12)
        /*
        const int e[12][3] = {
            {0,0,1},{0,1,0},{1,0,0},{1,1,0},
            {1,0,1},{0,1,1},{0,0,1},{1,1,1},
            {0,1,0},{1,0,0},{0,0,0},{1,1,1}
        };
        */

        // simpler explicit loops
        for (int i = 0; i <= 1; ++i)
            for (int j = 0; j <= 1; ++j) {
                sum += 4.0f * eval(0.5f, i, j);
                sum += 4.0f * eval(i, 0.5f, j);
                sum += 4.0f * eval(i, j, 0.5f);
            }

        // face centers (6)
        sum += 16.0f * eval(0.5f, 0.5f, 0.0f);
        sum += 16.0f * eval(0.5f, 0.5f, 1.0f);
        sum += 16.0f * eval(0.5f, 0.0f, 0.5f);
        sum += 16.0f * eval(0.5f, 1.0f, 0.5f);
        sum += 16.0f * eval(0.0f, 0.5f, 0.5f);
        sum += 16.0f * eval(1.0f, 0.5f, 0.5f);

        // center
        sum += 64.0f * eval(0.5f, 0.5f, 0.5f);

        return (sum / 216.0f) * sigma;
    }

    throw std::invalid_argument("Unknown integration method: " + method);
}

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
) {
    auto kernel = create_kernel(kernel_name, 3, true);

    const int gridnum_x = gridnums[0];
    const int gridnum_y = gridnums[1];
    const int gridnum_z = gridnums[2];
    const bool periodic_x = axis_periodic(periodic, 0);
    const bool periodic_y = axis_periodic(periodic, 1);
    const bool periodic_z = axis_periodic(periodic, 2);
    const float cellSize_x = boxsizes[0] / static_cast<float>(gridnum_x);
    const float cellSize_y = boxsizes[1] / static_cast<float>(gridnum_y);
    const float cellSize_z = boxsizes[2] / static_cast<float>(gridnum_z);
    const float max_cell = std::max({1.0f, std::abs(cellSize_x), std::abs(cellSize_y), std::abs(cellSize_z)});
    if (std::abs(cellSize_x - cellSize_y) > 1e-6f * max_cell ||
        std::abs(cellSize_x - cellSize_z) > 1e-6f * max_cell) {
        throw std::invalid_argument("anisotropic_kernel_deposition_3d_cpp requires uniform cell sizes");
    }
    const float cellSize = cellSize_x;
    const float support_factor = kernel->support();

    const int stride_x = gridnum_y * gridnum_z * num_fields;
    const int stride_y = gridnum_z * num_fields;
    const int stride_z = num_fields;
    const int weight_stride_x = gridnum_y * gridnum_z;
    const int weight_stride_y = gridnum_z;

    std::memset(fields,  0, sizeof(float) * gridnum_x * gridnum_y * gridnum_z * num_fields);
    std::memset(weights, 0, sizeof(float) * gridnum_x * gridnum_y * gridnum_z);

    const bool parallel = allow_openmp(use_openmp);
    const int threads = (parallel && omp_threads > 0) ? omp_threads : 0;
    SPHKernel* kernel_ptr = kernel.get();
    for_each_particle(N, parallel, threads, [&](int n) {
        const float* vecs = &hmat_eigvecs[n * 9];
        const float* vals = &hmat_eigvals[n * 3];

        float vals_gu[3] = { vals[0] / cellSize, vals[1] / cellSize, vals[2] / cellSize };
        float krs = support_factor * std::max({ vals_gu[0], vals_gu[1], vals_gu[2] });

        float xpos = pos[3 * n + 0] / cellSize;
        float ypos = pos[3 * n + 1] / cellSize;
        float zpos = pos[3 * n + 2] / cellSize;

        int i = static_cast<int>(xpos);
        int j = static_cast<int>(ypos);
        int k = static_cast<int>(zpos);

        int num_left   = i - static_cast<int>(xpos - krs);
        int num_right  = static_cast<int>(xpos + krs + 0.5f) - i;
        int num_bottom = j - static_cast<int>(ypos - krs);
        int num_top    = static_cast<int>(ypos + krs + 0.5f) - j;
        int num_front  = k - static_cast<int>(zpos - krs);
        int num_back   = static_cast<int>(zpos + krs + 0.5f) - k;
        const float* particle = quantities + n * num_fields;

        for (int a = i - num_left; a <= i + num_right; ++a) {
            for (int b = j - num_bottom; b <= j + num_top; ++b) {
                for (int c = k - num_front; c <= k + num_back; ++c) {
                    float fraction = compute_fraction_anisotropic_3d_cpp(
                        integration_method, vecs, vals_gu,
                        xpos, ypos, zpos, a, b, c, gridnum_x, gridnum_y, gridnum_z, periodic, kernel_ptr
                    );

                    if (fraction == 0.0f) continue;

                    int an = a;
                    int bn = b;
                    int cn = c;
                    if (periodic_x) {
                        an = apply_pbc(an, gridnum_x);
                    } else if (an < 0 || an >= gridnum_x) {
                        continue;
                    }

                    if (periodic_y) {
                        bn = apply_pbc(bn, gridnum_y);
                    } else if (bn < 0 || bn >= gridnum_y) {
                        continue;
                    }

                    if (periodic_z) {
                        cn = apply_pbc(cn, gridnum_z);
                    } else if (cn < 0 || cn >= gridnum_z) {
                        continue;
                    }

                    int base_idx = an * stride_x + bn * stride_y + cn * stride_z;
                    int weight_idx = an * weight_stride_x + bn * weight_stride_y + cn;
                    accumulate_fields(fields, base_idx, particle, num_fields, fraction, parallel);
                    accumulate_weight(weights, weight_idx, fraction, parallel);
                }
            }
        }
    });
}
