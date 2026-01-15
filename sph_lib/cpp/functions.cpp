#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>
#include <array>
#include <string>
#include <stdexcept>
#include "kernels.h"
#include "functions.h"

// =============================================================================
// Utilities
// =============================================================================

inline int apply_pbc(int idx, int gridnum) {
    int r = idx % gridnum;
    return (r < 0) ? r + gridnum : r;
}

// =============================================================================
// pure C++ deposition functions
// =============================================================================

void ngp_2d_cpp(
    const float* pos,            // (N, 2)
    const float* quantities,     // (N, num_fields)
    int N,
    int num_fields,
    float extent_min,
    float extent_max,
    int gridnum,
    float* fields,               // (gridnum, gridnum, num_fields)
    float* weights               // (gridnum, gridnum)
) {
    const float boxsize = extent_max - extent_min;
    const float inv_dx = static_cast<float>(gridnum) / boxsize;

    // strides for C-contiguous layout (x, y, f)
    const int field_stride_x = gridnum * num_fields;
    const int field_stride_y = num_fields;

    // zero output arrays
    std::memset(fields,  0, sizeof(float) * gridnum * gridnum * num_fields);
    std::memset(weights, 0, sizeof(float) * gridnum * gridnum);

    for (int n = 0; n < N; ++n) {
        // map positions to grid indices, mimicking Python astype(int)
        int ix = static_cast<int>((pos[2*n + 0] - extent_min) * inv_dx);
        int iy = static_cast<int>((pos[2*n + 1] - extent_min) * inv_dx);

        // skip particles outside the extent
        if (ix < 0 || ix >= gridnum || iy < 0 || iy >= gridnum) continue;

        const int base_idx   = ix * field_stride_x + iy * field_stride_y;
        const int weight_idx = ix * gridnum + iy;

        for (int f = 0; f < num_fields; ++f) {
            fields[base_idx + f] += quantities[n * num_fields + f];
        }

        weights[weight_idx] += 1.0f; // matches Python increment
    }
}


void ngp_3d_cpp(
    const float* pos,            // (N, 3)
    const float* quantities,     // (N, num_fields)
    int N,
    int num_fields,
    float extent_min,
    float extent_max,
    int gridnum,
    float* fields,               // (gridnum, gridnum, gridnum, num_fields)
    float* weights               // (gridnum, gridnum, gridnum)
) {
    const float boxsize = extent_max - extent_min;
    const float inv_dx = static_cast<float>(gridnum) / boxsize;

    // strides for C-contiguous layout (x, y, z, f)
    const int field_stride_x = gridnum * gridnum * num_fields;
    const int field_stride_y = gridnum * num_fields;
    const int field_stride_z = num_fields;

    // zero output arrays
    std::memset(fields,  0, sizeof(float) * gridnum * gridnum * gridnum * num_fields);
    std::memset(weights, 0, sizeof(float) * gridnum * gridnum * gridnum);

    for (int n = 0; n < N; ++n) {
        // map positions to grid indices, mimicking Python astype(int)
        int ix = static_cast<int>((pos[3*n + 0] - extent_min) * inv_dx);
        int iy = static_cast<int>((pos[3*n + 1] - extent_min) * inv_dx);
        int iz = static_cast<int>((pos[3*n + 2] - extent_min) * inv_dx);

        // skip particles outside the extent
        if (ix < 0 || ix >= gridnum || iy < 0 || iy >= gridnum || iz < 0 || iz >= gridnum) continue;

        const int base_idx   = ix * field_stride_x + iy * field_stride_y + iz * field_stride_z;
        const int weight_idx = ix * gridnum * gridnum + iy * gridnum + iz;

        for (int f = 0; f < num_fields; ++f) {
            fields[base_idx + f] += quantities[n * num_fields + f];
        }
        weights[weight_idx] += 1.0f; // matches Python increment
    }
}


void cic_2d_cpp(
    const float* pos,            // (N, 2)
    const float* quantities,     // (N, num_fields)
    int N,
    int num_fields,
    float extent_min,
    float extent_max,
    int gridnum,
    bool periodic,
    float* fields,               // (gridnum, gridnum, num_fields)
    float* weights               // (gridnum, gridnum)
) {
    const float cellSize = (extent_max - extent_min) / static_cast<float>(gridnum);

    // strides for C-contiguous layout (x, y, f)
    const int field_stride_x = gridnum * num_fields;
    const int field_stride_y = num_fields;

    // zero output arrays
    std::memset(fields,  0, sizeof(float) * gridnum * gridnum * num_fields);
    std::memset(weights, 0, sizeof(float) * gridnum * gridnum);

    for (int n = 0; n < N; ++n) {
        float xpos = (pos[2*n + 0] - extent_min) / cellSize;
        float ypos = (pos[2*n + 1] - extent_min) / cellSize;

        // Skip particles completely outside the domain
        if (!periodic && 
            (xpos < 0.0f || xpos >= gridnum || 
             ypos < 0.0f || ypos >= gridnum))
            continue;

        int i0 = static_cast<int>(std::floor(xpos));
        int j0 = static_cast<int>(std::floor(ypos));
        int i1 = i0 + 1;
        int j1 = j0 + 1;

        float dx = xpos - i0;
        float dy = ypos - j0;
        float dx_ = 1.0f - dx;
        float dy_ = 1.0f - dy;

        if (!periodic) {
            if (xpos < 0.5f || xpos > gridnum - 0.5f ||
                ypos < 0.5f || ypos > gridnum - 0.5f)
                continue;
        }

        // Wrap indices only if periodic
        if (periodic) {
            i0 = apply_pbc(i0, gridnum);
            i1 = apply_pbc(i1, gridnum);
            j0 = apply_pbc(j0, gridnum);
            j1 = apply_pbc(j1, gridnum);
        }

        // weights for bilinear stencil
        float w00 = dx_ * dy_;
        float w10 = dx  * dy_;
        float w01 = dx_ * dy;
        float w11 = dx  * dy;

        auto deposit = [&](int ix, int jy, float w) {
            if (!periodic && (ix < 0 || ix >= gridnum || jy < 0 || jy >= gridnum)) return;

            int base_idx   = ix * field_stride_x + jy * field_stride_y;
            int weight_idx = ix * gridnum + jy;

            for (int f = 0; f < num_fields; ++f) {
                fields[base_idx + f] += w * quantities[n * num_fields + f];
            }
            weights[weight_idx] += w;
        };

        deposit(i0, j0, w00);
        deposit(i1, j0, w10);
        deposit(i0, j1, w01);
        deposit(i1, j1, w11);
    }
}


void cic_3d_cpp(
    const float* pos,        // (N, 3)
    const float* quantities, // (N, num_fields)
    int N,
    int num_fields,
    float extent_min,
    float extent_max,
    int gridnum,
    bool periodic,
    float* fields,           // (gridnum, gridnum, gridnum, num_fields)
    float* weights           // (gridnum, gridnum, gridnum)
) {
    const float cellSize = (extent_max - extent_min) / static_cast<float>(gridnum);

    // Strides for C-contiguous layout (x, y, z, f)
    const int field_stride_x = gridnum * gridnum * num_fields;
    const int field_stride_y = gridnum * num_fields;
    const int field_stride_z = num_fields;

    // Zero output arrays
    std::memset(fields,  0, sizeof(float) * gridnum * gridnum * gridnum * num_fields);
    std::memset(weights, 0, sizeof(float) * gridnum * gridnum * gridnum);

    for (int n = 0; n < N; ++n) {
        float xpos = (pos[3*n + 0] - extent_min) / cellSize;
        float ypos = (pos[3*n + 1] - extent_min) / cellSize;
        float zpos = (pos[3*n + 2] - extent_min) / cellSize;

        // Skip particles completely outside the domain (non-periodic)
        if (!periodic && (xpos < 0.0f || xpos >= gridnum ||
                          ypos < 0.0f || ypos >= gridnum ||
                          zpos < 0.0f || zpos >= gridnum))
            continue;

        // Base indices and fractional distances
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

        // Wrap indices only if periodic
        if (periodic) {
            i0 = apply_pbc(i0, gridnum);
            i1 = apply_pbc(i1, gridnum);
            j0 = apply_pbc(j0, gridnum);
            j1 = apply_pbc(j1, gridnum);
            k0 = apply_pbc(k0, gridnum);
            k1 = apply_pbc(k1, gridnum);
        }

        // Trilinear weights
        float w000 = dx_ * dy_ * dz_;
        float w100 = dx  * dy_ * dz_;
        float w010 = dx_ * dy  * dz_;
        float w110 = dx  * dy  * dz_;
        float w001 = dx_ * dy_ * dz;
        float w101 = dx  * dy_ * dz;
        float w011 = dx_ * dy  * dz;
        float w111 = dx  * dy  * dz;

        // Deposit lambda
        auto deposit = [&](int ix, int jy, int kz, float w) {
            if (!periodic && (ix < 0 || ix >= gridnum ||
                              jy < 0 || jy >= gridnum ||
                              kz < 0 || kz >= gridnum))
                return;

            int base_idx   = ix * field_stride_x + jy * field_stride_y + kz * field_stride_z;
            int weight_idx = ix * gridnum * gridnum + jy * gridnum + kz;

            for (int f = 0; f < num_fields; ++f) {
                fields[base_idx + f] += w * quantities[n * num_fields + f];
            }
            weights[weight_idx] += w;
        };

        // Deposit to 8 surrounding grid points
        deposit(i0, j0, k0, w000);
        deposit(i1, j0, k0, w100);
        deposit(i0, j1, k0, w010);
        deposit(i1, j1, k0, w110);
        deposit(i0, j0, k1, w001);
        deposit(i1, j0, k1, w101);
        deposit(i0, j1, k1, w011);
        deposit(i1, j1, k1, w111);
    }
}

void cic_2d_adaptive_cpp(
    const float* pos,         // (N,2)
    const float* quantities,  // (N,num_fields)
    int N,
    int num_fields,
    float extent_min,
    float extent_max,
    int gridnum,
    bool periodic,
    const float* pcellsizesHalf, // (N)
    float* fields,            // (gridnum, gridnum, num_fields)
    float* weights            // (gridnum, gridnum)
) {
    const float cellSize = (extent_max - extent_min) / static_cast<float>(gridnum);
    const int stride_x = gridnum * num_fields;
    const int stride_y = num_fields;
    const int weight_stride_x = gridnum;

    std::memset(fields,  0, sizeof(float) * gridnum * gridnum * num_fields);
    std::memset(weights, 0, sizeof(float) * gridnum * gridnum);

    for (int n = 0; n < N; ++n) {
        float pcs = pcellsizesHalf[n] / cellSize;
        float V = std::pow(2.0f * pcs, 2.0f);

        float xpos = (pos[2*n + 0] - extent_min) / cellSize;
        float ypos = (pos[2*n + 1] - extent_min) / cellSize;

        int i = static_cast<int>(xpos);
        int j = static_cast<int>(ypos);

        int num_left   = i - static_cast<int>(std::round(xpos - pcs - 0.5f));
        int num_right  = static_cast<int>(xpos + pcs) - i;
        int num_bottom = j - static_cast<int>(std::round(ypos - pcs - 0.5f));
        int num_top    = static_cast<int>(ypos + pcs) - j;

        float c1 = xpos - pcs, c2 = xpos + pcs;
        float c3 = ypos - pcs, c4 = ypos + pcs;

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
                if (periodic) {
                    an = apply_pbc(a, gridnum);
                    bn = apply_pbc(b, gridnum);
                } else {
                    if (a < 0 || a >= gridnum || b < 0 || b >= gridnum) continue;
                }

                int base_idx = an * stride_x + bn * stride_y;
                int weight_idx = an * weight_stride_x + bn;

                for (int f = 0; f < num_fields; ++f) {
                    fields[base_idx + f] += fraction * quantities[n*num_fields + f];
                }
                weights[weight_idx] += fraction;
            }
        }
    }
}

void cic_3d_adaptive_cpp(
    const float* pos,         // (N,3)
    const float* quantities,  // (N,num_fields)
    int N,
    int num_fields,
    float extent_min,
    float extent_max,
    int gridnum,
    bool periodic,
    const float* pcellsizesHalf, // (N)
    float* fields,            // (gridnum, gridnum, gridnum, num_fields)
    float* weights            // (gridnum, gridnum, gridnum)
) {
    const float cellSize = (extent_max - extent_min) / static_cast<float>(gridnum);
    const int stride_x = gridnum * gridnum * num_fields;
    const int stride_y = gridnum * num_fields;
    const int stride_z = num_fields;
    const int weight_stride_x = gridnum * gridnum;
    const int weight_stride_y = gridnum;
    const int weight_stride_z = 1;

    std::memset(fields,  0, sizeof(float) * gridnum * gridnum * gridnum * num_fields);
    std::memset(weights, 0, sizeof(float) * gridnum * gridnum * gridnum);

    for (int n = 0; n < N; ++n) {
        float pcs = pcellsizesHalf[n] / cellSize;
        float V = std::pow(2.0f * pcs, 3.0f);

        float xpos = (pos[3*n + 0] - extent_min) / cellSize;
        float ypos = (pos[3*n + 1] - extent_min) / cellSize;
        float zpos = (pos[3*n + 2] - extent_min) / cellSize;

        int i = static_cast<int>(xpos);
        int j = static_cast<int>(ypos);
        int k = static_cast<int>(zpos);

        int num_left   = i - static_cast<int>(std::round(xpos - pcs - 0.5f));
        int num_right  = static_cast<int>(xpos + pcs) - i;
        int num_bottom = j - static_cast<int>(std::round(ypos - pcs - 0.5f));
        int num_top    = static_cast<int>(ypos + pcs) - j;
        int num_back   = k - static_cast<int>(std::round(zpos - pcs - 0.5f));
        int num_fwd    = static_cast<int>(zpos + pcs) - k;

        float c1 = xpos - pcs, c2 = xpos + pcs;
        float c3 = ypos - pcs, c4 = ypos + pcs;
        float c5 = zpos - pcs, c6 = zpos + pcs;

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
                    if (periodic) {
                        an = apply_pbc(a, gridnum);
                        bn = apply_pbc(b, gridnum);
                        cn = apply_pbc(c, gridnum);
                    } else {
                        if (a < 0 || a >= gridnum || b < 0 || b >= gridnum || c < 0 || c >= gridnum)
                            continue;
                    }

                    int base_idx = an * stride_x + bn * stride_y + cn * stride_z;
                    int weight_idx = an * weight_stride_x + bn * weight_stride_y + cn * weight_stride_z;

                    for (int f = 0; f < num_fields; ++f) {
                        fields[base_idx + f] += fraction * quantities[n*num_fields + f];
                    }
                    weights[weight_idx] += fraction;
                }
            }
        }
    }
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
    float extent_min,
    float extent_max,
    int gridnum,
    bool periodic,
    float* fields,           // (gridnum, gridnum, num_fields)
    float* weights           // (gridnum, gridnum)
) {
    const float boxsize = extent_max - extent_min;
    const float inv_dx = static_cast<float>(gridnum) / boxsize;

    // Strides for C-contiguous layout (x, y, f)
    const int stride_x = gridnum * num_fields;
    const int stride_y = num_fields;

    // Zero output arrays
    std::memset(fields,  0, sizeof(float) * gridnum * gridnum * num_fields);
    std::memset(weights, 0, sizeof(float) * gridnum * gridnum);

    // Neighbor offsets
    const int offsets[3] = {-1, 0, 1};

    for (int n = 0; n < N; ++n) {
        // Convert particle position to grid coordinates
        float xpos = (pos[2*n + 0] - extent_min) * inv_dx;
        float ypos = (pos[2*n + 1] - extent_min) * inv_dx;

        int i_base = static_cast<int>(std::floor(xpos));
        int j_base = static_cast<int>(std::floor(ypos));

        float dx = xpos - i_base;
        float dy = ypos - j_base;

        // Compute TSC weights for each axis
        float wx[3], wy[3];
        tsc_weights(dx, wx);
        tsc_weights(dy, wy);

        // Loop over neighbor offsets
        for (int dx_i = 0; dx_i < 3; ++dx_i) {
            for (int dy_i = 0; dy_i < 3; ++dy_i) {
                int ix = i_base + offsets[dx_i];
                int iy = j_base + offsets[dy_i];

                // Apply periodic boundaries if needed
                if (periodic) {
                    ix = apply_pbc(ix, gridnum);
                    iy = apply_pbc(iy, gridnum);
                } else {
                    // skip if out-of-bounds
                    if (ix < 0 || ix >= gridnum || iy < 0 || iy >= gridnum) continue;
                }

                float w = wx[dx_i] * wy[dy_i];
                if (w == 0.0f) continue;

                int base_idx   = ix * stride_x + iy * stride_y;
                int weight_idx = ix * gridnum + iy;

                for (int f = 0; f < num_fields; ++f) {
                    fields[base_idx + f] += w * quantities[n * num_fields + f];
                }
                weights[weight_idx] += w;
            }
        }
    }
}

void tsc_3d_cpp(
    const float* pos,        // (N,3)
    const float* quantities, // (N,num_fields)
    int N,
    int num_fields,
    float extent_min,
    float extent_max,
    int gridnum,
    bool periodic,
    float* fields,           // (gridnum, gridnum, gridnum, num_fields)
    float* weights           // (gridnum, gridnum, gridnum)
) {
    const float boxsize = extent_max - extent_min;
    const float inv_dx = static_cast<float>(gridnum) / boxsize;

    // Strides for C-contiguous layout (x, y, z, f)
    const int stride_x = gridnum *   gridnum * num_fields;
    const int stride_y = gridnum * num_fields;
    const int stride_z = num_fields;

    // Zero output arrays
    std::memset(fields,  0, sizeof(float) * gridnum * gridnum * gridnum * num_fields);
    std::memset(weights, 0, sizeof(float) * gridnum * gridnum * gridnum);

    // Neighbor offsets
    const int offsets[3] = {-1, 0, 1};

    for (int n = 0; n < N; ++n) {
        // Convert particle position to grid coordinates
        float xpos = (pos[3*n + 0] - extent_min) * inv_dx;
        float ypos = (pos[3*n + 1] - extent_min) * inv_dx;
        float zpos = (pos[3*n + 2] - extent_min) * inv_dx;

        int i_base = static_cast<int>(std::floor(xpos));
        int j_base = static_cast<int>(std::floor(ypos));
        int k_base = static_cast<int>(std::floor(zpos));

        float dx = xpos - i_base;
        float dy = ypos - j_base;
        float dz = zpos - k_base;

        // Compute TSC weights along each axis
        float wx[3], wy[3], wz[3];
        tsc_weights(dx, wx);
        tsc_weights(dy, wy);
        tsc_weights(dz, wz);

        // Loop over 3×3×3 neighbors
        for (int dx_i = 0; dx_i < 3; ++dx_i) {
            for (int dy_i = 0; dy_i < 3; ++dy_i) {
                for (int dz_i = 0; dz_i < 3; ++dz_i) {
                    int ix = i_base + offsets[dx_i];
                    int iy = j_base + offsets[dy_i];
                    int iz = k_base + offsets[dz_i];

                    // Apply periodic boundaries if needed
                    if (periodic) {
                        ix = apply_pbc(ix, gridnum);
                        iy = apply_pbc(iy, gridnum);
                        iz = apply_pbc(iz, gridnum);
                    } else {
                        // Skip out-of-bounds neighbors
                        if (ix < 0 || ix >= gridnum ||
                            iy < 0 || iy >= gridnum ||
                            iz < 0 || iz >= gridnum) continue;
                    }

                    float w = wx[dx_i] * wy[dy_i] * wz[dz_i];
                    if (w == 0.0f) continue;

                    int base_idx   = ix * stride_x + iy * stride_y + iz * stride_z;
                    int weight_idx = ix * gridnum * gridnum + iy * gridnum + iz;

                    for (int f = 0; f < num_fields; ++f) {
                        fields[base_idx + f] += w * quantities[n * num_fields + f];
                    }
                    weights[weight_idx] += w;
                }
            }
        }
    }
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
    float extent_min,
    float extent_max,
    int gridnum,
    bool periodic,
    const float* pcellsizesHalf, // (N)
    float* fields,           // (gridnum, gridnum, num_fields)
    float* weights           // (gridnum, gridnum)
) {
    const float cellSize = (extent_max - extent_min) / static_cast<float>(gridnum);

    const int stride_x = gridnum * num_fields;
    const int stride_y = num_fields;
    const int weight_stride_x = gridnum;
    const int weight_stride_y = 1;

    std::memset(fields,  0, sizeof(float) * gridnum * gridnum * num_fields);
    std::memset(weights, 0, sizeof(float) * gridnum * gridnum);

    for (int n = 0; n < N; ++n) {
        float x = (pos[2*n + 0] - extent_min) / cellSize;
        float y = (pos[2*n + 1] - extent_min) / cellSize;
        float h = pcellsizesHalf[n] / cellSize;

        float support = 1.5f * h;
        int i_min = static_cast<int>(std::floor(x - support));
        int i_max = static_cast<int>(std::ceil(x + support));
        int j_min = static_cast<int>(std::floor(y - support));
        int j_max = static_cast<int>(std::ceil(y + support));

        for (int i = i_min; i <= i_max; ++i) {
            int ii = i;
            if (periodic) ii = apply_pbc(i, gridnum);
            else if (ii < 0 || ii >= gridnum) continue;

            float wx = tsc_integrated_weight_1d(x, float(i), float(i+1), h);
            if (wx == 0.0f) continue;

            for (int j = j_min; j <= j_max; ++j) {
                int jj = j;
                if (periodic) jj = apply_pbc(j, gridnum);
                else if (jj < 0 || jj >= gridnum) continue;

                float wy = tsc_integrated_weight_1d(y, float(j), float(j+1), h);
                if (wy == 0.0f) continue;

                float w = wx * wy;
                int base_idx = ii * stride_x + jj * stride_y;
                int weight_idx = ii * weight_stride_x + jj * weight_stride_y;

                for (int f = 0; f < num_fields; ++f) {
                    fields[base_idx + f] += w * quantities[n*num_fields + f];
                }
                weights[weight_idx] += w;
            }
        }
    }
}

void tsc_3d_adaptive_cpp(
    const float* pos,        // (N,3)
    const float* quantities, // (N,num_fields)
    int N,
    int num_fields,
    float extent_min,
    float extent_max,
    int gridnum,
    bool periodic,
    const float* pcellsizesHalf, // (N)
    float* fields,           // (gridnum, gridnum, gridnum, num_fields)
    float* weights           // (gridnum, gridnum, gridnum)
) {
    const float cellSize = (extent_max - extent_min) / static_cast<float>(gridnum);

    const int stride_x = gridnum * gridnum * num_fields;
    const int stride_y = gridnum * num_fields;
    const int stride_z = num_fields;

    const int weight_stride_x = gridnum * gridnum;
    const int weight_stride_y = gridnum;
    const int weight_stride_z = 1;

    std::memset(fields,  0, sizeof(float) * gridnum * gridnum * gridnum * num_fields);
    std::memset(weights, 0, sizeof(float) * gridnum * gridnum * gridnum);

    for (int n = 0; n < N; ++n) {
        float x = (pos[3*n + 0] - extent_min) / cellSize;
        float y = (pos[3*n + 1] - extent_min) / cellSize;
        float z = (pos[3*n + 2] - extent_min) / cellSize;
        float h = pcellsizesHalf[n] / cellSize;

        float support = 1.5f * h;
        int i_min = static_cast<int>(std::floor(x - support));
        int i_max = static_cast<int>(std::ceil(x + support));
        int j_min = static_cast<int>(std::floor(y - support));
        int j_max = static_cast<int>(std::ceil(y + support));
        int k_min = static_cast<int>(std::floor(z - support));
        int k_max = static_cast<int>(std::ceil(z + support));

        for (int i = i_min; i <= i_max; ++i) {
            int ii = i;
            if (periodic) ii = apply_pbc(i, gridnum);
            else if (ii < 0 || ii >= gridnum) continue;
            float wx = tsc_integrated_weight_1d(x, float(i), float(i+1), h);
            if (wx == 0.0f) continue;

            for (int j = j_min; j <= j_max; ++j) {
                int jj = j;
                if (periodic) jj = apply_pbc(j, gridnum);
                else if (jj < 0 || jj >= gridnum) continue;
                float wy = tsc_integrated_weight_1d(y, float(j), float(j+1), h);
                if (wy == 0.0f) continue;

                for (int k = k_min; k <= k_max; ++k) {
                    int kk = k;
                    if (periodic) kk = apply_pbc(k, gridnum);
                    else if (kk < 0 || kk >= gridnum) continue;
                    float wz = tsc_integrated_weight_1d(z, float(k), float(k+1), h);
                    if (wz == 0.0f) continue;

                    float w = wx * wy * wz;

                    int base_idx = ii * stride_x + jj * stride_y + kk * stride_z;
                    int weight_idx = ii * weight_stride_x + jj * weight_stride_y + kk * weight_stride_z;

                    for (int f = 0; f < num_fields; ++f) {
                        fields[base_idx + f] += w * quantities[n*num_fields + f];
                    }
                    weights[weight_idx] += w;
                }
            }
        }
    }
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
    int gridnum,
    bool periodic,
    float h,
    SPHKernel* kernel
) {
    float sigma = kernel->normalization(h);

    if (method == "midpoint") {
        float dx = xpos - (a + 0.5f);
        float dy = ypos - (b + 0.5f);

        if (periodic) {
            if (dx > gridnum / 2.0f) dx -= gridnum;
            if (dy > gridnum / 2.0f) dy -= gridnum;
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

            if (periodic) {
                if (dx > gridnum / 2.0f) dx -= gridnum;
                if (dy > gridnum / 2.0f) dy -= gridnum;
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

            if (periodic) {
                if (dx > gridnum / 2.0f) dx -= gridnum;
                if (dy > gridnum / 2.0f) dy -= gridnum;
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

            if (periodic) {
                if (dx > gridnum / 2.0f) dx -= gridnum;
                if (dy > gridnum / 2.0f) dy -= gridnum;
            }

            float r = std::sqrt(dx * dx + dy * dy);
            sum += 2.0f * kernel->weight(r, h);
        }

        for (int dx_c = 0; dx_c <= 1; ++dx_c) {
            for (int dy_c = 0; dy_c <= 1; ++dy_c) {
                float dx = xpos - (a + dx_c);
                float dy = ypos - (b + dy_c);

                if (periodic) {
                    if (dx > gridnum / 2.0f) dx -= gridnum;
                    if (dy > gridnum / 2.0f) dy -= gridnum;
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
    float extent_min,
    float extent_max,
    int gridnum,
    bool periodic,
    const std::string& kernel_name,
    const std::string& integration_method,
    float* fields,             // (gridnum, gridnum, num_fields)
    float* weights             // (gridnum, gridnum)
) {
    auto kernel = create_kernel(kernel_name, 2, false);

    const float cellSize = (extent_max - extent_min) / static_cast<float>(gridnum);
    const float support_factor = kernel->support();

    const int stride_x = gridnum * num_fields;
    const int stride_y = num_fields;
    const int weight_stride_x = gridnum;

    std::memset(fields,  0, sizeof(float) * gridnum * gridnum * num_fields);
    std::memset(weights, 0, sizeof(float) * gridnum * gridnum);

    for (int n = 0; n < N; ++n) {
        float hsn = hsm[n] / cellSize;
        float support = support_factor * hsn;

        float xpos = (pos[2*n + 0] - extent_min) / cellSize;
        float ypos = (pos[2*n + 1] - extent_min) / cellSize;

        int i = static_cast<int>(xpos);
        int j = static_cast<int>(ypos);

        int num_left   = i - static_cast<int>(xpos - support);
        int num_right  = static_cast<int>(xpos + support + 0.5f) - i;
        int num_bottom = j - static_cast<int>(ypos - support);
        int num_top    = static_cast<int>(ypos + support + 0.5f) - j;

        for (int a = i - num_left; a <= i + num_right; ++a) {
            for (int b = j - num_bottom; b <= j + num_top; ++b) {
                float w = compute_fraction_isotropic_2d_cpp(
                    integration_method, xpos, ypos,
                    a, b, gridnum, periodic, hsn, kernel.get()
                );

                if (w == 0.0f) continue;

                int an = a;
                int bn = b;
                if (periodic) {
                    an = apply_pbc(an, gridnum);
                    bn = apply_pbc(bn, gridnum);
                } else if (an < 0 || an >= gridnum || bn < 0 || bn >= gridnum) {
                    continue;
                }

                int base_idx = an * stride_x + bn * stride_y;
                int weight_idx = an * weight_stride_x + bn;

                for (int f = 0; f < num_fields; ++f) {
                    fields[base_idx + f] += quantities[n * num_fields + f] * w;
                }
                weights[weight_idx] += w;
            }
        }
    }
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
    int gridnum,
    bool periodic,
    float h,
    SPHKernel* kernel
) {
    float sigma = kernel->normalization(h);

    if (method == "midpoint") {
        float dx = xpos - (a + 0.5f);
        float dy = ypos - (b + 0.5f);
        float dz = zpos - (c + 0.5f);

        if (periodic) {
            if (dx > gridnum / 2.0f) dx -= gridnum;
            if (dy > gridnum / 2.0f) dy -= gridnum;
            if (dz > gridnum / 2.0f) dz -= gridnum;
        }

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

            if (periodic) {
                if (dx > gridnum / 2.0f) dx -= gridnum;
                if (dy > gridnum / 2.0f) dy -= gridnum;
                if (dz > gridnum / 2.0f) dz -= gridnum;
            }

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
            if (periodic) {
                if (dx > gridnum / 2.0f) dx -= gridnum;
                if (dy > gridnum / 2.0f) dy -= gridnum;
                if (dz > gridnum / 2.0f) dz -= gridnum;
            }
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
            if (periodic) {
                if (dx > gridnum / 2.0f) dx -= gridnum;
                if (dy > gridnum / 2.0f) dy -= gridnum;
                if (dz > gridnum / 2.0f) dz -= gridnum;
            }
            float r = std::sqrt(dx * dx + dy * dy + dz * dz);
            sum += 4.0f * kernel->weight(r, h);
        }

        for (int dx_c = 0; dx_c <= 1; ++dx_c) {
            for (int dy_c = 0; dy_c <= 1; ++dy_c) {
                for (int dz_c = 0; dz_c <= 1; ++dz_c) {
                    float dx = xpos - (a + dx_c);
                    float dy = ypos - (b + dy_c);
                    float dz = zpos - (c + dz_c);
                    if (periodic) {
                        if (dx > gridnum / 2.0f) dx -= gridnum;
                        if (dy > gridnum / 2.0f) dy -= gridnum;
                        if (dz > gridnum / 2.0f) dz -= gridnum;
                    }
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
    float extent_min,
    float extent_max,
    int gridnum,
    bool periodic,
    const std::string& kernel_name,
    const std::string& integration_method,
    float* fields,             // (gridnum, gridnum, gridnum, num_fields)
    float* weights             // (gridnum, gridnum, gridnum)
) {
    auto kernel = create_kernel(kernel_name, 3, false);

    const float cellSize = (extent_max - extent_min) / static_cast<float>(gridnum);
    const float support_factor = kernel->support();

    const int stride_x = gridnum * gridnum * num_fields;
    const int stride_y = gridnum * num_fields;
    const int stride_z = num_fields;
    const int weight_stride_x = gridnum * gridnum;
    const int weight_stride_y = gridnum;

    std::memset(fields,  0, sizeof(float) * gridnum * gridnum * gridnum * num_fields);
    std::memset(weights, 0, sizeof(float) * gridnum * gridnum * gridnum);

    for (int n = 0; n < N; ++n) {
        float hsn = hsm[n] / cellSize;
        float support = support_factor * hsn;

        float xpos = (pos[3*n + 0] - extent_min) / cellSize;
        float ypos = (pos[3*n + 1] - extent_min) / cellSize;
        float zpos = (pos[3*n + 2] - extent_min) / cellSize;

        int i = static_cast<int>(xpos);
        int j = static_cast<int>(ypos);
        int k = static_cast<int>(zpos);

        int num_left   = i - static_cast<int>(xpos - support);
        int num_right  = static_cast<int>(xpos + support + 0.5f) - i;
        int num_bottom = j - static_cast<int>(ypos - support);
        int num_top    = static_cast<int>(ypos + support + 0.5f) - j;
        int num_front  = k - static_cast<int>(zpos - support);
        int num_back   = static_cast<int>(zpos + support + 0.5f) - k;

        for (int a = i - num_left; a <= i + num_right; ++a) {
            for (int b = j - num_bottom; b <= j + num_top; ++b) {
                for (int c = k - num_front; c <= k + num_back; ++c) {
                    float w = compute_fraction_isotropic_3d_cpp(
                        integration_method, xpos, ypos, zpos,
                        a, b, c, gridnum, periodic, hsn, kernel.get()
                    );

                    if (w == 0.0f) continue;

                    int an = a;
                    int bn = b;
                    int cn = c;
                    if (periodic) {
                        an = apply_pbc(an, gridnum);
                        bn = apply_pbc(bn, gridnum);
                        cn = apply_pbc(cn, gridnum);
                    } else if (an < 0 || an >= gridnum || bn < 0 || bn >= gridnum || cn < 0 || cn >= gridnum) {
                        continue;
                    }

                    int base_idx = an * stride_x + bn * stride_y + cn * stride_z;
                    int weight_idx = an * weight_stride_x + bn * weight_stride_y + cn;

                    for (int f = 0; f < num_fields; ++f) {
                        fields[base_idx + f] += quantities[n * num_fields + f] * w;
                    }
                    weights[weight_idx] += w;
                }
            }
        }
    }
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
    int gridnum,
    bool periodic,
    SPHKernel* kernel
) {
    float detH = vals_gu[0] * vals_gu[1];
    float sigma = kernel->normalization(detH);

    auto wrap = [&](float& d) {
        if (d >  gridnum * 0.5f) d -= gridnum;
        if (d < -gridnum * 0.5f) d += gridnum;
    };

    auto eval = [&](float ox, float oy) {
        float dx = xpos - (a + ox);
        float dy = ypos - (b + oy);
        if (periodic) { wrap(dx); wrap(dy); }

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
    float extent_min,
    float extent_max,
    int gridnum,
    bool periodic,
    const std::string& kernel_name,
    const std::string& integration_method,
    float* fields,
    float* weights
) {
    auto kernel = create_kernel(kernel_name, 2, true);

    const float cellSize = (extent_max - extent_min) / static_cast<float>(gridnum);
    const float support_factor = kernel->support();

    const int stride_x = gridnum * num_fields;
    const int stride_y = num_fields;
    const int weight_stride_x = gridnum;

    std::memset(fields,  0, sizeof(float) * gridnum * gridnum * num_fields);
    std::memset(weights, 0, sizeof(float) * gridnum * gridnum);

    for (int n = 0; n < N; ++n) {
        const float* vecs = &hmat_eigvecs[n * 4];
        const float* vals = &hmat_eigvals[n * 2];

        float vals_gu[2] = { vals[0] / cellSize, vals[1] / cellSize };
        float krs = support_factor * std::max({ vals_gu[0], vals_gu[1] });

        float xpos = (pos[2 * n + 0] - extent_min) / cellSize;
        float ypos = (pos[2 * n + 1] - extent_min) / cellSize;

        int i = static_cast<int>(xpos);
        int j = static_cast<int>(ypos);

        int num_left   = i - static_cast<int>(xpos - krs);
        int num_right  = static_cast<int>(xpos + krs + 0.5f) - i;
        int num_bottom = j - static_cast<int>(ypos - krs);
        int num_top    = static_cast<int>(ypos + krs + 0.5f) - j;

        for (int a = i - num_left; a <= i + num_right; ++a) {
            for (int b = j - num_bottom; b <= j + num_top; ++b) {
                float fraction = compute_fraction_anisotropic_2d_cpp(
                    integration_method, vecs, vals_gu,
                    xpos, ypos, a, b, gridnum, periodic, kernel.get()
                );

                if (fraction == 0.0f) continue;

                int an = a;
                int bn = b;
                if (periodic) {
                    an = apply_pbc(an, gridnum);
                    bn = apply_pbc(bn, gridnum);
                } else if (an < 0 || an >= gridnum || bn < 0 || bn >= gridnum) {
                    continue;
                }

                int base_idx = an * stride_x + bn * stride_y;
                int weight_idx = an * weight_stride_x + bn;

                for (int f = 0; f < num_fields; ++f) {
                    fields[base_idx + f] += quantities[n * num_fields + f] * fraction;
                }
                weights[weight_idx] += fraction;
            }
        }
    }
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
    int gridnum,
    bool periodic,
    SPHKernel* kernel
) {
    float detH = vals_gu[0] * vals_gu[1] * vals_gu[2];
    float sigma = kernel->normalization(detH);

    auto wrap = [&](float& d) {
        if (d >  gridnum * 0.5f) d -= gridnum;
        if (d < -gridnum * 0.5f) d += gridnum;
    };

    auto eval = [&](float ox, float oy, float oz) {
        float dx = xpos - (a + ox);
        float dy = ypos - (b + oy);
        float dz = zpos - (c + oz);
        if (periodic) { wrap(dx); wrap(dy); wrap(dz); }

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
        const int e[12][3] = {
            {0,0,1},{0,1,0},{1,0,0},{1,1,0},
            {1,0,1},{0,1,1},{0,0,1},{1,1,1},
            {0,1,0},{1,0,0},{0,0,0},{1,1,1}
        };

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
    float extent_min,
    float extent_max,
    int gridnum,
    bool periodic,
    const std::string& kernel_name,
    const std::string& integration_method,
    float* fields,
    float* weights
) {
    auto kernel = create_kernel(kernel_name, 3, true);

    const float cellSize = (extent_max - extent_min) / static_cast<float>(gridnum);
    const float support_factor = kernel->support();

    const int stride_x = gridnum * gridnum * num_fields;
    const int stride_y = gridnum * num_fields;
    const int stride_z = num_fields;
    const int weight_stride_x = gridnum * gridnum;
    const int weight_stride_y = gridnum;

    std::memset(fields,  0, sizeof(float) * gridnum * gridnum * gridnum * num_fields);
    std::memset(weights, 0, sizeof(float) * gridnum * gridnum * gridnum);

    for (int n = 0; n < N; ++n) {
        const float* vecs = &hmat_eigvecs[n * 9];
        const float* vals = &hmat_eigvals[n * 3];

        float vals_gu[3] = { vals[0] / cellSize, vals[1] / cellSize, vals[2] / cellSize };
        float krs = support_factor * std::max({ vals_gu[0], vals_gu[1], vals_gu[2] });

        float xpos = (pos[3 * n + 0] - extent_min) / cellSize;
        float ypos = (pos[3 * n + 1] - extent_min) / cellSize;
        float zpos = (pos[3 * n + 2] - extent_min) / cellSize;

        int i = static_cast<int>(xpos);
        int j = static_cast<int>(ypos);
        int k = static_cast<int>(zpos);

        int num_left   = i - static_cast<int>(xpos - krs);
        int num_right  = static_cast<int>(xpos + krs + 0.5f) - i;
        int num_bottom = j - static_cast<int>(ypos - krs);
        int num_top    = static_cast<int>(ypos + krs + 0.5f) - j;
        int num_front  = k - static_cast<int>(zpos - krs);
        int num_back   = static_cast<int>(zpos + krs + 0.5f) - k;

        for (int a = i - num_left; a <= i + num_right; ++a) {
            for (int b = j - num_bottom; b <= j + num_top; ++b) {
                for (int c = k - num_front; c <= k + num_back; ++c) {
                    float fraction = compute_fraction_anisotropic_3d_cpp(
                        integration_method, vecs, vals_gu,
                        xpos, ypos, zpos, a, b, c, gridnum, periodic, kernel.get()
                    );

                    if (fraction == 0.0f) continue;

                    int an = a;
                    int bn = b;
                    int cn = c;
                    if (periodic) {
                        an = apply_pbc(an, gridnum);
                        bn = apply_pbc(bn, gridnum);
                        cn = apply_pbc(cn, gridnum);
                    } else if (an < 0 || an >= gridnum || bn < 0 || bn >= gridnum || cn < 0 || cn >= gridnum) {
                        continue;
                    }

                    int base_idx = an * stride_x + bn * stride_y + cn * stride_z;
                    int weight_idx = an * weight_stride_x + bn * weight_stride_y + cn;

                    for (int f = 0; f < num_fields; ++f) {
                        fields[base_idx + f] += quantities[n * num_fields + f] * fraction;
                    }
                    weights[weight_idx] += fraction;
                }
            }
        }
    }
}
