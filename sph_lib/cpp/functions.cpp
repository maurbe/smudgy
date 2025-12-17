#include <torch/extension.h>
#include <cmath>
#include <tuple>
#include <vector>
#include <utility>
#include <iostream>
#include "kernels.h"


// Apply periodic boundary conditions
inline std::pair<int, float> account_for_pbc(int a, int gridnum, bool periodic) {
    int an = a;
    float fraction = -1.0f;
    if (a < 0) {
        an = a + gridnum;
        if (!periodic) fraction = 0.0f;
    } else if (a >= gridnum) {
        an = a - gridnum;
        if (!periodic) fraction = 0.0f;
    }
    return std::make_pair(an, fraction);
}


std::vector<at::Tensor> ngp_2d(
    at::Tensor pos,          // (N, 2)
    at::Tensor quantities,   // (N, num_fields)
    at::Tensor extent,       // [min, max]
    int gridnum,
    bool periodic
) {
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(pos.device());
    int num_fields = quantities.size(1);

    at::Tensor fields = torch::zeros({gridnum, gridnum, num_fields}, options);
    at::Tensor weights = torch::zeros({gridnum, gridnum}, options);

    float extent_min = extent[0].item<float>();
    float extent_max = extent[1].item<float>();
    float boxsize = extent_max - extent_min;
    float cellSize = boxsize / static_cast<float>(gridnum);
    float inv_cellSize = 1.0f / cellSize;

    float* __restrict__ fields_ptr = fields.data_ptr<float>();
    float* __restrict__ weights_ptr = weights.data_ptr<float>();
    float* __restrict__ pos_ptr = pos.data_ptr<float>();
    float* __restrict__ quant_ptr = quantities.data_ptr<float>();

    int field_stride_x = gridnum * num_fields;
    int field_stride_y = num_fields;

    int N = pos.size(0);

    //#pragma omp parallel for // schedule(static)
    for (int n = 0; n < N; ++n) {
        float xpos = (pos_ptr[n*2 + 0] - extent_min) * inv_cellSize;
        float ypos = (pos_ptr[n*2 + 1] - extent_min) * inv_cellSize;

        int ix = static_cast<int>(floorf(xpos));
        int iy = static_cast<int>(floorf(ypos));

        int base_idx = ix * field_stride_x + iy * field_stride_y;
        int weight_idx = ix * gridnum + iy;

        for (int f = 0; f < num_fields; ++f) {
            float val = quant_ptr[n * num_fields + f];
            
            //#pragma omp atomic
            fields_ptr[base_idx + f] += val;
        }
        //#pragma omp atomic
        weights_ptr[weight_idx] += 1.0f;
    }

    return {fields, weights};
}


std::vector<at::Tensor> ngp_3d(
    at::Tensor pos,          // (N, 3)
    at::Tensor quantities,   // (N, num_fields)
    at::Tensor extent,       // [min, max]
    int gridnum,
    bool periodic
) {
    //#ifdef _OPENMP
    //std::cout << "Using OpenMP with " << omp_get_max_threads() << " threads\n";
    //#else
    //std::cout << "Not using OpenMP\n";
    //#endif


    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(pos.device());
    int num_fields = quantities.size(1);

    at::Tensor fields = torch::zeros({gridnum, gridnum, gridnum, num_fields}, options);
    at::Tensor weights = torch::zeros({gridnum, gridnum, gridnum}, options);

    float extent_min = extent[0].item<float>();
    float extent_max = extent[1].item<float>();
    float boxsize = extent_max - extent_min;
    float cellSize = boxsize / static_cast<float>(gridnum);
    float inv_cellSize = 1.0f / cellSize;

    float* __restrict__ fields_ptr = fields.data_ptr<float>();
    float* __restrict__ weights_ptr = weights.data_ptr<float>();
    float* __restrict__ pos_ptr = pos.data_ptr<float>();
    float* __restrict__ quant_ptr = quantities.data_ptr<float>();

    int field_stride_x = gridnum * gridnum * num_fields;
    int field_stride_y = gridnum * num_fields;
    int field_stride_z = num_fields;

    int weight_stride_x = gridnum * gridnum;
    int weight_stride_y = gridnum;

    int N = pos.size(0);

    //#pragma omp parallel for // schedule(static)
    for (int n = 0; n < N; ++n) {
        float xpos = (pos_ptr[n*3 + 0] - extent_min) * inv_cellSize;
        float ypos = (pos_ptr[n*3 + 1] - extent_min) * inv_cellSize;
        float zpos = (pos_ptr[n*3 + 2] - extent_min) * inv_cellSize;

        int ix = static_cast<int>(floorf(xpos));
        int iy = static_cast<int>(floorf(ypos));
        int iz = static_cast<int>(floorf(zpos));

        int base_idx = ix * field_stride_x + iy * field_stride_y + iz * field_stride_z;
        int weight_idx = ix * weight_stride_x + iy * weight_stride_y + iz;

        for (int f = 0; f < num_fields; ++f) {
            float val = quant_ptr[n * num_fields + f];
            
            //#pragma omp atomic
            fields_ptr[base_idx + f] += val;
        }
        //#pragma omp atomic
        weights_ptr[weight_idx] += 1.0f;
    }
    return {fields, weights};
}


std::vector<at::Tensor> cic_2d(
    at::Tensor pos,         // (N, 2)
    at::Tensor quantities,  // (N, num_fields)
    at::Tensor extent,      // [min, max]
    int gridnum,
    bool periodic
) {
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(pos.device());
    int num_fields = quantities.size(1);

    at::Tensor fields = torch::zeros({gridnum, gridnum, num_fields}, options);
    at::Tensor weights = torch::zeros({gridnum, gridnum}, options);

    float extent_min = extent[0].item<float>();
    float extent_max = extent[1].item<float>();
    float cellSize = (extent_max - extent_min) / static_cast<float>(gridnum);

    float* __restrict__ fields_ptr = fields.data_ptr<float>();
    float* __restrict__ weights_ptr = weights.data_ptr<float>();
    float* __restrict__ pos_ptr = pos.data_ptr<float>();
    float* __restrict__ quant_ptr = quantities.data_ptr<float>();

    int stride_x = gridnum * num_fields;
    int stride_y = num_fields;

    int weight_stride_x = gridnum;
    int weight_stride_y = 1;

    int N = pos.size(0);

    for (int n = 0; n < N; ++n) {
        float xpos = (pos_ptr[n * 2 + 0] - extent_min) / cellSize;
        float ypos = (pos_ptr[n * 2 + 1] - extent_min) / cellSize;

        if (xpos < 0.0f || xpos > gridnum ||
            ypos < 0.0f || ypos > gridnum)
            continue;

        int i  = static_cast<int>(xpos + 0.4999f);
        int j  = static_cast<int>(ypos + 0.4999f);
        int i_ = i - 1;
        int j_ = j - 1;

        float dx = (xpos + 0.5f) - static_cast<float>(i);
        float dy = (ypos + 0.5f) - static_cast<float>(j);
        float dx_ = 1.0f - dx;
        float dy_ = 1.0f - dy;

        if (!periodic) {
            if (xpos < 0.5f || xpos > gridnum - 0.5f ||
                ypos < 0.5f || ypos > gridnum - 0.5f)
                continue;
        }

        auto [i0, q0] = account_for_pbc(i_, gridnum, periodic);
        auto [i1, q1] = account_for_pbc(i,  gridnum, periodic);
        auto [j0, q2] = account_for_pbc(j_, gridnum, periodic);
        auto [j1, q3] = account_for_pbc(j,  gridnum, periodic);

        float w00 = dx_ * dy_;
        float w10 = dx  * dy_;
        float w01 = dx_ * dy;
        float w11 = dx  * dy;

        auto deposit = [&](int ix, int jy, float w) {
            if (!periodic && (ix < 0 || ix >= gridnum || jy < 0 || jy >= gridnum)) return;

            int base_idx = ix * stride_x + jy * stride_y;
            int weight_idx = ix * weight_stride_x + jy * weight_stride_y;

            for (int f = 0; f < num_fields; ++f) {
                fields_ptr[base_idx + f] += w * quant_ptr[n * num_fields + f];
            }

            weights_ptr[weight_idx] += w;
        };

        // Explicit bilinear stencil
        deposit(i0, j0, w00);
        deposit(i1, j0, w10);
        deposit(i0, j1, w01);
        deposit(i1, j1, w11);
    }

    return {fields, weights};
}


std::vector<at::Tensor> cic_3d(
    at::Tensor pos,         // (N, 3)
    at::Tensor quantities,  // (N, num_fields)
    at::Tensor extent,      // [min, max]
    int gridnum,
    bool periodic
) {
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(pos.device());
    int num_fields = quantities.size(1);

    at::Tensor fields = torch::zeros({gridnum, gridnum, gridnum, num_fields}, options);
    at::Tensor weights = torch::zeros({gridnum, gridnum, gridnum}, options);

    float extent_min = extent[0].item<float>();
    float extent_max = extent[1].item<float>();
    float cellSize = (extent_max - extent_min) / static_cast<float>(gridnum);

    float* __restrict__ fields_ptr = fields.data_ptr<float>();
    float* __restrict__ weights_ptr = weights.data_ptr<float>();
    float* __restrict__ pos_ptr = pos.data_ptr<float>();
    float* __restrict__ quant_ptr = quantities.data_ptr<float>();

    int stride_x = gridnum * gridnum * num_fields;
    int stride_y = gridnum * num_fields;
    int stride_z = num_fields;

    int weight_stride_x = gridnum * gridnum;
    int weight_stride_y = gridnum;

    int N = pos.size(0);

    //#ifdef _OPENMP
    //#pragma omp parallel for schedule(static)
    //#endif
    for (int n = 0; n < N; ++n) {
        float xpos = (pos_ptr[n * 3 + 0] - extent_min) / cellSize;
        float ypos = (pos_ptr[n * 3 + 1] - extent_min) / cellSize;
        float zpos = (pos_ptr[n * 3 + 2] - extent_min) / cellSize;

        // Skip particles outside domain
        if (xpos < 0.0f || xpos > gridnum ||
            ypos < 0.0f || ypos > gridnum ||
            zpos < 0.0f || zpos > gridnum)
            continue;

        int i  = static_cast<int>(xpos + 0.4999f);
        int j  = static_cast<int>(ypos + 0.4999f);
        int k  = static_cast<int>(zpos + 0.4999f);
        int i_ = i - 1;
        int j_ = j - 1;
        int k_ = k - 1;

        float dx = (xpos + 0.5f) - static_cast<float>(i);
        float dy = (ypos + 0.5f) - static_cast<float>(j);
        float dz = (zpos + 0.5f) - static_cast<float>(k);
        float dx_ = 1.0f - dx;
        float dy_ = 1.0f - dy;
        float dz_ = 1.0f - dz;

        if (!periodic) {
            if (xpos < 0.5f || xpos > gridnum - 0.5f ||
                ypos < 0.5f || ypos > gridnum - 0.5f ||
                zpos < 0.5f || zpos > gridnum - 0.5f)
                continue;
        }

        // PBC-corrected indices
        auto [i0, q0] = account_for_pbc(i_, gridnum, periodic);
        auto [i1, q1] = account_for_pbc(i,  gridnum, periodic);
        auto [j0, q2] = account_for_pbc(j_, gridnum, periodic);
        auto [j1, q3] = account_for_pbc(j,  gridnum, periodic);
        auto [k0, q4] = account_for_pbc(k_, gridnum, periodic);
        auto [k1, q5] = account_for_pbc(k,  gridnum, periodic);

        // Weights
        float w000 = dx_ * dy_ * dz_;
        float w100 = dx  * dy_ * dz_;
        float w010 = dx_ * dy  * dz_;
        float w110 = dx  * dy  * dz_;
        float w001 = dx_ * dy_ * dz;
        float w101 = dx  * dy_ * dz;
        float w011 = dx_ * dy  * dz;
        float w111 = dx  * dy  * dz;

        // Helper lambda to deposit
        auto deposit = [&](int ix, int jy, int kz, float w) {
            if (!periodic && (ix < 0 || ix >= gridnum || jy < 0 || jy >= gridnum || kz < 0 || kz >= gridnum)) return;

            int base_idx = ix * stride_x + jy * stride_y + kz * stride_z;
            int weight_idx = ix * weight_stride_x + jy * weight_stride_y + kz;

            for (int f = 0; f < num_fields; ++f) {
            //#ifdef _OPENMP
            //#pragma omp atomic
            //#endif
                fields_ptr[base_idx + f] += w * quant_ptr[n * num_fields + f];
            }

            //#ifdef _OPENMP
            //#pragma omp atomic
            //#endif
            weights_ptr[weight_idx] += w;
        };

        // Explicitly apply CIC stencil
        deposit(i0, j0, k0, w000);
        deposit(i1, j0, k0, w100);
        deposit(i0, j1, k0, w010);
        deposit(i1, j1, k0, w110);
        deposit(i0, j0, k1, w001);
        deposit(i1, j0, k1, w101);
        deposit(i0, j1, k1, w011);
        deposit(i1, j1, k1, w111);
    }

    return {fields, weights};
}


std::vector<at::Tensor> cic_2d_adaptive(
    at::Tensor pos,               // (N, 2)
    at::Tensor quantities,        // (N, num_fields)
    at::Tensor extent,            // [min, max]
    int gridnum,
    bool periodic,
    at::Tensor pcellsizesHalf     // (N,)
) {
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(pos.device());
    int num_fields = quantities.size(1);

    at::Tensor fields = torch::zeros({gridnum, gridnum, num_fields}, options);
    at::Tensor weights = torch::zeros({gridnum, gridnum}, options);

    float extent_min = extent[0].item<float>();
    float extent_max = extent[1].item<float>();
    float cellSize = (extent_max - extent_min) / static_cast<float>(gridnum);

    float* __restrict__ fields_ptr = fields.data_ptr<float>();
    float* __restrict__ weights_ptr = weights.data_ptr<float>();
    float* __restrict__ pos_ptr = pos.data_ptr<float>();
    float* __restrict__ quant_ptr = quantities.data_ptr<float>();
    float* __restrict__ pcs_ptr = pcellsizesHalf.data_ptr<float>();

    int stride_x = gridnum * num_fields;
    int stride_y = num_fields;

    int weight_stride_x = gridnum;

    int N = pos.size(0);

    for (int n = 0; n < N; ++n) {
        float pcs = pcs_ptr[n] / cellSize;
        float V = std::pow(2.0f * pcs, 2.0f);

        float xpos = (pos_ptr[n * 2 + 0] - extent_min) / cellSize;
        float ypos = (pos_ptr[n * 2 + 1] - extent_min) / cellSize;

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

                auto [an, fx] = account_for_pbc(a, gridnum, periodic);
                auto [bn, fy] = account_for_pbc(b, gridnum, periodic);

                if (fx == 0.0f || fy == 0.0f) continue;

                int base_idx = an * stride_x + bn * stride_y;
                int weight_idx = an * weight_stride_x + bn;

                for (int f = 0; f < num_fields; ++f) {
                    fields_ptr[base_idx + f] += fraction * quant_ptr[n * num_fields + f];
                }

                weights_ptr[weight_idx] += fraction;
            }
        }
    }

    return {fields, weights};
}


std::vector<at::Tensor> cic_3d_adaptive(
    at::Tensor pos,               // (N, 3)
    at::Tensor quantities,        // (N, num_fields)
    at::Tensor extent,            // [min, max]
    int gridnum,
    bool periodic,
    at::Tensor pcellsizesHalf     // (N,)
) {
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(pos.device());
    int num_fields = quantities.size(1);

    at::Tensor fields = torch::zeros({gridnum, gridnum, gridnum, num_fields}, options);
    at::Tensor weights = torch::zeros({gridnum, gridnum, gridnum}, options);

    float extent_min = extent[0].item<float>();
    float extent_max = extent[1].item<float>();
    float cellSize = (extent_max - extent_min) / static_cast<float>(gridnum);

    float* __restrict__ fields_ptr = fields.data_ptr<float>();
    float* __restrict__ weights_ptr = weights.data_ptr<float>();
    float* __restrict__ pos_ptr = pos.data_ptr<float>();
    float* __restrict__ quant_ptr = quantities.data_ptr<float>();
    float* __restrict__ pcs_ptr = pcellsizesHalf.data_ptr<float>();

    int stride_x = gridnum * gridnum * num_fields;
    int stride_y = gridnum * num_fields;
    int stride_z = num_fields;

    int weight_stride_x = gridnum * gridnum;
    int weight_stride_y = gridnum;

    int N = pos.size(0);

    for (int n = 0; n < N; ++n) {
        float pcs = pcs_ptr[n] / cellSize;
        float V = std::pow(2.0f * pcs, 3.0f);

        float xpos = (pos_ptr[n * 3 + 0] - extent_min) / cellSize;
        float ypos = (pos_ptr[n * 3 + 1] - extent_min) / cellSize;
        float zpos = (pos_ptr[n * 3 + 2] - extent_min) / cellSize;

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

                    // Use helper to correct for PBC
                    auto [an, fx] = account_for_pbc(a, gridnum, periodic);
                    auto [bn, fy] = account_for_pbc(b, gridnum, periodic);
                    auto [cn, fz] = account_for_pbc(c, gridnum, periodic);

                    // If any of the axes was out-of-bounds and not periodic, skip deposit
                    if (fx == 0.0f || fy == 0.0f || fz == 0.0f) continue;

                    int base_idx = an * stride_x + bn * stride_y + cn * stride_z;
                    int weight_idx = an * weight_stride_x + bn * weight_stride_y + cn;

                    for (int f = 0; f < num_fields; ++f) {
                        fields_ptr[base_idx + f] += fraction * quant_ptr[n * num_fields + f];
                    }

                    weights_ptr[weight_idx] += fraction;
                }
            }
        }
    }

    return {fields, weights};
}


std::vector<at::Tensor> tsc_2d(
    at::Tensor pos,         // (N, 2)
    at::Tensor quantities,  // (N, num_fields)
    at::Tensor extent,      // [min, max]
    int gridnum,
    bool periodic
) {
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(pos.device());
    int num_fields = quantities.size(1);

    at::Tensor fields = torch::zeros({gridnum, gridnum, num_fields}, options);
    at::Tensor weights = torch::zeros({gridnum, gridnum}, options);

    float extent_min = extent[0].item<float>();
    float extent_max = extent[1].item<float>();
    float cellSize = (extent_max - extent_min) / static_cast<float>(gridnum);

    float* __restrict__ fields_ptr = fields.data_ptr<float>();
    float* __restrict__ weights_ptr = weights.data_ptr<float>();
    float* __restrict__ pos_ptr = pos.data_ptr<float>();
    float* __restrict__ quant_ptr = quantities.data_ptr<float>();

    int stride_x = gridnum * num_fields;
    int stride_y = num_fields;

    int weight_stride_x = gridnum;
    int weight_stride_y = 1;

    int N = pos.size(0);

    auto tsc_weight = [](float d) -> float {
        d = std::fabs(d);
        if (d < 0.5f) {
            return 0.75f - d * d;
        } else if (d < 1.5f) {
            return 0.5f * (1.5f - d) * (1.5f - d);
        } else {
            return 0.0f;
        }
    };

    for (int n = 0; n < N; ++n) {
        float xpos = (pos_ptr[n * 2 + 0] - extent_min) / cellSize;
        float ypos = (pos_ptr[n * 2 + 1] - extent_min) / cellSize;

        int i_center = static_cast<int>(std::floor(xpos));
        int j_center = static_cast<int>(std::floor(ypos));

        for (int di = -1; di <= 1; ++di) {
            for (int dj = -1; dj <= 1; ++dj) {
                int ix = i_center + di;
                int jy = j_center + dj;

                float dx = static_cast<float>(ix) + 0.5f - xpos;
                float dy = static_cast<float>(jy) + 0.5f - ypos;

                float wx = tsc_weight(dx);
                float wy = tsc_weight(dy);
                float w = wx * wy;

                if (w == 0.0f) continue;

                if (!periodic &&
                    (ix < 0 || ix >= gridnum ||
                     jy < 0 || jy >= gridnum)) continue;

                auto [ix_corr, fx] = account_for_pbc(ix, gridnum, periodic);
                auto [jy_corr, fy] = account_for_pbc(jy, gridnum, periodic);

                if (!periodic && (fx == 0.0f || fy == 0.0f)) continue;

                float corrected_w = (fx < 0.0f ? 1.0f : fx) *
                                    (fy < 0.0f ? 1.0f : fy) *
                                    w;

                int base_idx = ix_corr * stride_x + jy_corr * stride_y;
                int weight_idx = ix_corr * weight_stride_x + jy_corr * weight_stride_y;

                for (int f = 0; f < num_fields; ++f) {
                    fields_ptr[base_idx + f] += corrected_w * quant_ptr[n * num_fields + f];
                }

                weights_ptr[weight_idx] += corrected_w;
            }
        }
    }

    return {fields, weights};
}


std::vector<at::Tensor> tsc_3d(
    at::Tensor pos,         // (N, 3)
    at::Tensor quantities,  // (N, num_fields)
    at::Tensor extent,      // [min, max]
    int gridnum,
    bool periodic
) {
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(pos.device());
    int num_fields = quantities.size(1);

    at::Tensor fields = torch::zeros({gridnum, gridnum, gridnum, num_fields}, options);
    at::Tensor weights = torch::zeros({gridnum, gridnum, gridnum}, options);

    float extent_min = extent[0].item<float>();
    float extent_max = extent[1].item<float>();
    float cellSize = (extent_max - extent_min) / static_cast<float>(gridnum);

    float* __restrict__ fields_ptr = fields.data_ptr<float>();
    float* __restrict__ weights_ptr = weights.data_ptr<float>();
    float* __restrict__ pos_ptr = pos.data_ptr<float>();
    float* __restrict__ quant_ptr = quantities.data_ptr<float>();

    int stride_x = gridnum * gridnum * num_fields;
    int stride_y = gridnum * num_fields;
    int stride_z = num_fields;

    int weight_stride_x = gridnum * gridnum;
    int weight_stride_y = gridnum;

    int N = pos.size(0);

    // TSC weight kernel function
    auto tsc_weight = [](float d) -> float {
        d = std::fabs(d);
        if (d < 0.5f) {
            return 0.75f - d * d;
        } else if (d < 1.5f) {
            return 0.5f * (1.5f - d) * (1.5f - d);
        } else {
            return 0.0f;
        }
    };

    for (int n = 0; n < N; ++n) {
        float xpos = (pos_ptr[n * 3 + 0] - extent_min) / cellSize;
        float ypos = (pos_ptr[n * 3 + 1] - extent_min) / cellSize;
        float zpos = (pos_ptr[n * 3 + 2] - extent_min) / cellSize;

        int i_center = static_cast<int>(std::floor(xpos));
        int j_center = static_cast<int>(std::floor(ypos));
        int k_center = static_cast<int>(std::floor(zpos));

        for (int di = -1; di <= 1; ++di) {
            for (int dj = -1; dj <= 1; ++dj) {
                for (int dk = -1; dk <= 1; ++dk) {
                    int ix = i_center + di;
                    int jy = j_center + dj;
                    int kz = k_center + dk;

                    float dx = static_cast<float>(ix) + 0.5f - xpos;
                    float dy = static_cast<float>(jy) + 0.5f - ypos;
                    float dz = static_cast<float>(kz) + 0.5f - zpos;

                    float wx = tsc_weight(dx);
                    float wy = tsc_weight(dy);
                    float wz = tsc_weight(dz);
                    float w = wx * wy * wz;

                    if (w == 0.0f) continue;

                    if (!periodic &&
                        (ix < 0 || ix >= gridnum ||
                         jy < 0 || jy >= gridnum ||
                         kz < 0 || kz >= gridnum)) continue;

                    auto [ix_corr, fx] = account_for_pbc(ix, gridnum, periodic);
                    auto [jy_corr, fy] = account_for_pbc(jy, gridnum, periodic);
                    auto [kz_corr, fz] = account_for_pbc(kz, gridnum, periodic);

                    if (!periodic && (fx == 0.0f || fy == 0.0f || fz == 0.0f)) continue;
                    float corrected_w = (fx < 0.0f ? 1.0f : fx) *
                                        (fy < 0.0f ? 1.0f : fy) *
                                        (fz < 0.0f ? 1.0f : fz) *
                                        w;

                    int base_idx = ix_corr * stride_x + jy_corr * stride_y + kz_corr * stride_z;
                    int weight_idx = ix_corr * weight_stride_x + jy_corr * weight_stride_y + kz_corr;

                    for (int f = 0; f < num_fields; ++f) {
                        fields_ptr[base_idx + f] += corrected_w * quant_ptr[n * num_fields + f];
                    }
                    weights_ptr[weight_idx] += corrected_w;
                }
            }
        }
    }

    return {fields, weights};
}


float tsc_cdf_1d(float z, float h) {
    float abs_z = std::abs(z);
    float s = z < 0 ? -1.0f : 1.0f;
    float x = abs_z / h;

    if (x >= 1.5f) return s > 0 ? 1.0f : 0.0f;

    float integral = 0.0f;

    if (x < 0.5f) {
        integral = (0.75f * x - (x * x * x) / 3.0f) * h;
    } else if (x < 1.5f) {
        float a = 1.5f - x;
        float I = 0.5f * a * a * h;
        integral = 0.5f * h - I;
    } else {
        integral = 0.5f * h;
    }

    return s > 0 ? 0.5f + integral / h : 0.5f - integral / h;
}

float tsc_integrated_weight_1d(float x_center, float cell_left, float cell_right, float h) {
    return tsc_cdf_1d(cell_right - x_center, h) - tsc_cdf_1d(cell_left - x_center, h);
}


std::vector<at::Tensor> tsc_2d_adaptive(
    at::Tensor pos,             // (N, 2)
    at::Tensor quantities,      // (N, num_fields)
    at::Tensor extent,          // [min, max]
    int gridnum,
    bool periodic,
    at::Tensor pcellsizesHalf  // (N)
) {
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(pos.device());
    int num_fields = quantities.size(1);
    int N = pos.size(0);

    at::Tensor fields = torch::zeros({gridnum, gridnum, num_fields}, options);
    at::Tensor weights = torch::zeros({gridnum, gridnum}, options);

    float* __restrict__ fields_ptr = fields.data_ptr<float>();
    float* __restrict__ weights_ptr = weights.data_ptr<float>();
    float* __restrict__ pos_ptr = pos.data_ptr<float>();
    float* __restrict__ quant_ptr = quantities.data_ptr<float>();
    float* __restrict__ pcs_ptr = pcellsizesHalf.data_ptr<float>();

    float extent_min = extent[0].item<float>();
    float extent_max = extent[1].item<float>();
    float cellSize = (extent_max - extent_min) / static_cast<float>(gridnum);

    int stride_x = gridnum * num_fields;
    int stride_y = num_fields;

    int weight_stride_x = gridnum;
    int weight_stride_y = 1;

    for (int n = 0; n < N; ++n) {
        float x = (pos_ptr[n * 2 + 0] - extent_min) / cellSize;
        float y = (pos_ptr[n * 2 + 1] - extent_min) / cellSize;
        float pcs = pcs_ptr[n] / cellSize;

        float kernel_support = 1.5f * pcs;
        int i_min = static_cast<int>(std::floor(x - kernel_support));
        int i_max = static_cast<int>(std::ceil(x + kernel_support));
        int j_min = static_cast<int>(std::floor(y - kernel_support));
        int j_max = static_cast<int>(std::ceil(y + kernel_support));

        for (int i = i_min; i <= i_max; ++i) {
            auto [ii, fx] = account_for_pbc(i, gridnum, periodic);
            if (!periodic && fx == 0.0f) continue;

            float x_left = static_cast<float>(i);
            float x_right = static_cast<float>(i + 1);
            float wx = tsc_integrated_weight_1d(x, x_left, x_right, pcs);
            if (wx == 0.0f) continue;

            for (int j = j_min; j <= j_max; ++j) {
                auto [jj, fy] = account_for_pbc(j, gridnum, periodic);
                if (!periodic && fy == 0.0f) continue;

                float y_left = static_cast<float>(j);
                float y_right = static_cast<float>(j + 1);
                float wy = tsc_integrated_weight_1d(y, y_left, y_right, pcs);
                if (wy == 0.0f) continue;

                float w = wx * wy;

                int base_idx = ii * stride_x + jj * stride_y;
                int weight_idx = ii * weight_stride_x + jj * weight_stride_y;

                for (int f = 0; f < num_fields; ++f) {
                    fields_ptr[base_idx + f] += quant_ptr[n * num_fields + f] * w;
                }
                weights_ptr[weight_idx] += w;
            }
        }
    }

    return {fields, weights};
}


std::vector<at::Tensor> tsc_3d_adaptive(
    at::Tensor pos,             // (N, 3)
    at::Tensor quantities,      // (N, num_fields)
    at::Tensor extent,          // [min, max]
    int gridnum,
    bool periodic,
    at::Tensor pcellsizesHalf  // (N)
) {
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(pos.device());
    int num_fields = quantities.size(1);
    int N = pos.size(0);

    at::Tensor fields = torch::zeros({gridnum, gridnum, gridnum, num_fields}, options);
    at::Tensor weights = torch::zeros({gridnum, gridnum, gridnum}, options);

    float* __restrict__ fields_ptr = fields.data_ptr<float>();
    float* __restrict__ weights_ptr = weights.data_ptr<float>();
    float* __restrict__ pos_ptr = pos.data_ptr<float>();
    float* __restrict__ quant_ptr = quantities.data_ptr<float>();
    float* __restrict__ pcs_ptr = pcellsizesHalf.data_ptr<float>();

    float extent_min = extent[0].item<float>();
    float extent_max = extent[1].item<float>();
    float cellSize = (extent_max - extent_min) / static_cast<float>(gridnum);

    int stride_x = gridnum * gridnum * num_fields;
    int stride_y = gridnum * num_fields;
    int stride_z = num_fields;

    int weight_stride_x = gridnum * gridnum;
    int weight_stride_y = gridnum;
    int weight_stride_z = 1;

    for (int n = 0; n < N; ++n) {
        float x = (pos_ptr[n * 3 + 0] - extent_min) / cellSize;
        float y = (pos_ptr[n * 3 + 1] - extent_min) / cellSize;
        float z = (pos_ptr[n * 3 + 2] - extent_min) / cellSize;
        float pcs = pcs_ptr[n] / cellSize;

        float kernel_support = 1.5f * pcs;
        int i_min = static_cast<int>(std::floor(x - kernel_support));
        int i_max = static_cast<int>(std::ceil(x + kernel_support));
        int j_min = static_cast<int>(std::floor(y - kernel_support));
        int j_max = static_cast<int>(std::ceil(y + kernel_support));
        int k_min = static_cast<int>(std::floor(z - kernel_support));
        int k_max = static_cast<int>(std::ceil(z + kernel_support));

        for (int i = i_min; i <= i_max; ++i) {
            auto [ii, fx] = account_for_pbc(i, gridnum, periodic);
            if (!periodic && fx == 0.0f) continue;

            float x_left = static_cast<float>(i);
            float x_right = static_cast<float>(i + 1);
            float wx = tsc_integrated_weight_1d(x, x_left, x_right, pcs);
            if (wx == 0.0f) continue;

            for (int j = j_min; j <= j_max; ++j) {
                auto [jj, fy] = account_for_pbc(j, gridnum, periodic);
                if (!periodic && fy == 0.0f) continue;

                float y_left = static_cast<float>(j);
                float y_right = static_cast<float>(j + 1);
                float wy = tsc_integrated_weight_1d(y, y_left, y_right, pcs);
                if (wy == 0.0f) continue;

                for (int k = k_min; k <= k_max; ++k) {
                    auto [kk, fz] = account_for_pbc(k, gridnum, periodic);
                    if (!periodic && fz == 0.0f) continue;

                    float z_left = static_cast<float>(k);
                    float z_right = static_cast<float>(k + 1);
                    float wz = tsc_integrated_weight_1d(z, z_left, z_right, pcs);
                    if (wz == 0.0f) continue;

                    float w = wx * wy * wz;

                    int base_idx = ii * stride_x + jj * stride_y + kk * stride_z;
                    int weight_idx = ii * weight_stride_x + jj * weight_stride_y + kk * weight_stride_z;

                    for (int f = 0; f < num_fields; ++f) {
                        fields_ptr[base_idx + f] += quant_ptr[n * num_fields + f] * w;
                    }
                    weights_ptr[weight_idx] += w;
                }
            }
        }
    }

    return {fields, weights};
}

//===========================================================================================

float compute_fraction_isotropic_2d(
    const std::string& method,
    float xpos, float ypos,
    int a, int b,
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

        // Center (weight 4)
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

        // Edges (weight 2)
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

        // Corners (weight 1)
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


float compute_fraction_isotropic_3d(
    const std::string& method,
    float xpos, float ypos, float zpos,
    int a, int b, int c,
    int gridnum,
    bool periodic,
    float hsn,         // smoothing length in grid units
    SPHKernel* kernel
) {
    float sigma = kernel->normalization(hsn);

    // --- Midpoint ---
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
        return kernel->weight(r, hsn) * sigma;
    }

    // --- Trapezoidal rule: evaluate kernel at 6 face centers ---
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
            sum += kernel->weight(r, hsn);
        }

        return (sum / 6.0f) * sigma;
    }

    // --- Simpson’s rule: center (×8), faces (×4), corners (×1) ---
    if (method == "simpson") {
        float sum = 0.0f;

        // Center (weight 8)
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
            sum += 8.0f * kernel->weight(r, hsn);
        }

        // Faces (weight 4)
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
            sum += 4.0f * kernel->weight(r, hsn);
        }

        // Corners (weight 1)
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
                    sum += kernel->weight(r, hsn);
                }
            }
        }

        return (sum / 40.0f) * sigma;
    }

    // Fallback
    throw std::invalid_argument("Unknown integration method in isotropic 3D");
}


float compute_fraction_anisotropic_2d(
    const std::string& method,
    const float* vecs,         // 2×2 = 4 values, row-major
    const float* vals_gu,      // 2 values
    float xpos, float ypos,
    int a, int b,
    int gridnum,
    bool periodic,
    SPHKernel* kernel
) {
    float detH = vals_gu[0] * vals_gu[1];
    float sigma = kernel->normalization(detH);

    // --- Midpoint ---
    if (method == "midpoint") {
        float dx = xpos - (a + 0.5f);
        float dy = ypos - (b + 0.5f);

        if (periodic) {
            if (dx > gridnum / 2.0f) dx -= gridnum;
            if (dy > gridnum / 2.0f) dy -= gridnum;
        }

        float xi_1 = (vecs[0] * dx + vecs[1] * dy) / vals_gu[0];
        float xi_2 = (vecs[2] * dx + vecs[3] * dy) / vals_gu[1];
        float q = std::sqrt(xi_1 * xi_1 + xi_2 * xi_2);

        return kernel->weight(q, 1.0f) * sigma;
    }

    // --- Trapezoidal: edges of the square ---
    if (method == "trapezoidal") {
        float sum = 0.0f;
        const float offsets[4][2] = {
            {0.0f, 0.5f}, {1.0f, 0.5f},
            {0.5f, 0.0f}, {0.5f, 1.0f}
        };
        for (int i = 0; i < 4; ++i) {
            float dx = xpos - (a + offsets[i][0]);
            float dy = ypos - (b + offsets[i][1]);

            if (periodic) {
                if (dx > gridnum / 2.0f) dx -= gridnum;
                if (dy > gridnum / 2.0f) dy -= gridnum;
            }

            float xi_1 = (vecs[0] * dx + vecs[1] * dy) / vals_gu[0];
            float xi_2 = (vecs[2] * dx + vecs[3] * dy) / vals_gu[1];
            float q = std::sqrt(xi_1 * xi_1 + xi_2 * xi_2);

            sum += kernel->weight(q, 1.0f);
        }

        return (sum / 4.0f) * sigma;
    }

    // --- Simpson's rule ---
    if (method == "simpson") {
        float sum = 0.0f;

        // Center point (weight 4)
        {
            float dx = xpos - (a + 0.5f);
            float dy = ypos - (b + 0.5f);

            if (periodic) {
                if (dx > gridnum / 2.0f) dx -= gridnum;
                if (dy > gridnum / 2.0f) dy -= gridnum;
            }

            float xi_1 = (vecs[0] * dx + vecs[1] * dy) / vals_gu[0];
            float xi_2 = (vecs[2] * dx + vecs[3] * dy) / vals_gu[1];
            float q = std::sqrt(xi_1 * xi_1 + xi_2 * xi_2);

            sum += 4.0f * kernel->weight(q, 1.0f);
        }

        // Edge centers (weight 2)
        const float edges[4][2] = {
            {0.0f, 0.5f}, {1.0f, 0.5f},
            {0.5f, 0.0f}, {0.5f, 1.0f}
        };
        for (int i = 0; i < 4; ++i) {
            float dx = xpos - (a + edges[i][0]);
            float dy = ypos - (b + edges[i][1]);

            if (periodic) {
                if (dx > gridnum / 2.0f) dx -= gridnum;
                if (dy > gridnum / 2.0f) dy -= gridnum;
            }

            float xi_1 = (vecs[0] * dx + vecs[1] * dy) / vals_gu[0];
            float xi_2 = (vecs[2] * dx + vecs[3] * dy) / vals_gu[1];
            float q = std::sqrt(xi_1 * xi_1 + xi_2 * xi_2);

            sum += 2.0f * kernel->weight(q, 1.0f);
        }

        // Corners (weight 1)
        for (int dx_c = 0; dx_c <= 1; ++dx_c) {
            for (int dy_c = 0; dy_c <= 1; ++dy_c) {
                float dx = xpos - (a + dx_c);
                float dy = ypos - (b + dy_c);

                if (periodic) {
                    if (dx > gridnum / 2.0f) dx -= gridnum;
                    if (dy > gridnum / 2.0f) dy -= gridnum;
                }

                float xi_1 = (vecs[0] * dx + vecs[1] * dy) / vals_gu[0];
                float xi_2 = (vecs[2] * dx + vecs[3] * dy) / vals_gu[1];
                float q = std::sqrt(xi_1 * xi_1 + xi_2 * xi_2);

                sum += kernel->weight(q, 1.0f);
            }
        }

        return (sum / 12.0f) * sigma;  // 4 (center) + 4×2 (edges) + 4×1 (corners) = 28 → normalized by 12
    }

    // Unknown method
    throw std::invalid_argument("Unknown integration method: " + method);
}


float compute_fraction_anisotropic_3d(
    const std::string& method,
    const float* vecs,
    const float* vals_gu,
    float xpos, float ypos, float zpos,
    int a, int b, int c,
    int gridnum,
    bool periodic,
    SPHKernel* kernel
) {
    float detH = vals_gu[0] * vals_gu[1] * vals_gu[2];
    float sigma = kernel->normalization(detH);

    // Midpoint rule
    if (method == "midpoint") {
        float dx = xpos - (a + 0.5f);
        float dy = ypos - (b + 0.5f);
        float dz = zpos - (c + 0.5f);

        if (periodic) {
            if (dx > gridnum / 2.0f) dx -= gridnum;
            if (dy > gridnum / 2.0f) dy -= gridnum;
            if (dz > gridnum / 2.0f) dz -= gridnum;
        }

        float xi_1 = (vecs[0] * dx + vecs[1] * dy + vecs[2] * dz) / vals_gu[0];
        float xi_2 = (vecs[3] * dx + vecs[4] * dy + vecs[5] * dz) / vals_gu[1];
        float xi_3 = (vecs[6] * dx + vecs[7] * dy + vecs[8] * dz) / vals_gu[2];
        float q = std::sqrt(xi_1 * xi_1 + xi_2 * xi_2 + xi_3 * xi_3);

        return kernel->weight(q, 1.0f) * sigma;
    }

    // Trapezoidal rule (average of kernel values at centers of faces)
    if (method == "trapezoidal") {
        float sum = 0.0f;
        const float offsets[6][3] = {
            {0.0f, 0.5f, 0.5f},  // -x face
            {1.0f, 0.5f, 0.5f},  // +x face
            {0.5f, 0.0f, 0.5f},  // -y face
            {0.5f, 1.0f, 0.5f},  // +y face
            {0.5f, 0.5f, 0.0f},  // -z face
            {0.5f, 0.5f, 1.0f}   // +z face
        };
        for (int i = 0; i < 6; ++i) {
            float dx = xpos - (a + offsets[i][0]);
            float dy = ypos - (b + offsets[i][1]);
            float dz = zpos - (c + offsets[i][2]);

            if (periodic) {
            if (dx > gridnum / 2.0f) dx -= gridnum;
            if (dy > gridnum / 2.0f) dy -= gridnum;
            if (dz > gridnum / 2.0f) dz -= gridnum;
            }

            float xi_1 = (vecs[0] * dx + vecs[1] * dy + vecs[2] * dz) / vals_gu[0];
            float xi_2 = (vecs[3] * dx + vecs[4] * dy + vecs[5] * dz) / vals_gu[1];
            float xi_3 = (vecs[6] * dx + vecs[7] * dy + vecs[8] * dz) / vals_gu[2];
            float q = std::sqrt(xi_1 * xi_1 + xi_2 * xi_2 + xi_3 * xi_3);

            sum += kernel->weight(q, 1.0f);
        }
        return (sum / 6.0f) * sigma;
    }

    // Simpson's rule: evaluate at center, 6 face centers, 8 corners
    if (method == "simpson") {
        float sum = 0.0f;

        // Center point (weight 8)
        {
            float dx = xpos - (a + 0.5f);
            float dy = ypos - (b + 0.5f);
            float dz = zpos - (c + 0.5f);

            if (periodic) {
                if (dx > gridnum / 2.0f) dx -= gridnum;
                if (dy > gridnum / 2.0f) dy -= gridnum;
                if (dz > gridnum / 2.0f) dz -= gridnum;
            }

            float xi_1 = (vecs[0] * dx + vecs[1] * dy + vecs[2] * dz) / vals_gu[0];
            float xi_2 = (vecs[3] * dx + vecs[4] * dy + vecs[5] * dz) / vals_gu[1];
            float xi_3 = (vecs[6] * dx + vecs[7] * dy + vecs[8] * dz) / vals_gu[2];
            float q = std::sqrt(xi_1 * xi_1 + xi_2 * xi_2 + xi_3 * xi_3);

            sum += 8.0f * kernel->weight(q, 1.0f);
        }

        // Face centers (weight 4)
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

            float xi_1 = (vecs[0] * dx + vecs[1] * dy + vecs[2] * dz) / vals_gu[0];
            float xi_2 = (vecs[3] * dx + vecs[4] * dy + vecs[5] * dz) / vals_gu[1];
            float xi_3 = (vecs[6] * dx + vecs[7] * dy + vecs[8] * dz) / vals_gu[2];
            float q = std::sqrt(xi_1 * xi_1 + xi_2 * xi_2 + xi_3 * xi_3);

            sum += 4.0f * kernel->weight(q, 1.0f);
        }

        // Corners (weight 1)
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

                    float xi_1 = (vecs[0] * dx + vecs[1] * dy + vecs[2] * dz) / vals_gu[0];
                    float xi_2 = (vecs[3] * dx + vecs[4] * dy + vecs[5] * dz) / vals_gu[1];
                    float xi_3 = (vecs[6] * dx + vecs[7] * dy + vecs[8] * dz) / vals_gu[2];
                    float q = std::sqrt(xi_1 * xi_1 + xi_2 * xi_2 + xi_3 * xi_3);

                    sum += kernel->weight(q, 1.0f);
                }
            }
        }

        return (sum / 40.0f) * sigma;  // normalization: (8 + 6×4 + 8×1) = 27
    }

    // Unknown integration method
    throw std::invalid_argument("Unknown integration method: " + method);

}


std::vector<at::Tensor> isotropic_kernel_deposition_2d(
    at::Tensor pos, 
    at::Tensor quantities,
    at::Tensor extent, 
    int gridnum, 
    bool periodic, 
    at::Tensor hsm, 
    std::string kernel_name,
    std::string integration_method
) {
    int N = pos.size(0);
    int dim = pos.size(1);
    auto kernel = create_kernel(kernel_name, dim, false);

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(pos.device());
    int num_fields = quantities.size(1);

    at::Tensor fields  = torch::zeros({gridnum, gridnum, num_fields}, options);
    at::Tensor weights = torch::zeros({gridnum, gridnum}, options);

    float* pos_ptr     = pos.data_ptr<float>();
    float* hsm_ptr     = hsm.data_ptr<float>();
    float* quant_ptr   = quantities.data_ptr<float>();
    float* fields_ptr  = fields.data_ptr<float>();
    float* weights_ptr = weights.data_ptr<float>();

    float extent_min = extent[0].item<float>();
    float extent_max = extent[1].item<float>();
    float boxsize = extent_max - extent_min;
    float cellSize = boxsize / static_cast<float>(gridnum);

    int stride_x = gridnum * num_fields;
    int stride_y = num_fields;

    int weight_stride_x = gridnum;

    for (int n = 0; n < N; ++n) {
        float hsn = hsm_ptr[n] / cellSize;
        float support = kernel->support() * hsn;

        float xpos = (pos_ptr[n * 2 + 0] - extent_min) / cellSize;
        float ypos = (pos_ptr[n * 2 + 1] - extent_min) / cellSize;

        int i = static_cast<int>(xpos);
        int j = static_cast<int>(ypos);

        int num_left   = i - static_cast<int>(xpos - support);
        int num_right  = static_cast<int>(xpos + support + 0.5f) - i;
        int num_bottom = j - static_cast<int>(ypos - support);
        int num_top    = static_cast<int>(ypos + support + 0.5f) - j;

        for (int a = i - num_left; a <= i + num_right; ++a) {
            for (int b = j - num_bottom; b <= j + num_top; ++b) {

                float w = compute_fraction_isotropic_2d(
                                    integration_method, xpos, ypos,
                                    a, b, gridnum, periodic, hsn, kernel.get()
                                );
                
                int an = a, bn = b;
                if (periodic) {
                    an = (an + gridnum) % gridnum;
                    bn = (bn + gridnum) % gridnum;
                } else {
                    if (an < 0 || an >= gridnum || bn < 0 || bn >= gridnum)
                        continue;
                }

                int base_idx = an * stride_x + bn * stride_y;
                int weight_idx = an * weight_stride_x + bn;

                for (int f = 0; f < num_fields; ++f)
                    fields_ptr[base_idx + f] += quant_ptr[n * num_fields + f] * w;

                weights_ptr[weight_idx] += w;
            }
        }
    }

    return {fields, weights};
}


std::vector<at::Tensor> isotropic_kernel_deposition_3d(
    at::Tensor pos, 
    at::Tensor quantities,
    at::Tensor extent, 
    int gridnum, 
    bool periodic, 
    at::Tensor hsm, 
    std::string kernel_name,
    std::string integration_method
) {
    int N = pos.size(0);
    int dim = pos.size(1);
    auto kernel = create_kernel(kernel_name, dim, false);

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(pos.device());
    int num_fields = quantities.size(1);

    at::Tensor fields  = torch::zeros({gridnum, gridnum, gridnum, num_fields}, options);
    at::Tensor weights = torch::zeros({gridnum, gridnum, gridnum}, options);

    float* pos_ptr   = pos.data_ptr<float>();
    float* hsm_ptr   = hsm.data_ptr<float>();
    float* quant_ptr = quantities.data_ptr<float>();
    float* fields_ptr = fields.data_ptr<float>();
    float* weights_ptr = weights.data_ptr<float>();

    float extent_min = extent[0].item<float>();
    float extent_max = extent[1].item<float>();
    float boxsize = extent_max - extent_min;
    float cellSize = boxsize / static_cast<float>(gridnum);

    int stride_x = gridnum * gridnum * num_fields;
    int stride_y = gridnum * num_fields;
    int stride_z = num_fields;

    int weight_stride_x = gridnum * gridnum;
    int weight_stride_y = gridnum;
    
    for (int n = 0; n < N; ++n) {
        float hsn = hsm_ptr[n] / cellSize;
        float support = kernel->support() * hsn;

        float xpos = (pos_ptr[n * 3 + 0] - extent_min) / cellSize;
        float ypos = (pos_ptr[n * 3 + 1] - extent_min) / cellSize;
        float zpos = (pos_ptr[n * 3 + 2] - extent_min) / cellSize;

        int i = static_cast<int>(xpos);
        int j = static_cast<int>(ypos);
        int k = static_cast<int>(zpos);

        int num_left   = i - static_cast<int>(xpos - support);
        int num_right  = static_cast<int>(xpos + support + 0.5f) - i;
        int num_bottom = j - static_cast<int>(ypos - support);
        int num_top    = static_cast<int>(ypos + support + 0.5f) - j;
        int num_front  = k - static_cast<int>(zpos - support);
        int num_back   = static_cast<int>(zpos + support) - k;

        for (int a = i - num_left; a <= i + num_right; ++a) {
            for (int b = j - num_bottom; b <= j + num_top; ++b) {
                for (int c = k - num_front; c <= k + num_back; ++c) {
                    
                    int an = a, bn = b, cn = c;
                    if (periodic) {
                        an = (an + gridnum) % gridnum;
                        bn = (bn + gridnum) % gridnum;
                        cn = (cn + gridnum) % gridnum;
                    } else {
                        if (an < 0 || an >= gridnum || bn < 0 || bn >= gridnum || cn < 0 || cn >= gridnum)
                            continue;
                    }

                    float w = compute_fraction_isotropic_3d(
                        integration_method, xpos, ypos, zpos,
                        a, b, c, gridnum, periodic, hsn, kernel.get()
                    );

                    int base_idx = an * stride_x + bn * stride_y + cn * stride_z;
                    int weight_idx = an * weight_stride_x + bn * weight_stride_y + cn;
                    
                    for (int f = 0; f < num_fields; ++f) {
                        fields_ptr[base_idx + f] += quant_ptr[n * num_fields + f] * w;
                    }
                    weights_ptr[weight_idx] += w;

                    }
                }
            }
        }
    return {fields, weights};
}


std::vector<at::Tensor> anisotropic_kernel_deposition_2d(
    at::Tensor pos,                // [N, 2]
    at::Tensor quantities,         // [N, F]
    at::Tensor extent,             // [2]
    int gridnum,
    bool periodic,
    at::Tensor hmat_eigvecs,       // [N, 2, 2]
    at::Tensor hmat_eigvals,       // [N, 2]
    std::string kernel_name,
    std::string integration_method // ✅ Add this
) {
    int N = pos.size(0);
    int dim = pos.size(1);
    auto kernel = create_kernel(kernel_name, dim, true);

    int num_fields = quantities.size(1);

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(pos.device());
    at::Tensor fields  = torch::zeros({gridnum, gridnum, num_fields}, options);
    at::Tensor weights = torch::zeros({gridnum, gridnum}, options);

    float extent_min = extent[0].item<float>();
    float extent_max = extent[1].item<float>();
    float boxsize = extent_max - extent_min;
    float cellSize = boxsize / static_cast<float>(gridnum);

    float* pos_ptr     = pos.data_ptr<float>();
    float* eigvec_ptr  = hmat_eigvecs.data_ptr<float>();  // [N * 4]
    float* eigval_ptr  = hmat_eigvals.data_ptr<float>();  // [N * 2]
    float* quant_ptr   = quantities.data_ptr<float>();
    float* fields_ptr  = fields.data_ptr<float>();
    float* weights_ptr = weights.data_ptr<float>();

    int stride_x = gridnum * num_fields;
    int stride_y = num_fields;
    int weight_stride_x = gridnum;

    for (int n = 0; n < N; ++n) {
        const float* vecs = &eigvec_ptr[n * 4];  // 2x2
        const float* vals = &eigval_ptr[n * 2];  // 2

        /*
        float hvals[2] = { vals[0] / cellSize, vals[1] / cellSize };
        //float krs = std::max(hvals[0], hvals[1]) * 2.0f;
        */

        // scale the extent with the kernel support
        float vals_gu[2] = { vals[1] / cellSize, vals[1] / cellSize };
        float krs = kernel->support() * std::max({ vals_gu[0], vals_gu[1] });

        float xpos = (pos_ptr[n * 2 + 0] - extent_min) / cellSize;
        float ypos = (pos_ptr[n * 2 + 1] - extent_min) / cellSize;

        int i = static_cast<int>(xpos);
        int j = static_cast<int>(ypos);

        int num_left   = i - static_cast<int>(xpos - krs);
        int num_right  = static_cast<int>(xpos + krs + 0.5f) - i;

        int num_bottom = j - static_cast<int>(ypos - krs);
        int num_top    = static_cast<int>(ypos + krs + 0.5f) - j;

        for (int a = i - num_left; a <= i + num_right; ++a) {
            for (int b = j - num_bottom; b <= j + num_top; ++b) {
                
                float fraction = compute_fraction_anisotropic_2d(
                    integration_method, vecs, vals_gu,
                    xpos, ypos, a, b, gridnum, periodic, kernel.get()
                );
                
                int an = a, bn = b;
                bool in_bounds = true;
                if (periodic) {
                    an = (an + gridnum) % gridnum;
                    bn = (bn + gridnum) % gridnum;
                } else {
                    in_bounds = (an >= 0 && an < gridnum && bn >= 0 && bn < gridnum);
                }

                if (!in_bounds) continue;
                
                int base_idx = an * stride_x + bn * stride_y;
                int weight_idx = an * weight_stride_x + bn;

                for (int f = 0; f < num_fields; ++f)
                    fields_ptr[base_idx + f] += quant_ptr[n * num_fields + f] * fraction;

                weights_ptr[weight_idx] += fraction;
            }
        }
    }

    return {fields, weights};
}


std::vector<at::Tensor> anisotropic_kernel_deposition_3d(
    at::Tensor pos,
    at::Tensor quantities,
    at::Tensor extent,
    int gridnum,
    bool periodic,
    at::Tensor hmat_eigvecs,
    at::Tensor hmat_eigvals,
    std::string kernel_name,
    std::string integration_method
) {
    int N = pos.size(0);
    int dim = pos.size(1);
    auto kernel = create_kernel(kernel_name, dim, true);

    int num_fields = quantities.size(1);

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(pos.device());
    at::Tensor fields  = torch::zeros({gridnum, gridnum, gridnum, num_fields}, options);
    at::Tensor weights = torch::zeros({gridnum, gridnum, gridnum}, options);

    float extent_min = extent[0].item<float>();
    float extent_max = extent[1].item<float>();
    float boxsize = extent_max - extent_min;
    float cellSize = boxsize / static_cast<float>(gridnum);

    // Raw pointers
    float* pos_ptr     = pos.data_ptr<float>();
    float* eigvec_ptr  = hmat_eigvecs.data_ptr<float>();
    float* eigval_ptr  = hmat_eigvals.data_ptr<float>();
    float* quant_ptr   = quantities.data_ptr<float>();
    float* fields_ptr  = fields.data_ptr<float>();
    float* weights_ptr = weights.data_ptr<float>();

    int stride_x = gridnum * gridnum * num_fields;
    int stride_y = gridnum * num_fields;
    int stride_z = num_fields;

    int weight_stride_x = gridnum * gridnum;
    int weight_stride_y = gridnum;

    for (int n = 0; n < N; ++n) {
        const float* vecs = &eigvec_ptr[n * 9];    // 3x3 matrix
        const float* vals = &eigval_ptr[n * 3];    // 3 eigenvalues

        //float hsn = hsm_ptr[n] / cellSize;
        //float support_ = kernel->support() * hsn;

        //--------------
        /*
        float vecs_gu[9];
        for (int vi = 0; vi < 9; ++vi) {
            vecs_gu[vi] = vecs[vi] / cellSize;
        }

        //float hvals[3] = { vals[0] / cellSize, vals[1] / cellSize, vals[2] / cellSize };
        float max_extent[3] = {0.0f, 0.0f, 0.0f};
        
        for (int i = 0; i < 3; ++i) {
            float scaled_vec[3] = {
                vals_gu[0] * vecs_gu[0 * 3 + i],
                vals_gu[1] * vecs_gu[1 * 3 + i],
                vals_gu[2] * vecs_gu[2 * 3 + i]
            };
            max_extent[i] = (std::abs(scaled_vec[0]) + std::abs(scaled_vec[1]) + std::abs(scaled_vec[2]));
        }
        float ks = kernel->support();
        float support[3] = { 10.0f, 10.0f, 10.0f }; //ks * max_extent[0], ks * max_extent[1], ks * max_extent[2] };
        */
        //--------------

        // scale the extent with the kernel support
        float vals_gu[3] = { vals[0] / cellSize, vals[1] / cellSize, vals[2] / cellSize };
        float krs = kernel->support() * std::max({ vals_gu[0], vals_gu[1], vals_gu[2] });

        float xpos = (pos_ptr[n * 3 + 0] - extent_min) / cellSize;
        float ypos = (pos_ptr[n * 3 + 1] - extent_min) / cellSize;
        float zpos = (pos_ptr[n * 3 + 2] - extent_min) / cellSize;

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
                    
                    float fraction = compute_fraction_anisotropic_3d(
                        integration_method, vecs, vals_gu,
                        xpos, ypos, zpos, a, b, c, gridnum, periodic, kernel.get()
                    );
                    
                    int an = a, bn = b, cn = c;
                    bool in_bounds = true;
                    if (periodic) {
                        an = (an + gridnum) % gridnum;
                        bn = (bn + gridnum) % gridnum;
                        cn = (cn + gridnum) % gridnum;
                    } else {
                        in_bounds = (an >= 0 && an < gridnum &&
                                     bn >= 0 && bn < gridnum &&
                                     cn >= 0 && cn < gridnum);
                    }
                    if (!in_bounds) continue;

                    int base_idx = an * stride_x + bn * stride_y + cn * stride_z;
                    int weight_idx = an * weight_stride_x + bn * weight_stride_y + cn;

                    for (int f = 0; f < num_fields; ++f) {
                        fields_ptr[base_idx + f] += quant_ptr[n * num_fields + f] * fraction;
                    }
                    weights_ptr[weight_idx] += fraction;
                }
            }
        }
    }

    return {fields, weights};
}

//===========================================================================================
// Pybind
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

    //m.def("get_num_threads", &get_num_threads, "Get number of OpenMP threads");
    //m.def("set_num_threads", &set_num_threads, "Set number of OpenMP threads");

    m.def("ngp_2d", &ngp_2d, "2D NGP deposition (CPU)");
    m.def("ngp_3d", &ngp_3d, "3D NGP deposition (CPU)");

    m.def("cic_2d", &cic_2d, "2D CIC deposition (CPU)");
    m.def("cic_3d", &cic_3d, "3D CIC deposition (CPU)");

    m.def("cic_2d_adaptive", &cic_2d_adaptive, "2D adaptive CIC deposition (CPU)");
    m.def("cic_3d_adaptive", &cic_3d_adaptive, "3D adaptive CIC deposition (CPU)");

    m.def("tsc_2d", &tsc_2d, "2D TSC deposition (CPU)");
    m.def("tsc_3d", &tsc_3d, "3D TSC deposition (CPU)");

    m.def("tsc_2d_adaptive", &tsc_2d_adaptive, "2D adaptive TSC deposition (CPU)");
    m.def("tsc_3d_adaptive", &tsc_3d_adaptive, "3D adaptive TSC deposition (CPU)");

    //=======================================

    m.def("isotropic_kernel_deposition_2d", &isotropic_kernel_deposition_2d, "2D isotropic kernel deposition (CPU)");
    m.def("isotropic_kernel_deposition_3d", &isotropic_kernel_deposition_3d, "3D isotropic kernel deposition (CPU)");

    m.def("anisotropic_kernel_deposition_2d", &anisotropic_kernel_deposition_2d, "2D anisotropic kernel deposition (CPU)");
    m.def("anisotropic_kernel_deposition_3d", &anisotropic_kernel_deposition_3d, "3D anisotropic kernel deposition (CPU)");
}