#include "kernels.h"
#include <algorithm>


class Gaussian : public SPHKernel {
public:
    explicit Gaussian(int dim): SPHKernel(dim) {}

    float evaluate(float q) const override {
        float s = support();
        if (q >= s) return 0.0f;
        return std::exp(-q * q);
    }

    float support() const override {
        return 3.0f;
    }

    float normalization(float detH) const override {
        const float pi = 3.14159265358979323846f;
        float sigma;
        if (dim_ == 1) sigma = 1.0f / std::sqrt(pi);
        else if (dim_ == 2) sigma = 1.0f / pi;
        else if (dim_ == 3) sigma = 1.0f / std::pow(pi, 1.5f);
        else throw std::invalid_argument("Unsupported dimension for Gaussian");
        return sigma / detH;
    }
};


class SuperGaussian : public SPHKernel {
public:
    explicit SuperGaussian(int dim) : SPHKernel(dim) {}

    float evaluate(float q) const override {
        float s = support();
        if (q > s) return 0.0f;

        return std::exp(-q * q) * (dim_ / 2.0f + 1.0f - q * q);
    }

    float support() const override {
        return 3.0f;
    }

    float normalization(float detH) const override {
        const float pi = 3.14159265358979323846f;
        float sigma = 1.0f / std::pow(pi, dim_ / 2.0f);
        return sigma / detH;
    }
};


class CubicSpline : public SPHKernel {
public:
    explicit CubicSpline(int dim) : SPHKernel(dim) {}

    float evaluate(float q) const override {
        float s = support();
        if (q >= s) return 0.0f;

        float result = 0.0f;
        if (q <= 0.5f) {
            result = 1 - 6 * q * q + 6 * q * q * q;
        } else {
            result = 2 * std::pow(1 - q, 3);
        }
        return result;
    }

    float support() const override {
        return 1.0f;
    }

    float normalization(float detH) const override {
        const float pi = 3.14159265358979323846f;
        if (dim_ == 1) return 4.0f / (3.0f * detH);
        if (dim_ == 2) return 40.0f / (7.0f * pi * detH);
        if (dim_ == 3) return 8.0f / (pi * detH);
        throw std::invalid_argument("Unsupported dimension for CubicSpline");
    }
};


class QuinticSpline : public SPHKernel {
public:
    explicit QuinticSpline(int dim) : SPHKernel(dim) {}

    float evaluate(float q) const override {
        float s = support();
        if (q >= s) return 0.0f;

        float result = 0.0f;
        if (q < 1.0f) {
            result = std::pow(3 - q, 5) - 6 * std::pow(2 - q, 5) + 15 * std::pow(1 - q, 5);
        } else if (q < 2.0f) {
            result = std::pow(3 - q, 5) - 6 * std::pow(2 - q, 5);
        } else {
            result = std::pow(3 - q, 5);
        }
        return result;
    }

    float support() const override {
        return 3.0f;
    }

    float normalization(float detH) const override {
        const float pi = 3.14159265358979323846f;
        if (dim_ == 1) return 1.0f / (120.0f * detH);
        if (dim_ == 2) return 7.0f / (478.0f * pi * detH);
        if (dim_ == 3) return 3.0f / (359.0f * pi * detH);
        throw std::invalid_argument("Unsupported dimension for QuinticSpline");
    }
};


class WendlandC2 : public SPHKernel {
public:
    explicit WendlandC2(int dim) : SPHKernel(dim) {}

    float evaluate(float q) const override {
        float s = support();
        if (q >= s) return 0.0f;

        float z = 1.0f - 0.5f * q;
        if (dim_ == 1)
            return std::pow(z, 3) * (1.5f * q + 1.0f);
        else
            return std::pow(z, 4) * (2.0f * q + 1.0f);
    }

    float support() const override {
        return 2.0f;
    }

    float normalization(float detH) const override {
        const float pi = 3.14159265358979323846f;
        if (dim_ == 1) return 5.0f / (8.0f * detH);
        if (dim_ == 2) return 7.0f / (4.0f * pi * detH);
        if (dim_ == 3) return 21.0f / (16.0f * pi * detH);
        throw std::invalid_argument("Unsupported dimension for WendlandC2");
    }
};


class WendlandC4 : public SPHKernel {
public:
    explicit WendlandC4(int dim) : SPHKernel(dim) {}

    float evaluate(float q) const override {
        float s = support();
        if (q >= s) return 0.0f;

        float z = 1.0f - 0.5f * q;
        if (dim_ == 1)
            return std::pow(z, 5) * (2 * q * q + 2.5f * q + 1.0f);
        else
            return std::pow(z, 6) * ((35.0f / 12.0f) * q * q + 3.0f * q + 1.0f);
    }

    float support() const override {
        return 2.0f;
    }

    float normalization(float detH) const override {
        const float pi = 3.14159265358979323846f;
        if (dim_ == 1) return 3.0f / (4.0f * detH);
        if (dim_ == 2) return 9.0f / (4.0f * pi * detH);
        if (dim_ == 3) return 495.0f / (256.0f * pi * detH);
        throw std::invalid_argument("Unsupported dimension for WendlandC4");
    }
};


class WendlandC6 : public SPHKernel {
public:
    explicit WendlandC6(int dim) : SPHKernel(dim) {}

    float evaluate(float q) const override {
        float s = support();
        if (q >= s) return 0.0f;

        float z = 1.0f - 0.5f * q;
        if (dim_ == 1)
            return std::pow(z, 7) * (21.0f / 8.0f * q * q * q + 19.0f / 4.0f * q * q + 3.5f * q + 1.0f);
        else
            return std::pow(z, 8) * (4.0f * q * q * q + 6.25f * q * q + 4.0f * q + 1.0f);
    }

    float support() const override {
        return 2.0f;
    }

    float normalization(float detH) const override {
        const float pi = 3.14159265358979323846f;
        if (dim_ == 1) return 55.0f / (64.0f * detH);
        if (dim_ == 2) return 78.0f / (28.0f * pi * detH);
        if (dim_ == 3) return 1365.0f / (512.0f * pi * detH);
        throw std::invalid_argument("Unsupported dimension for WendlandC6");
    }
};

namespace {
inline void factor_counts_2d(int total, int& n_r, int& n_theta) {
    n_r = std::max(1, static_cast<int>(std::floor(std::sqrt(static_cast<float>(total)))));
    while (n_r > 1 && (total % n_r) != 0) {
        --n_r;
    }
    n_theta = total / n_r;
}

inline void factor_counts_3d(int total, int& n_r, int& n_theta, int& n_phi) {
    n_r = std::max(1, static_cast<int>(std::floor(std::cbrt(static_cast<float>(total)))));
    for (int r = n_r; r >= 1; --r) {
        if (total % r != 0) continue;
        int remaining = total / r;
        int t = std::max(1, static_cast<int>(std::floor(std::sqrt(static_cast<float>(remaining)))));
        for (int th = t; th >= 1; --th) {
            if (remaining % th != 0) continue;
            n_r = r;
            n_theta = th;
            n_phi = remaining / th;
            return;
        }
    }
    n_r = 1;
    n_theta = 1;
    n_phi = total;
}
} // namespace

KernelSampleGrid build_kernel_sample_grid(const SPHKernel& kernel, int min_kernel_evaluations) {
    if (min_kernel_evaluations <= 0) {
        throw std::invalid_argument("min_kernel_evaluations must be > 0");
    }

    KernelSampleGrid grid;
    grid.dim = kernel.dim();
    grid.count = min_kernel_evaluations;
    grid.coords.reserve(static_cast<size_t>(min_kernel_evaluations) * grid.dim);
    grid.q.reserve(min_kernel_evaluations);
    grid.values.reserve(min_kernel_evaluations);

    constexpr float pi = 3.14159265358979323846f;
    const float support = kernel.support();

    if (grid.dim == 2) {
        int n_r = 1, n_theta = 1;
        factor_counts_2d(min_kernel_evaluations, n_r, n_theta);

        const float dr = support / static_cast<float>(n_r);
        const float dtheta = 2.0f * pi / static_cast<float>(n_theta);

        for (int ir = 0; ir < n_r; ++ir) {
            float r = (ir + 0.5f) * dr;
            for (int it = 0; it < n_theta; ++it) {
                float theta = (it + 0.5f) * dtheta;
                float x = r * std::cos(theta);
                float y = r * std::sin(theta);
                float q = r;
                float value = kernel.evaluate(q);

                grid.coords.push_back(x);
                grid.coords.push_back(y);
                grid.q.push_back(q);
                grid.values.push_back(value);
            }
        }
        return grid;
    }

    if (grid.dim == 3) {
        int n_r = 1, n_theta = 1, n_phi = 1;
        factor_counts_3d(min_kernel_evaluations, n_r, n_theta, n_phi);

        const float dr = support / static_cast<float>(n_r);
        const float dtheta = 2.0f * pi / static_cast<float>(n_theta);
        const float dphi = pi / static_cast<float>(n_phi);

        for (int ir = 0; ir < n_r; ++ir) {
            float r = (ir + 0.5f) * dr;
            for (int it = 0; it < n_theta; ++it) {
                float theta = (it + 0.5f) * dtheta;
                for (int ip = 0; ip < n_phi; ++ip) {
                    float phi = (ip + 0.5f) * dphi;

                    float sin_phi = std::sin(phi);
                    float x = r * sin_phi * std::cos(theta);
                    float y = r * sin_phi * std::sin(theta);
                    float z = r * std::cos(phi);
                    float q = r;
                    float value = kernel.evaluate(q);

                    grid.coords.push_back(x);
                    grid.coords.push_back(y);
                    grid.coords.push_back(z);
                    grid.q.push_back(q);
                    grid.values.push_back(value);
                }
            }
        }
        return grid;
    }

    throw std::invalid_argument("KernelSampleGrid supports only dim=2 or dim=3");
}




std::shared_ptr<SPHKernel> create_kernel(const std::string& name, int dim) {

    if (name == "gaussian") {
        return std::make_shared<Gaussian>(dim);
    } 
    else if (name == "super_gaussian") {
        return std::make_shared<SuperGaussian>(dim);
    } 
    else if (name == "cubic") {
        return std::make_shared<CubicSpline>(dim);
    }
    else if (name == "quintic") {
        return std::make_shared<QuinticSpline>(dim);
    } 
    else if (name == "wendland_c2") {
        return std::make_shared<WendlandC2>(dim);
    } 
    else if (name == "wendland_c4") {
        return std::make_shared<WendlandC4>(dim);
    } 
    else if (name == "wendland_c6") {
        return std::make_shared<WendlandC6>(dim);
    } 
    throw std::invalid_argument("Unknown kernel: " + name);
}
