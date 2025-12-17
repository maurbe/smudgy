#include "kernels.h"


class Gaussian : public SPHKernel {
public:
    explicit Gaussian(int dim, bool anisotropic): SPHKernel(dim, anisotropic) {}

    float weight(float r, float h) const override {
        float q = r / h;
        float s = support();
        if (q >= s) return 0.0f;
        return std::exp(-q * q);
    }

    float support() const override {
        return 3.0f;
    }

    float normalization(float h) const override {
        const float pi = 3.14159265358979323846f;
        float sigma;
        if (dim_ == 1) sigma = 1.0f / std::sqrt(pi);
        else if (dim_ == 2) sigma = 1.0f / pi;
        else if (dim_ == 3) sigma = 1.0f / std::pow(pi, 1.5f);
        else throw std::invalid_argument("Unsupported dimension for Gaussian");

        float detH = anisotropic_ ? h : std::pow(h, dim_); // if anisotropic, h=detH, otherwise detH=pow(h, dim)
        return sigma / detH;
    }
};


class SuperGaussian : public SPHKernel {
public:
    explicit SuperGaussian(int dim, bool anisotropic) : SPHKernel(dim, anisotropic) {}

    float weight(float r, float h) const override {
        float q = r / h;
        float s = support();
        if (q > s) return 0.0f;

        return std::exp(-q * q) * (dim_ / 2.0f + 1.0f - q * q);
    }

    float support() const override {
        return 3.0f;
    }

    float normalization(float h) const override {
        const float pi = 3.14159265358979323846f;
        float sigma = 1.0f / std::pow(pi, dim_ / 2.0f);
        float detH = anisotropic_ ? h : std::pow(h, dim_);
        return sigma / detH;
    }
};


class CubicSpline : public SPHKernel {
public:
    explicit CubicSpline(int dim, bool anisotropic) : SPHKernel(dim, anisotropic) {}

    float weight(float r, float h) const override {
        float q = r / h;
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

    float normalization(float h) const override {
        const float pi = 3.14159265358979323846f;
        float detH = anisotropic_ ? h : std::pow(h, dim_);
        if (dim_ == 1) return 4.0f / (3.0f * detH);
        if (dim_ == 2) return 40.0f / (7.0f * pi * detH);
        if (dim_ == 3) return 8.0f / (pi * detH);
        throw std::invalid_argument("Unsupported dimension for CubicSpline");
    }
};


class QuinticSpline : public SPHKernel {
public:
    explicit QuinticSpline(int dim, bool anisotropic) : SPHKernel(dim, anisotropic) {}

    float weight(float r, float h) const override {
        float q = r / h;
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

    float normalization(float h) const override {
        const float pi = 3.14159265358979323846f;
        float detH = anisotropic_ ? h : std::pow(h, dim_);
        if (dim_ == 1) return 1.0f / (120.0f * detH);
        if (dim_ == 2) return 7.0f / (478.0f * pi * detH);
        if (dim_ == 3) return 3.0f / (359.0f * pi * detH);
        throw std::invalid_argument("Unsupported dimension for QuinticSpline");
    }
};


class WendlandC2 : public SPHKernel {
public:
    explicit WendlandC2(int dim, bool anisotropic) : SPHKernel(dim, anisotropic) {}

    float weight(float r, float h) const override {
        float q = r / h;
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

    float normalization(float h) const override {
        const float pi = 3.14159265358979323846f;
        float detH = anisotropic_ ? h : std::pow(h, dim_);
        if (dim_ == 1) return 5.0f / (8.0f * detH);
        if (dim_ == 2) return 7.0f / (4.0f * pi * detH);
        if (dim_ == 3) return 21.0f / (16.0f * pi * detH);
        throw std::invalid_argument("Unsupported dimension for WendlandC2");
    }
};


class WendlandC4 : public SPHKernel {
public:
    explicit WendlandC4(int dim, bool anisotropic) : SPHKernel(dim, anisotropic) {}

    float weight(float r, float h) const override {
        float q = r / h;
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

    float normalization(float h) const override {
        const float pi = 3.14159265358979323846f;
        float detH = anisotropic_ ? h : std::pow(h, dim_);
        if (dim_ == 1) return 3.0f / (4.0f * detH);
        if (dim_ == 2) return 9.0f / (4.0f * pi * detH);
        if (dim_ == 3) return 495.0f / (256.0f * pi * detH);
        throw std::invalid_argument("Unsupported dimension for WendlandC4");
    }
};


class WendlandC6 : public SPHKernel {
public:
    explicit WendlandC6(int dim, bool anisotropic) : SPHKernel(dim, anisotropic) {}

    float weight(float r, float h) const override {
        float q = r / h;
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

    float normalization(float h) const override {
        const float pi = 3.14159265358979323846f;
        float detH = anisotropic_ ? h : std::pow(h, dim_);
        if (dim_ == 1) return 55.0f / (64.0f * detH);
        if (dim_ == 2) return 78.0f / (28.0f * pi * detH);
        if (dim_ == 3) return 1365.0f / (512.0f * pi * detH);
        throw std::invalid_argument("Unsupported dimension for WendlandC6");
    }
};




std::shared_ptr<SPHKernel> create_kernel(const std::string& name, int dim, bool anisotropic) {

    if (name == "gaussian") {
        return std::make_shared<Gaussian>(dim, anisotropic);
    } 
    else if (name == "super_gaussian") {
        return std::make_shared<SuperGaussian>(dim, anisotropic);
    } 
    else if (name == "cubic") {
        return std::make_shared<CubicSpline>(dim, anisotropic);
    }
    else if (name == "quintic") {
        return std::make_shared<QuinticSpline>(dim, anisotropic);
    } 
    else if (name == "wendland_c2") {
        return std::make_shared<WendlandC2>(dim, anisotropic);
    } 
    else if (name == "wendland_c4") {
        return std::make_shared<WendlandC4>(dim, anisotropic);
    } 
    else if (name == "wendland_c6") {
        return std::make_shared<WendlandC6>(dim, anisotropic);
    } 
    throw std::invalid_argument("Unknown kernel: " + name);
}
