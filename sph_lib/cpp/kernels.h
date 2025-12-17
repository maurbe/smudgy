#pragma once
#include <memory>
#include <string>
#include <cmath>
#include <stdexcept>

class SPHKernel {
protected:
    int dim_;  // ⬅️ store dimension internally
    bool anisotropic_; // ⬅️ flag for anisotropic kernels

public:
    explicit SPHKernel(int dim, bool anisotropic) : dim_(dim), anisotropic_(anisotropic) {}
    virtual float weight(float r, float h) const = 0;
    virtual float support() const = 0;
    virtual float normalization(float h) const = 0;
    virtual ~SPHKernel() {}

    int dim() const { return dim_; }
};

std::shared_ptr<SPHKernel> create_kernel(const std::string& name, int dim, bool anisotropic);

