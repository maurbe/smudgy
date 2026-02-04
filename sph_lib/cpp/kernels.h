#pragma once
#include <memory>
#include <string>
#include <cmath>
#include <stdexcept>
#include <vector>

class SPHKernel {
protected:
    int dim_;  // ⬅️ store dimension internally

public:
    explicit SPHKernel(int dim) : dim_(dim) {}
    virtual float evaluate(float q) const = 0;
    virtual float support() const = 0;
    virtual float normalization(float detH) const = 0;
    virtual ~SPHKernel() {}

    int dim() const { return dim_; }
};

struct KernelSampleGrid {
    int dim;
    int count;
    std::vector<float> coords;
    std::vector<float> q;
    std::vector<float> values;
};

KernelSampleGrid build_kernel_sample_grid(const SPHKernel& kernel, int min_kernel_evaluations);

std::shared_ptr<SPHKernel> create_kernel(const std::string& name, int dim);

