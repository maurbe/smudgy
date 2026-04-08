#pragma once
#include <memory>
#include <string>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <iostream>

class SPHKernel {
    
    public:
        explicit SPHKernel(int dim) : dim_(dim) {}
        virtual ~SPHKernel() = default;

        int dim() const { return dim_; }

        // dimensionless kernel shape K(q)
        virtual float evaluate(float q) const = 0;

        // compact support in q-space
        virtual float support() const = 0;

        // normalization constant σ (no h or detH!)
        virtual float sigma() const = 0;

        // radial integral F(q) = ∫ K(q) * d\Omega
        virtual float F(float q) const = 0;

        // function to evaluate the integral of the kernel between two points
        virtual float evaluate_integral(float q1, float q2) const {
            if (q2 <= q1) return 0.0f;
             if (q2 >= support()) q2 = support();
             if (q1 <= 0.0f) q1 = 0.0f;
             return F(q2) - F(q1);
        }

    protected:
        int dim_;
};

struct KernelSampleGrid {
    int dim;
    int count;
    std::vector<float> coords;
    std::vector<float> q;
    std::vector<float> integrals;
};

KernelSampleGrid build_kernel_sample_grid(const SPHKernel& kernel,
                                           int min_kernel_evaluations_per_axis
                                        );


std::shared_ptr<SPHKernel> create_kernel(const std::string& name, int dim);

// Computes the total integral of the kernel over its sample grid
float compute_kernel_total_integral(const std::string& kernel_name, int dim, int min_kernel_evaluations_per_axis);

