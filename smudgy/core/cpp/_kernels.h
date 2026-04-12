#pragma once
#include <iostream>
#include <memory>
#include <string>
#include <cmath>
#include <stdexcept>
#include <vector>

// =============================================================================
// Kernel classes and constructor functions
// =============================================================================
class SeparableKernel {

    public:
        explicit SeparableKernel(int dim) : dim_(dim) {
            if (dim_ != 1 && dim_ != 2 && dim_ != 3) {
                throw std::invalid_argument("Unsupported dimension for Gaussian");
            }
        }
        virtual ~SeparableKernel() = default;

        int dim() const { return dim_; }

        // compact support in q-space
        virtual float support() const = 0;

        // normalization constant σ (no h or detH!)
        virtual float sigma() const = 0;

        // 1D kernel evaluation
        virtual float evaluate_1d(float q) const = 0;
        
        // for separable kernels, the full ND kernel is the product of 1D kernels along each axis
        virtual float evaluate(const std::vector<float>& q) const 
        {
            float val = 1.0f;
            for (int d = 0; d < dim_; ++d) 
            {
                val *= evaluate_1d(q[d]);
            }
            return val;
        }

        // indefinite integral
        virtual float F_1d(float q) const = 0; // ∫ K(q) dq

        // 1D kernel integral between two bounds
        float integrate_1d(float q0, float q1) const 
        {
            if (q1 <= q0) return 0.0f;

            // clamp to support
            q0 = std::max(q0, -support());
            q1 = std::min(q1,  support());

            // fully on one side
            if (q0 >= 0.0f) {
                return F_1d(q1) - F_1d(q0);
            }
            if (q1 <= 0.0f) {
                return F_1d(-q0) - F_1d(-q1); // flip
            }
            // crosses zero → split
            return F_1d(-q0) + F_1d(q1);
        }

        // for separable kernels, the integral over a box [x0,x1,y0,y1,...] is the product of 1D integrals
        float evaluate_integral(const std::vector<float>& bounds) const {
            float val = 1.0f;
            for (int d = 0; d < dim_; ++d) 
            {
                float q0 = bounds[2*d];
                float q1 = bounds[2*d + 1];
                val *= integrate_1d(q0, q1);
            }
            return val;
        }

    protected:
        int dim_;
};

class SphericalKernel {
    
    public:
        explicit SphericalKernel(int dim) : dim_(dim) {
            if (dim_ != 1 && dim_ != 2 && dim_ != 3) {
                throw std::invalid_argument("Unsupported dimension for Gaussian");
            }
        }
        virtual ~SphericalKernel() = default;

        int dim() const { return dim_; }
        
        // compact support in q-space
        virtual float support() const = 0;

        // normalization constant σ (no h or detH!)
        virtual float sigma() const = 0;

        // dimensionless kernel shape K(q)
        virtual float evaluate(float q) const = 0;

        // radial integral F(q) = ∫ K(q) * d\Omega
        virtual float F(float q) const = 0;

        // function to evaluate the integral of the kernel between two radii
        virtual float evaluate_integral(float q1, float q2) const 
        {
            if (q2 <= q1) return 0.0f;
             if (q2 >= support()) q2 = support();
             if (q1 <= 0.0f) q1 = 0.0f;
             return F(q2) - F(q1);
        }

    protected:
        int dim_;
};

std::shared_ptr<SeparableKernel> create_separable_kernel(const std::string& name, int dim);
std::shared_ptr<SphericalKernel> create_spherical_kernel(const std::string& name, int dim);


// =============================================================================
// Sample grid structs and constructor functions
// =============================================================================
struct SphericalKernelSampleGrid {
    int dim;
    int count;
    std::vector<float> coords;
    std::vector<float> q;
    std::vector<float> integrals;
};

SphericalKernelSampleGrid build_kernel_sample_grid(const SphericalKernel& kernel, int min_kernel_evaluations_per_axis);


// =============================================================================
// Utility functions (mainly for debugging and testing)
// =============================================================================
float compute_total_integral_separable(const std::string& kernel_name, int dim);
float compute_total_integral_spherical(const std::string& kernel_name, int dim, int min_kernel_evaluations_per_axis);

std::tuple<std::vector<float>, std::vector<float>> get_spherical_kernel_values_1D(const std::string& kernel_name);
std::tuple<std::vector<float>, std::vector<float>> get_separable_kernel_values_1D(const std::string& kernel_name);