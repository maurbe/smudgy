#pragma once
#include <iostream>
#include <memory>
#include <string>
#include <cmath>
#include <stdexcept>
#include <vector>

class SeparableKernel {

    public:
        explicit SeparableKernel(int dim) : dim_(dim) {}
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
        virtual float F(float q) const = 0; // ∫ K(q) dq

        // 1D kernel integral between two bounds
        float integrate_1d(float q0, float q1) const 
        {
            if (q1 <= q0) return 0.0f;

            // clamp to support
            q0 = std::max(q0, -support());
            q1 = std::min(q1,  support());

            // fully on one side
            if (q0 >= 0.0f) {
                return F(q1) - F(q0);
            }
            if (q1 <= 0.0f) {
                return F(-q0) - F(-q1); // flip
            }
            // crosses zero → split
            return F(-q0) + F(q1);
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
        explicit SphericalKernel(int dim) : dim_(dim) {}
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

struct KernelSampleGrid {
    int dim;
    int count;
    std::vector<float> coords;
    std::vector<float> q;
    std::vector<float> integrals;
};

KernelSampleGrid build_kernel_sample_grid(const SphericalKernel& kernel, int min_kernel_evaluations_per_axis);

std::shared_ptr<SeparableKernel> create_separable_kernel(const std::string& name, int dim);
std::shared_ptr<SphericalKernel> create_spherical_kernel(const std::string& name, int dim);

// Computes the total integral of the kernel over its sample grid
float compute_kernel_total_integral(const std::string& kernel_name, int dim, int min_kernel_evaluations_per_axis);
std::tuple<std::vector<float>, std::vector<float>> get_kernel_values_1D(const std::string& kernel_name);
