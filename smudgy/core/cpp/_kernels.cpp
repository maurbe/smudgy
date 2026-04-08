#include "_kernels.h"
#include "_integration.h"

#include <iostream>
#include <algorithm>
#include <cmath>

namespace {
constexpr float kPi = 3.14159265358979323846f;
}


class Lucy : public SPHKernel {
public:
    explicit Lucy(int dim) : SPHKernel(dim) {}

    float evaluate(float q) const override {
        if (q > support()) return 0.0f;
        return (1.0f + 3.0f * q) * std::pow(1.0f - q, 3);
    }

    float support() const override {
        return 1.0f;
    }

    float sigma() const override {
        if (dim_ == 1) return 5.0f / (4.0f);
        if (dim_ == 2) return 5.0f / (kPi);
        if (dim_ == 3) return 105.0f / (16.0f * kPi);
        throw std::invalid_argument("Unsupported dimension for Lucy");
    }

    float F(float q) const override {
        if (q < 0.0f) return 0.0f;
        if (q > support()) q = support();

        if (dim_ == 1) {
            // ∫ K(q) dq
            return q
                - 2.0f * std::pow(q, 3)
                + 2.0f * std::pow(q, 4)
                - 0.6f * std::pow(q, 5);
        }

        if (dim_ == 2) {
            // ∫ K(q) * q dq
            return 0.5f * std::pow(q, 2)
                - 1.5f * std::pow(q, 4)
                + 1.6f * std::pow(q, 5)
                - 0.5f * std::pow(q, 6);
        }

        if (dim_ == 3) {
            // ∫ K(q) * q^2 dq
            return (1.0f/3.0f) * std::pow(q, 3)
                - (6.0f/5.0f) * std::pow(q, 5)
                + (4.0f/3.0f) * std::pow(q, 6)
                - (3.0f/7.0f) * std::pow(q, 7);
        }

        throw std::invalid_argument("Unsupported dimension");
        }
};

class Gaussian : public SPHKernel {
public:
    explicit Gaussian(int dim): SPHKernel(dim) {}

    float evaluate(float q) const override {
        if (q >= support()) return 0.0f;
        return std::exp(-q * q);
    }

    float support() const override {
        return 3.0f;
    }

    float sigma() const override {
        if (dim_ == 1) return 1.0f / std::sqrt(kPi);
        if (dim_ == 2) return 1.0f / kPi;
        if (dim_ == 3) return 1.0f / std::pow(kPi, 1.5f);
        throw std::invalid_argument("Unsupported dimension for Gaussian");
    }

    float F(float q) const override {
        if (q < 0.0f) return 0.0f;
        if (q > support()) q = support();

        if (dim_ == 1) {
            return 0.5f * std::sqrt(kPi) * std::erf(q);
        }

        if (dim_ == 2) {
            return -0.5f * std::exp(- q * q);
        }

        if (dim_ == 3) {
            return 0.25f * (std::sqrt(kPi) * std::erf(q) - 2.0f * q * std::exp(-q * q));
        }

        throw std::invalid_argument("Unsupported dimension");
    }
};

class CubicSpline : public SPHKernel {
public:
    explicit CubicSpline(int dim) : SPHKernel(dim) {}

    const float BREAKPOINT_1 = 1.0f;
    const float EPS = 1e-6f;

    float evaluate(float q) const override {
        if (q >= support()) return 0.0f;

        float r = 2.0f - q;
        float r3 = r * r * r;
        float h = 1.0f - q;
        float h3 = h * h * h;
        if (q <= BREAKPOINT_1) {
            return r3 - 4.0f * h3; // std::pow(2.0f - q, 3) - 4.0f * std::pow(1.0f - q, 3);
        } else {
            return r3;  //std::pow(2.0f - q, 3);
        }
    }

    float support() const override {
        return 2.0f;
    }

    float sigma() const override {
        if (dim_ == 1) return 1.0f / (6.0f);
        if (dim_ == 2) return 15.0f / (14.0f * 3.0f * kPi); // ??? differs from monaghan definition
        if (dim_ == 3) return 1.0f / (4.0f * kPi);
        throw std::invalid_argument("Unsupported dimension for CubicSpline");
    }

    float F(float q) const override {
        if (q < 0.0f) return 0.0f;
        if (q > support()) q = support();

        if (dim_ == 1) {
            // ∫ K(q) dq
            if (q <= BREAKPOINT_1) {
                return q * (4.0f - 2.0f * std::pow(q, 2) + 0.75f * std::pow(q, 3));
            } else {
                return -0.25f * std::pow(2.0f - q, 4);
            }
        }

        if (dim_ == 2) {
            // ∫ K(q) * q dq
            if (q <= BREAKPOINT_1) {
                return std::pow(q, 2) * 
                (
                    2.0f 
                    - 1.5f * std::pow(q, 2) 
                    + 0.6f * std::pow(q, 3)
                );
            } else {
                return std::pow(q, 2) * 
                (
                    4.0f 
                    - 4.0f * q 
                    + 1.5f * std::pow(q, 2) 
                    - 0.2f * std::pow(q, 3)
                );
            }
        }

        if (dim_ == 3) {
            // ∫ K(q) * q^2 dq
            // Since the kernel is piecewise, we need to integrate separately 
            // taking into account the values at the breakpoints
            // e.g. F(0->2) = F(0->1) + F(1->2) ≠ F(0->2) evaluated using the q>1 expression, since the kernel shape changes at q=1

            float q2 = q * q;
            float q3 = q2 * q;
            if (q <= BREAKPOINT_1) {
                return q3 * 
                    (
                        4.0f / 3.0f 
                        - 1.2f * q2 
                        + 0.5f * q3
                    );
            } else {
                return q3 * 
                    (
                        8.0f / 3.0f 
                        - 3.0f * q 
                        + 1.2f * q2 
                        - q3 / 6.0f 
                    );
            }
        }

        throw std::invalid_argument("Unsupported dimension");
    }

    float evaluate_integral(float q1, float q2) const override {
        if (q2 >= support()) q2 = support();
        if (q1 <= 0.0f) q1 = 0.0f;

        if (q1 <= BREAKPOINT_1 && BREAKPOINT_1 < q2) {
            return F(BREAKPOINT_1) - F(q1) + F(q2) - F(BREAKPOINT_1 + EPS); // add small epsilon to ensure we evaluate the second part of the integral using the correct kernel shape
        }
        else {
            return F(q2) - F(q1);
        }
    }
};

class QuinticSpline : public SPHKernel {
public:
    explicit QuinticSpline(int dim) : SPHKernel(dim) {}

    const float BREAKPOINT_1 = 1.0f;
    const float BREAKPOINT_2 = 2.0f;
    const float EPS = 1e-6f;

    float evaluate(float q) const override {
        if (q >= support()) return 0.0f;

        float result = 0.0f;
        float f = 3.0f - q;
        float s = 2.0f - q;
        float t = 1.0f - q;

        float f5 = f * f * f * f * f;
        float s5 = s * s * s * s * s;
        float t5 = t * t * t * t * t;

        if (q < BREAKPOINT_1) {
            result = f5 - 6 * s5 + 15 * t5;
        } else if (q < BREAKPOINT_2) {
            result = f5 - 6 * s5;
        } else {
            result = f5;
        }
        return result;
    }

    float support() const override {
        return 3.0f;
    }

    float sigma() const override {
        if (dim_ == 1) return 1.0f / (120.0f);
        if (dim_ == 2) return 7.0f / (478.0f * kPi);
        if (dim_ == 3) return 1.0f / (120.0f * kPi);
        throw std::invalid_argument("Unsupported dimension for QuinticSpline");
    }

    float F(float q) const override {
        if (q < 0.0f) return 0.0f;
        if (q > support()) q = support();

        // =========================
        // 1D: ∫ K(q) dq
        // =========================
        if (dim_ == 1) {
            if (q <= BREAKPOINT_1) {
                return 66.0f * q - 20.0f * std::pow(q, 3) + 6.0f * std::pow(q, 5) - 5.0f / 3.0f * std::pow(q, 6);
            } else if (q <= BREAKPOINT_2) {
                return 51.0f * q 
                        + 75.0f / 2.0f * std::pow(q, 2) 
                        - 70.0f * std::pow(q, 3) 
                        + 75.0f / 2.0f * std::pow(q, 4) 
                        - 9.0f * std::pow(q, 5) 
                        + 5.0f / 6.0f * std::pow(q, 6);
            } else {
                return -1.0f / 6.0f * std::pow(3.0f - q, 6);
            }
        }

        // =========================
        // 2D: ∫ K(q) * q dq
        // =========================
        if (dim_ == 2) {
            if (q <= BREAKPOINT_1) {
                return std::pow(q, 2) * (33.0f - 15.0f * std::pow(q, 2) + 5.0f * std::pow(q, 4) - 10.0f / 7.0f * std::pow(q, 5));
            } else if (q <= BREAKPOINT_2) {
                return std::pow(q, 2) * ( 25.5f 
                                        + 25.0f * q 
                                        - 52.5f * std::pow(q, 2)
                                        + 30.0f * std::pow(q, 3)
                                        - 7.5f * std::pow(q, 4)
                                        + 5.0f / 7.0f * std::pow(q, 5)
                                    );
            } else {
                return std::pow(q, 2) * ( 121.5f 
                                        - 135.0f * q 
                                        + 67.5f * std::pow(q, 2)
                                        - 18.0f * std::pow(q, 3)
                                        + 2.5f * std::pow(q, 4)
                                        - std::pow(q, 5) / 7.0f
                                    );
            }
        }

        // =========================
        // 3D: ∫ K(q) * q^2 dq
        // =========================
        if (dim_ == 3) {
            if (q <= BREAKPOINT_1) {
                return std::pow(q, 3) * (22.0f
                                        - 12.0f * std::pow(q, 2)
                                        + 30.0f / 7.0f * std::pow(q, 4)
                                        - 1.25f * std::pow(q, 5)
                                        );
            } else if (q <= BREAKPOINT_2) {
                return std::pow(q, 3) * ( 17.0f 
                                        + 75.0f / 4.0f * q 
                                        - 42.0f * std::pow(q, 2)
                                        + 25.0f * std::pow(q, 3)
                                        - 45.0f / 7.0f * std::pow(q, 4)
                                        + 5.0f / 8.0f * std::pow(q, 5)
                                    );
            } else {
                return std::pow(q, 3) * ( 81.0f 
                                        - 405.0f / 4.0f * q 
                                        + 54.0f * std::pow(q, 2)
                                        - 15.0f * std::pow(q, 3)
                                        + 15.0f / 7.0f * std::pow(q, 4)
                                        - std::pow(q, 5) / 8.0f
                                    );
            }
        }

        throw std::invalid_argument("Unsupported dimension");
    }

    float evaluate_integral(float q1, float q2) const override {
         if (q2 >= support()) q2 = support();
         if (q1 <= 0.0f) q1 = 0.0f;

        // case 1
        if (q1 <= BREAKPOINT_1 && BREAKPOINT_2 < q2) {
            return F(BREAKPOINT_1) - F(q1) 
                + (F(BREAKPOINT_2) - F(BREAKPOINT_1 + EPS)) 
                + (F(q2) - F(BREAKPOINT_2 + EPS));
        }
        // case 2
        else if (q1 <= BREAKPOINT_1 && BREAKPOINT_1 < q2) {
            return F(BREAKPOINT_1) - F(q1) + F(q2) - F(BREAKPOINT_1 + EPS);
        }
        else if (q1 <= BREAKPOINT_2 && BREAKPOINT_2 < q2) {
            return F(BREAKPOINT_2) - F(q1) + F(q2) - F(BREAKPOINT_2 + EPS);
        }
        else {
            return F(q2) - F(q1);
        }
    }
};

class WendlandC2 : public SPHKernel {
public:
    explicit WendlandC2(int dim) : SPHKernel(dim) {}

    float evaluate(float q) const override {
        if (q >= support()) return 0.0f;

        float z = 1.0f - 0.5f * q;
        if (dim_ == 1)
            return std::pow(z, 3) * (1.5f * q + 1.0f);
        else
            return std::pow(z, 4) * (2.0f * q + 1.0f);
    }

    float support() const override {
        return 2.0f;
    }

    float sigma() const override {
        if (dim_ == 1) return 5.0f / (8.0f);
        if (dim_ == 2) return 7.0f / (4.0f * kPi);
        if (dim_ == 3) return 21.0f / (16.0f * kPi);
        throw std::invalid_argument("Unsupported dimension for WendlandC2");
    }

    float F(float q) const override {
        if (q < 0.0f) return 0.0f;
        if (q > support()) q = support();

        if (dim_ == 1) {
            // ∫ K dq
            return q
                - 0.5f * std::pow(q, 3)
                + 0.25f * std::pow(q, 4)
                - 3.0f / 80.0f *std::pow(q, 5);
        }

        if (dim_ == 2) {
            // ∫ K * q dq
            return std::pow(q, 2) / 16.0f * (8.0f
                                            - 10.0f * std::pow(q, 2)
                                            + 8.0f * std::pow(q, 3)
                                            - 2.5f * std::pow(q, 4)
                                            + 2.0f / 7.0f * std::pow(q, 5)
                                            );
        }

        if (dim_ == 3) {
            // ∫ K * q^2 dq
            return std::pow(q, 3) / 16.0f * (16.0f / 3.0f
                                            - 8.0f * std::pow(q, 2)
                                            + 20.0f / 3.0f * std::pow(q, 3)
                                            - 15.0f / 7.0f * std::pow(q, 4)
                                            + 0.25f * std::pow(q, 5)
                                            );
        }

        throw std::invalid_argument("Unsupported dimension");
    }
};

class WendlandC4 : public SPHKernel {
public:
    explicit WendlandC4(int dim) : SPHKernel(dim) {}

    float evaluate(float q) const override {
        if (q >= support()) return 0.0f;

        float z = 1.0f - 0.5f * q;
        if (dim_ == 1)
            return std::pow(z, 5) * (2 * q * q + 2.5f * q + 1.0f);
        else
            return std::pow(z, 6) * ((35.0f / 12.0f) * q * q + 3.0f * q + 1.0f);
    }

    float support() const override {
        return 2.0f;
    }

    float sigma() const override {
        if (dim_ == 1) return 3.0f / (4.0f);
        if (dim_ == 2) return 9.0f / (4.0f * kPi);
        if (dim_ == 3) return 495.0f / (256.0f * kPi);
        throw std::invalid_argument("Unsupported dimension for WendlandC4");
    }

    float F(float q) const override {
        if (q < 0.0f) return 0.0f;
        if (q > support()) q = support();

        if (dim_ == 1) {
            return q / 64.0f * (64.0f
                               - 112.0f / 3.0f * std::pow(q, 2)
                               + 28.0f * std::pow(q, 4)
                               - 56.0f / 3.0f * std::pow(q, 5)
                               + 5.0f * std::pow(q, 6)
                               - 0.5f * std::pow(q, 7)
                            );
        }

        if (dim_ == 2) {
            return std::pow(q, 2) / 768.0f * ( 384.0f
                                             - 448.0f * std::pow(q, 2)
                                             + 560.0f * std::pow(q, 4)
                                             - 512.0f * std::pow(q, 5) 
                                             + 210.0f * std::pow(q, 6)
                                             - 128.0f / 3.0f * std::pow(q, 7)
                                             + 7.0f / 2.0f * std::pow(q, 8)
                                        );
        }

        if (dim_ == 3) {
            return std::pow(q, 3) / 768.0f * (256.0f
                                             - 1792.0f / 5.0f * std::pow(q, 2)
                                             + 480.0f * std::pow(q, 4)
                                             - 448.0f * std::pow(q, 5)
                                             + 560.0f / 3.0f * std::pow(q, 6)
                                             - 192.0f / 5.0f * std::pow(q, 7)
                                             + 35.0f / 11.0f * std::pow(q, 8)                                             
                                    );
        }

        throw std::invalid_argument("Unsupported dimension");
    }
};

class WendlandC6 : public SPHKernel {
public:
    explicit WendlandC6(int dim) : SPHKernel(dim) {}

    float evaluate(float q) const override {
        if (q >= support()) return 0.0f;

        float z = 1.0f - 0.5f * q;
        if (dim_ == 1)
            return std::pow(z, 7) * (21.0f / 8.0f * std::pow(q, 3) + 19.0f / 4.0f * std::pow(q, 2) + 3.5f * q + 1.0f);
        else
            return std::pow(z, 8) * (4.0f * std::pow(q, 3) + 6.25f * std::pow(q, 2) + 4.0f * q + 1.0f);
    }

    float support() const override {
        return 2.0f;
    }

    float sigma() const override {
        if (dim_ == 1) return 55.0f / (64.0f);
        if (dim_ == 2) return 39.0f / (14.0f * kPi);
        if (dim_ == 3) return 1365.0f / (512.0f * kPi);
        throw std::invalid_argument("Unsupported dimension for WendlandC6");
    }

    float F(float q) const override {
        if (q < 0.0f) return 0.0f;
        if (q > support()) q = support();

        if (dim_ == 1) {
            return 1.0f / 1024.0f * (
                1024.0f * q
                - 768.0f * std::pow(q, 3)
                + 2688.0f / 5.0f * std::pow(q, 5)
                - 480.0f * std::pow(q, 7)
                + 384.0f * std::pow(q, 8)
                - 140.0f * std::pow(q, 9)
                + 128.0f / 5.0f * std::pow(q, 10)
                - 21.0f / 11.0f * std::pow(q, 11)
            );
        }

        if (dim_ == 2) {
            return 0.5f * std::pow(q, 2) *
                ( 1.0f
                - 11.0f / 8.0f * std::pow(q, 2)
                + 11.0f / 8.0f * std::pow(q, 4)
                - 231.0f / 128.0f * std::pow(q, 6)
                + 11.0f / 6.0f * std::pow(q, 7)
                - 231.0f / 256.0f * std::pow(q, 8)
                + 0.25f * std::pow(q, 9)
                - 77.0f / 2048.0f * std::pow(q, 10)
                + 1.0f / 416.0f * std::pow(q, 11)
                );
        }

        if (dim_ == 3) {
            return std::pow(q, 3) *
            (
                1.0f / 3.0f
                - 11.0f / 20.0f * std::pow(q, 2)
                + 33.0f / 56.0f * std::pow(q, 4)
                - 77.0f / 96.0f * std::pow(q, 6)
                + 33.0f / 40.0f * std::pow(q, 7)
                - 105.0f / 256.0f * std::pow(q, 8)
                + 11.0f / 96.0f * std::pow(q, 9)
                - 231.0f / 13312.0f * std::pow(q, 10)
                + 1.0f / 896.0f * std::pow(q, 11)
            );
        }

        throw std::invalid_argument("Unsupported dimension");
    }
};


KernelSampleGrid build_kernel_sample_grid(const SPHKernel& kernel,
                                          int min_kernel_evaluations_per_axis
                                        ){
    if (min_kernel_evaluations_per_axis <= 0) {
        throw std::invalid_argument("min_kernel_evaluations_per_axis must be > 0");
    }

    // compute total number of kernel evaluations
    const int total_number_o = static_cast<int>(std::pow(min_kernel_evaluations_per_axis, kernel.dim()));

    KernelSampleGrid grid;
    grid.dim = kernel.dim();
    grid.count = total_number_o;
    grid.coords.reserve(static_cast<size_t>(total_number_o) * grid.dim);
    grid.q.reserve(total_number_o);
    grid.integrals.reserve(total_number_o);

    const float support = kernel.support();

    if (grid.dim == 2) {
        
        int n_q = min_kernel_evaluations_per_axis;
        int n_phi = min_kernel_evaluations_per_axis;

        const float dq = support / static_cast<float>(n_q);
        const float dphi = 2.0f * kPi / static_cast<float>(n_phi);

        for (int iq = 0; iq < n_q; ++iq) {
            const float q0 = static_cast<float>(iq) * dq;
            const float q  = q0 + dq * 0.5f;
            const float q1 = q0 + dq;

            for (int it = 0; it < n_phi; ++it) {
                const float phiC = (it + 0.5f) * dphi;
                const float x = q * std::cos(phiC);
                const float y = q * std::sin(phiC);

                // this integral is now analytically evaluated and thus exact!
                float integral = kernel.sigma() * dphi * kernel.evaluate_integral(q0, q1);

                grid.coords.push_back(x);
                grid.coords.push_back(y);
                grid.q.push_back(q);
                grid.integrals.push_back(integral);
            }
        }
        return grid;
    }

    if (grid.dim == 3) {
        int n_q = min_kernel_evaluations_per_axis;
        int n_theta = min_kernel_evaluations_per_axis;
        int n_phi = min_kernel_evaluations_per_axis;

        const float dq = support / static_cast<float>(n_q);
        const float dtheta = kPi / static_cast<float>(n_theta);
        const float dphi = 2.0f * kPi / static_cast<float>(n_phi);

        for (int iq = 0; iq < n_q; ++iq) {
            const float q0 = static_cast<float>(iq) * dq;
            const float q  = q0 + 0.5f * dq;
            const float q1 = q0 + dq;

            for (int it = 0; it < n_theta; ++it) {
                const float theta0 = static_cast<float>(it) * dtheta;
                const float thetaC = (it + 0.5f) * dtheta;
                const float theta1 = theta0 + dtheta;

                for (int ip = 0; ip < n_phi; ++ip) {
                    const float phi0 = static_cast<float>(ip) * dphi;
                    const float phiC = phi0 + 0.5f * dphi;

                    const float sin_thetaC = std::sin(thetaC);
                    const float x = q * sin_thetaC * std::cos(phiC);
                    const float y = q * sin_thetaC * std::sin(phiC);
                    const float z = q * std::cos(thetaC);

                    // this integral is now analytically evaluated and thus exact!
                    float integral = kernel.sigma() * dphi * (-std::cos(theta1) + std::cos(theta0)) * kernel.evaluate_integral(q0, q1);
                   
                    grid.coords.push_back(x);
                    grid.coords.push_back(y);
                    grid.coords.push_back(z);
                    grid.q.push_back(q);
                    grid.integrals.push_back(integral);
                }
            }
        }
        return grid;
    }

    throw std::invalid_argument("[smudgy] KernelSampleGrid supports only dim=2 or dim=3");
}


std::shared_ptr<SPHKernel> create_kernel(const std::string& name, int dim) {

    if (name == "lucy") {
        return std::make_shared<Lucy>(dim);
    } 
    else if (name == "gaussian") {
        return std::make_shared<Gaussian>(dim);
    } 
    else if (name == "cubic_spline") {
        return std::make_shared<CubicSpline>(dim);
    }
    else if (name == "quintic_spline") {
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
    throw std::invalid_argument("[smudgy] Unknown kernel: " + name);
}


// Computes the total integral of the kernel over its sample grid
float compute_kernel_total_integral(const std::string& kernel_name, int dim, int min_kernel_evaluations_per_axis) {
    auto kernel = create_kernel(kernel_name, dim);
    const auto kernel_samples = build_kernel_sample_grid(*kernel, min_kernel_evaluations_per_axis);
    float total_integral = 0.0f;
    for (int s = 0; s < kernel_samples.count; ++s) {
        total_integral += kernel_samples.integrals[s];
    }
    return total_integral;
}