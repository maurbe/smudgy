#include <iostream>
#include <algorithm>
#include <cmath>

#include "_kernels.h"
#include "_integration.h"

namespace {
constexpr float kPi = 3.14159265358979323846f;
}

class TophatSep : public SeparableKernel {
    // Rectangular Tophat (NGP equivalent)
    public:
    explicit TophatSep(int dim) : SeparableKernel(dim) {}
    
    const float SUPPORT = 0.5f;
    
    float evaluate_1d(float q) const override {
        q = std::abs(q);
        if (q > SUPPORT) return 0.0f;
        return 1.0f;
    }
    
    float support() const override { return SUPPORT; }
    
    float sigma() const override {
        return 1.0f;
    }
    
    // this is the 1D integral of the kernel, as evaluate_integral for separable kernels is just the product of 1D integrals along each axis
    float F_1d(float q) const override {
        if (q < 0.0f) return 0.0f;
        if (q > SUPPORT) q = SUPPORT;
        return q;
    }
};

class TSCSep : public SeparableKernel {
    // Rectangular TSC
    public:
    explicit TSCSep(int dim) : SeparableKernel(dim) {}

    const float SUPPORT = 1.5f;
    
    float evaluate_1d(float q) const override {
        // 1D TSC kernel on support [-1.5, 1.5]
        q = std::abs(q);

        if (q >= SUPPORT) return 0.0f;
        if (q <= 0.5f) {
            return 0.75f - q * q;
        } else {
            float val = SUPPORT - q;
            return 0.5f * val * val;
        }
    }
    
    float support() const override { return SUPPORT; }
    
    float sigma() const override {
        return 1.0f;
    }
    
    float F_1d(float q) const override {
        if (q < 0.0f) return 0.0f;
        if (q > SUPPORT) q = SUPPORT;
        
        float f1d;
        if (q <= 0.5f) {
            f1d = 0.75f * q - (1.0f / 3.0f) * q * q * q;
        } else {
            f1d = (1.0f / 3.0f) - (1.0f / 6.0f) * (std::pow(SUPPORT - q, 3) - 1.0f);
        }
        
        // For rectangular kernels, F(q) returns the integral from 0 to q along each dimension
        return f1d;
    }
};

class GaussianSep : public SeparableKernel {
    // Rectangular Gaussian
    public:
    explicit GaussianSep(int dim) : SeparableKernel(dim) {}

     const float SUPPORT = 3.0f;

    float evaluate_1d(float q) const override {
        q = std::abs(q);
        if (q >= SUPPORT) return 0.0f;
        return std::exp(-q * q);
    }
    
    float support() const override { return SUPPORT; }
    
    float sigma() const override {
        if (dim_ == 1) return 1.0f / std::sqrt(kPi);
        if (dim_ == 2) return 1.0f / kPi;
        if (dim_ == 3) return 1.0f / std::pow(kPi, 1.5f);
    }
    
    float F_1d(float q) const override {
        if (q < 0.0f) return 0.0f;
        if (q > SUPPORT) q = SUPPORT;

        // For rectangular kernels, F(q) returns the integral from 0 to q along each dimension
        float f1d = 0.5f * std::sqrt(kPi) * std::erf(q);
        return f1d;
    }
};

class Tophat : public SphericalKernel {
// spherical tophat
public:
    explicit Tophat(int dim) : SphericalKernel(dim) {}

    const float SUPPORT = 0.5f;

    float evaluate(float q) const override {
        if (q > SUPPORT) return 0.0f;
        return 1.0f;
    }

    float support() const override { return SUPPORT; }

    float sigma() const override {
        if (dim_ == 1) return 1.0f;
        if (dim_ == 2) return 4.0f / kPi; // TODO: DOUBLE CHECK DIM=2, 3
        if (dim_ == 3) return 6.0f / kPi; // same here
    }

    float F(float q) const override {
        if (q < 0.0f) return 0.0f;
        if (q > SUPPORT) q = SUPPORT;

        if (dim_ == 1) {
            return q;
        }

        if (dim_ == 2) {
            return 0.5f * q * q;
        }

        if (dim_ == 3) {
            return (1.0f / 3.0f) * q * q * q;
        }
    }
};

class TSC : public SphericalKernel {
// spherical triangular-shaped cloud (TSC)
public:
    explicit TSC(int dim) : SphericalKernel(dim) {}

    const float SUPPORT = 1.5f;
    const float NODE_1 = 0.5f;
    const float EPS = 1e-6f;

    float evaluate(float q) const override {
        if (q >= SUPPORT) return 0.0f;
        if (q <= NODE_1) {
            return 0.75f - q * q;
        } else {
            float h = SUPPORT - q;
            return 0.5f * h * h;
        }
    }

    float support() const override { return SUPPORT; }

    float sigma() const override {
        if (dim_ == 1) return 1.0f;
        if (dim_ == 2) return 1.0f / 1.27627f;
        if (dim_ == 3) return 1.0f / 1.5708f;
    }

    float F(float q) const override {
        if (q < 0.0f) return 0.0f;
        if (q > SUPPORT) q = SUPPORT;

        if (dim_ == 1) {
            if (q <= NODE_1) {
                return 0.75f * q - (1.0f / 3.0f) * q * q * q;
            } else {
                return std::pow(q, 3) / 6.0f - 3.0f / 4.0f * q * q + 9.0f/8.0f * q;
            }
        }

        if (dim_ == 2) {
            if (q <= NODE_1) {
                return 0.375f * q * q - 0.25f * std::pow(q, 4);
            } else {
                return q * q / 2.0f * (0.25f * q * q - q + 1.125f);
            }
        }

        if (dim_ == 3) {
            if (q <= NODE_1) {
                return std::pow(q, 3) * (0.25f - 0.2f * q * q);
            } else {
                return std::pow(q, 3) * (0.375f - 0.375f * q + 0.1 * q * q);
            }
        }
    }

    float evaluate_integral(float q1, float q2) const override {
        if (q2 >= SUPPORT) q2 = SUPPORT;
        if (q1 <= 0.0f) q1 = 0.0f;

        if (q1 <= NODE_1 && NODE_1 < q2) {
            return F(NODE_1) - F(q1) + F(q2) - F(NODE_1 + EPS); // add small epsilon to ensure we evaluate the second part of the integral using the correct kernel shape
        }
        else {
            return F(q2) - F(q1);
        }
    }
};

class Lucy : public SphericalKernel {
public:
    explicit Lucy(int dim) : SphericalKernel(dim) {}

    const float SUPPORT = 1.0f;

    float evaluate(float q) const override {
        if (q > SUPPORT) return 0.0f;
        return (1.0f + 3.0f * q) * std::pow(1.0f - q, 3);
    }

    float support() const override { return SUPPORT; }

    float sigma() const override {
        if (dim_ == 1) return 5.0f / (4.0f);
        if (dim_ == 2) return 5.0f / (kPi);
        if (dim_ == 3) return 105.0f / (16.0f * kPi);
    }

    float F(float q) const override {
        if (q < 0.0f) return 0.0f;
        if (q > SUPPORT) q = SUPPORT;

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
    }
};

class Gaussian : public SphericalKernel {
public:
    explicit Gaussian(int dim): SphericalKernel(dim) {}

    const float SUPPORT = 3.0f;

    float evaluate(float q) const override {
        if (q >= SUPPORT) return 0.0f;
        return std::exp(-q * q);
    }

    float support() const override { return SUPPORT; }

    float sigma() const override {
        if (dim_ == 1) return 1.0f / std::sqrt(kPi);
        if (dim_ == 2) return 1.0f / kPi;
        if (dim_ == 3) return 1.0f / std::pow(kPi, 1.5f);
    }

    float F(float q) const override {
        if (q < 0.0f) return 0.0f;
        if (q > SUPPORT) q = SUPPORT;

        if (dim_ == 1) {
            return 0.5f * std::sqrt(kPi) * std::erf(q);
        }

        if (dim_ == 2) {
            return -0.5f * std::exp(- q * q);
        }

        if (dim_ == 3) {
            return 0.25f * (std::sqrt(kPi) * std::erf(q) - 2.0f * q * std::exp(-q * q));
        }
    }
};

class CubicSpline : public SphericalKernel {
public:
    explicit CubicSpline(int dim) : SphericalKernel(dim) {}

    const float SUPPORT = 2.0f;
    const float NODE_1 = 1.0f;
    const float EPS = 1e-6f;

    float evaluate(float q) const override {
        if (q >= SUPPORT) return 0.0f;

        float r = 2.0f - q;
        float r3 = r * r * r;
        float h = 1.0f - q;
        float h3 = h * h * h;
        if (q <= NODE_1) {
            return r3 - 4.0f * h3; // std::pow(2.0f - q, 3) - 4.0f * std::pow(1.0f - q, 3);
        } else {
            return r3;  //std::pow(2.0f - q, 3);
        }
    }

    float support() const override { return SUPPORT; }

    float sigma() const override {
        if (dim_ == 1) return 1.0f / (6.0f);
        if (dim_ == 2) return 15.0f / (14.0f * 3.0f * kPi); // ??? differs from monaghan definition
        if (dim_ == 3) return 1.0f / (4.0f * kPi);
    }

    float F(float q) const override {
        if (q < 0.0f) return 0.0f;
        if (q > SUPPORT) q = SUPPORT;

        if (dim_ == 1) {
            // ∫ K(q) dq
            if (q <= NODE_1) {
                return q * (4.0f - 2.0f * std::pow(q, 2) + 0.75f * std::pow(q, 3));
            } else {
                return -0.25f * std::pow(2.0f - q, 4);
            }
        }

        if (dim_ == 2) {
            // ∫ K(q) * q dq
            if (q <= NODE_1) {
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
            if (q <= NODE_1) {
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
    }

    float evaluate_integral(float q1, float q2) const override {
        if (q2 >= SUPPORT) q2 = SUPPORT;
        if (q1 <= 0.0f) q1 = 0.0f;

        if (q1 <= NODE_1 && NODE_1 < q2) {
            return F(NODE_1) - F(q1) + F(q2) - F(NODE_1 + EPS); // add small epsilon to ensure we evaluate the second part of the integral using the correct kernel shape
        }
        else {
            return F(q2) - F(q1);
        }
    }
};

class QuinticSpline : public SphericalKernel {
public:
    explicit QuinticSpline(int dim) : SphericalKernel(dim) {}

    const float SUPPORT = 3.0f;
    const float NODE_1 = 1.0f;
    const float NODE_2 = 2.0f;
    const float EPS = 1e-6f;

    float evaluate(float q) const override {
        if (q >= SUPPORT) return 0.0f;

        float result = 0.0f;
        float f = 3.0f - q;
        float s = 2.0f - q;
        float t = 1.0f - q;

        float f5 = f * f * f * f * f;
        float s5 = s * s * s * s * s;
        float t5 = t * t * t * t * t;

        if (q < NODE_1) {
            result = f5 - 6 * s5 + 15 * t5;
        } else if (q < NODE_2) {
            result = f5 - 6 * s5;
        } else {
            result = f5;
        }
        return result;
    }

    float support() const override { return SUPPORT; }

    float sigma() const override {
        if (dim_ == 1) return 1.0f / (120.0f);
        if (dim_ == 2) return 7.0f / (478.0f * kPi);
        if (dim_ == 3) return 1.0f / (120.0f * kPi);
    }

    float F(float q) const override {
        if (q < 0.0f) return 0.0f;
        if (q > SUPPORT) q = SUPPORT;

        // =========================
        // 1D: ∫ K(q) dq
        // =========================
        if (dim_ == 1) {
            if (q <= NODE_1) {
                return 66.0f * q - 20.0f * std::pow(q, 3) + 6.0f * std::pow(q, 5) - 5.0f / 3.0f * std::pow(q, 6);
            } else if (q <= NODE_2) {
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
            if (q <= NODE_1) {
                return std::pow(q, 2) * (33.0f - 15.0f * std::pow(q, 2) + 5.0f * std::pow(q, 4) - 10.0f / 7.0f * std::pow(q, 5));
            } else if (q <= NODE_2) {
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
            if (q <= NODE_1) {
                return std::pow(q, 3) * (22.0f
                                        - 12.0f * std::pow(q, 2)
                                        + 30.0f / 7.0f * std::pow(q, 4)
                                        - 1.25f * std::pow(q, 5)
                                        );
            } else if (q <= NODE_2) {
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
    }

    float evaluate_integral(float q1, float q2) const override {
         if (q2 >= SUPPORT) q2 = SUPPORT;
         if (q1 <= 0.0f) q1 = 0.0f;

        // case 1
        if (q1 <= NODE_1 && NODE_2 < q2) {
            return F(NODE_1) - F(q1) 
                + (F(NODE_2) - F(NODE_1 + EPS)) 
                + (F(q2) - F(NODE_2 + EPS));
        }
        // case 2
        else if (q1 <= NODE_1 && NODE_1 < q2) {
            return F(NODE_1) - F(q1) + F(q2) - F(NODE_1 + EPS);
        }
        else if (q1 <= NODE_2 && NODE_2 < q2) {
            return F(NODE_2) - F(q1) + F(q2) - F(NODE_2 + EPS);
        }
        else {
            return F(q2) - F(q1);
        }
    }
};

class WendlandC2 : public SphericalKernel {
public:
    explicit WendlandC2(int dim) : SphericalKernel(dim) {}

    const float SUPPORT = 2.0f;

    float evaluate(float q) const override {
        if (q >= SUPPORT) return 0.0f;

        float z = 1.0f - 0.5f * q;
        if (dim_ == 1)
            return std::pow(z, 3) * (1.5f * q + 1.0f);
        else
            return std::pow(z, 4) * (2.0f * q + 1.0f);
    }

    float support() const override { return SUPPORT; }

    float sigma() const override {
        if (dim_ == 1) return 5.0f / (8.0f);
        if (dim_ == 2) return 7.0f / (4.0f * kPi);
        if (dim_ == 3) return 21.0f / (16.0f * kPi);
    }

    float F(float q) const override {
        if (q < 0.0f) return 0.0f;
        if (q > SUPPORT) q = SUPPORT;

        if (dim_ == 1) {
            // F(q) = ∫ K dq
            return q
                - 0.5f * std::pow(q, 3)
                + 0.25f * std::pow(q, 4)
                - 3.0f / 80.0f *std::pow(q, 5);
        }

        if (dim_ == 2) {
            // F(q) = ∫ K * q dq
            return std::pow(q, 2) / 16.0f * (8.0f
                                            - 10.0f * std::pow(q, 2)
                                            + 8.0f * std::pow(q, 3)
                                            - 2.5f * std::pow(q, 4)
                                            + 2.0f / 7.0f * std::pow(q, 5)
                                            );
        }

        if (dim_ == 3) {
            // F(q) = ∫ K * q^2 dq
            return std::pow(q, 3) / 16.0f * (16.0f / 3.0f
                                            - 8.0f * std::pow(q, 2)
                                            + 20.0f / 3.0f * std::pow(q, 3)
                                            - 15.0f / 7.0f * std::pow(q, 4)
                                            + 0.25f * std::pow(q, 5)
                                            );
        }
    }
};

class WendlandC4 : public SphericalKernel {
public:
    explicit WendlandC4(int dim) : SphericalKernel(dim) {}

    const float SUPPORT = 2.0f;

    float evaluate(float q) const override {
        if (q >= support()) return 0.0f;

        float z = 1.0f - 0.5f * q;
        if (dim_ == 1)
            return std::pow(z, 5) * (2 * q * q + 2.5f * q + 1.0f);
        else
            return std::pow(z, 6) * ((35.0f / 12.0f) * q * q + 3.0f * q + 1.0f);
    }

    float support() const override { return SUPPORT; }

    float sigma() const override {
        if (dim_ == 1) return 3.0f / (4.0f);
        if (dim_ == 2) return 9.0f / (4.0f * kPi);
        if (dim_ == 3) return 495.0f / (256.0f * kPi);
    }

    float F(float q) const override {
        if (q < 0.0f) return 0.0f;
        if (q > SUPPORT) q = SUPPORT;

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
    }
};

class WendlandC6 : public SphericalKernel {
public:
    explicit WendlandC6(int dim) : SphericalKernel(dim) {}

    const float SUPPORT = 2.0f;

    float evaluate(float q) const override {
        if (q >= SUPPORT) return 0.0f;

        float z = 1.0f - 0.5f * q;
        if (dim_ == 1)
            return std::pow(z, 7) * (21.0f / 8.0f * std::pow(q, 3) + 19.0f / 4.0f * std::pow(q, 2) + 3.5f * q + 1.0f);
        else
            return std::pow(z, 8) * (4.0f * std::pow(q, 3) + 6.25f * std::pow(q, 2) + 4.0f * q + 1.0f);
    }

    float support() const override { return SUPPORT; }

    float sigma() const override {
        if (dim_ == 1) return 55.0f / (64.0f);
        if (dim_ == 2) return 39.0f / (14.0f * kPi);
        if (dim_ == 3) return 1365.0f / (512.0f * kPi);
    }

    float F(float q) const override {
        if (q < 0.0f) return 0.0f;
        if (q > SUPPORT) q = SUPPORT;

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
    }
};


std::shared_ptr<SeparableKernel> create_separable_kernel(const std::string& name, int dim) 
{
    if (name == "tophat_separable") {
        return std::make_shared<TophatSep>(dim);
    }
    else if (name == "tsc_separable") {
        return std::make_shared<TSCSep>(dim);
    }
    else if (name == "gaussian_separable") {
        return std::make_shared<GaussianSep>(dim);
    }
    throw std::invalid_argument("Unknown kernel: " + name);
}

std::shared_ptr<SphericalKernel> create_spherical_kernel(const std::string& name, int dim) {

    if (name == "tophat") {
        return std::make_shared<Tophat>(dim);
    }
    else if (name == "tsc") {
        return std::make_shared<TSC>(dim);
    }
    else if (name == "lucy") {
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
    throw std::invalid_argument("Unknown kernel: " + name);
}


SphericalKernelSampleGrid build_kernel_sample_grid(const SphericalKernel& kernel,
                                          int min_kernel_evaluations_per_axis
                                        ){
    if (min_kernel_evaluations_per_axis <= 0) {
        throw std::invalid_argument("min_kernel_evaluations_per_axis must be > 0");
    }

    // compute total number of kernel evaluations
    const int total_number_o = static_cast<int>(std::pow(min_kernel_evaluations_per_axis, kernel.dim()));

    SphericalKernelSampleGrid grid;
    grid.dim = kernel.dim();
    grid.count = total_number_o;
    grid.coords.reserve(static_cast<size_t>(total_number_o) * grid.dim);
    grid.q.reserve(total_number_o);
    grid.integrals.reserve(total_number_o);

    const float support = kernel.support();

    if (grid.dim == 1) {
        int n_q = min_kernel_evaluations_per_axis;

        const float dq = support / static_cast<float>(n_q);

        for (int iq = 0; iq < n_q; ++iq) {
            const float q0 = static_cast<float>(iq) * dq;
            const float q  = q0 + dq * 0.5f;
            const float q1 = q0 + dq;

            // this integral is now analytically evaluated and thus exact!
            // factor 2, since we integrate from [-support, + support]
            float integral = kernel.sigma() * 2.0f * kernel.evaluate_integral(q0, q1);

            grid.coords.push_back(q);
            grid.q.push_back(q);
            grid.integrals.push_back(integral);
        }
        return grid;
    }

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

    throw std::invalid_argument("SphericalKernelSampleGrid supports only dim = 1, 2 or 3");
}


float compute_total_integral_separable(const std::string& kernel_name, int dim) {
    
    auto kernel = create_separable_kernel(kernel_name, dim);
    auto support = kernel->support();
    
    std::vector<float> bounds;
    for (int d = 0; d < dim; ++d) {
        bounds.push_back(-support);
        bounds.push_back(support);
    }
    float total_integral = kernel->sigma() * kernel->evaluate_integral(bounds);
    return total_integral;
}

float compute_total_integral_spherical(const std::string& kernel_name, int dim, int min_kernel_evaluations_per_axis) {
    auto kernel = create_spherical_kernel(kernel_name, dim);
    const auto kernel_samples = build_kernel_sample_grid(*kernel, min_kernel_evaluations_per_axis);
    float total_integral = 0.0f;
    for (int s = 0; s < kernel_samples.count; ++s) {
        total_integral += kernel_samples.integrals[s];
    }
    return total_integral;
}

std::tuple<std::vector<float>, std::vector<float>> get_separable_kernel_values_1D(const std::string& kernel_name) 
{    
    auto kernel = create_separable_kernel(kernel_name, 1);
    float support = kernel->support();
    int num_samples = 100;
    float dq = 2.0f * support / static_cast<float>(num_samples);

    struct results {
        std::vector<float> q;
        std::vector<float> values;
    };
    results res;

    // setup the 1D cartesian coordinate from [-support, +support] and the corresponding kernel values
    for (int i = 0; i < num_samples; ++i) {
        float q_current = -support + i * dq;
        float kernel_value = kernel->sigma() * kernel->evaluate_1d(std::abs(q_current));
        res.q.push_back(q_current);
        res.values.push_back(kernel_value);
    } 
    return {res.q, res.values};
}

std::tuple<std::vector<float>, std::vector<float>> get_spherical_kernel_values_1D(const std::string& kernel_name) 
{    
    auto kernel = create_spherical_kernel(kernel_name, 1);
    float support = kernel->support();
    int num_samples = 100;
    float dq = 2.0f * support / static_cast<float>(num_samples);

    struct results {
        std::vector<float> q;
        std::vector<float> values;
    };
    results res;

    // setup the radial coordinate from [-support, +support] and the corresponding kernel values
    for (int i = 0; i < num_samples; ++i) {
        float q_current = -support + i * dq;
        float kernel_value = kernel->sigma() * kernel->evaluate(std::abs(q_current));
        res.q.push_back(q_current);
        res.values.push_back(kernel_value);
    } 
    return {res.q, res.values};
}