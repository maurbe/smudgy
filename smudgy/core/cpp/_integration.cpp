#include "_integration.h"

#include <stdexcept>

float integrate_cell_2d(const std::string& method,
                        const std::function<float(float, float)>& eval) {
    if (method == "midpoint") {
        return eval(0.5f, 0.5f);
    }

    if (method == "trapezoidal") {
        float sum = 0.0f;
        sum += eval(0.0f, 0.0f);
        sum += eval(1.0f, 0.0f);
        sum += eval(0.0f, 1.0f);
        sum += eval(1.0f, 1.0f);
        return (sum / 4.0f);
    }

    if (method == "simpson") {
        float sum = 0.0f;

        // corners
        sum += eval(0.0f, 0.0f);
        sum += eval(1.0f, 0.0f);
        sum += eval(0.0f, 1.0f);
        sum += eval(1.0f, 1.0f);

        // edge midpoints
        sum += 4.0f * eval(0.5f, 0.0f);
        sum += 4.0f * eval(0.5f, 1.0f);
        sum += 4.0f * eval(0.0f, 0.5f);
        sum += 4.0f * eval(1.0f, 0.5f);

        // center
        sum += 16.0f * eval(0.5f, 0.5f);

        return (sum / 36.0f);
    }

    throw std::invalid_argument("Unknown integration method: " + method);
}

float integrate_cell_3d(const std::string& method,
                        const std::function<float(float, float, float)>& eval) {
    if (method == "midpoint") {
        return eval(0.5f, 0.5f, 0.5f);
    }

    if (method == "trapezoidal") {
        float sum = 0.0f;
        for (int i = 0; i <= 1; ++i)
            for (int j = 0; j <= 1; ++j)
                for (int k = 0; k <= 1; ++k)
                    sum += eval(i, j, k);
        return (sum / 8.0f);
    }

    if (method == "simpson") {
        float sum = 0.0f;

        // corners
        for (int i = 0; i <= 1; ++i)
            for (int j = 0; j <= 1; ++j)
                for (int k = 0; k <= 1; ++k)
                    sum += eval(i, j, k);

        // edge midpoints
        for (int i = 0; i <= 1; ++i)
            for (int j = 0; j <= 1; ++j) {
                sum += 4.0f * eval(0.5f, i, j);
                sum += 4.0f * eval(i, 0.5f, j);
                sum += 4.0f * eval(i, j, 0.5f);
            }

        // face centers
        sum += 16.0f * eval(0.5f, 0.5f, 0.0f);
        sum += 16.0f * eval(0.5f, 0.5f, 1.0f);
        sum += 16.0f * eval(0.5f, 0.0f, 0.5f);
        sum += 16.0f * eval(0.5f, 1.0f, 0.5f);
        sum += 16.0f * eval(0.0f, 0.5f, 0.5f);
        sum += 16.0f * eval(1.0f, 0.5f, 0.5f);

        // center
        sum += 64.0f * eval(0.5f, 0.5f, 0.5f);

        return (sum / 216.0f);
    }

    throw std::invalid_argument("Unknown integration method: " + method);
}
