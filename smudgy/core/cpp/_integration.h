#pragma once

#include <functional>
#include <string>

float integrate_cell_2d(const std::string& method,
                        const std::function<float(float, float)>& eval);

float integrate_cell_3d(const std::string& method,
                        const std::function<float(float, float, float)>& eval);
